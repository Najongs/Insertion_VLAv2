"""
Optimal Transport Conditional Flow Matching (OT-CFM) for Action Prediction

Based on Pi0 paper: https://arxiv.org/pdf/2410.24164v1
Implementation of flow matching for continuous action prediction in robotics.

Key Concepts:
- Flow Matching: Learn a vector field v(x, t) that maps noise to data
- Optimal Transport: Use OT plan for better sample efficiency
- ODE Solver: Continuous-time formulation, discretized with Euler/RK4
- Inference: 10 steps (much faster than diffusion's 50-100 steps)

Advantages over Diffusion:
1. Faster inference (1-20 steps vs 50-100 steps)
2. Simpler training (direct velocity matching)
3. Better sample quality
4. No noise schedule tuning needed
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# =====================================
# 1️⃣ Optimal Transport Conditional Flow Matching
# =====================================

class OptimalTransportConditionalFlowMatching:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM)

    Learns a vector field v(x, t) that transports samples from noise to data
    using optimal transport paths.

    Flow ODE: dx/dt = v(x, t), x(0) ~ N(0, I), x(1) = data

    Training:
    - Sample t ~ U(0, 1)
    - Compute OT path: x_t = t * x_1 + (1-t) * x_0
    - Compute target velocity: u_t = x_1 - x_0
    - Loss: ||v(x_t, t) - u_t||^2

    Inference:
    - Start from x_0 ~ N(0, I)
    - Integrate ODE: x(1) = x(0) + ∫_0^1 v(x(t), t) dt
    - Use Euler or RK4 solver
    """

    def __init__(self, sigma_min: float = 1e-4):
        """
        Args:
            sigma_min: Minimum noise level for numerical stability
        """
        self.sigma_min = sigma_min

    def compute_flow_and_target(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the flow path and target velocity for training.

        Optimal Transport path: x_t = t * x_1 + (1 - t) * x_0
        Target velocity: u_t = x_1 - x_0 (constant velocity OT path)

        Args:
            x_1: Data samples (B, H, A)
            x_0: Source samples (B, H, A), if None sample from N(0, I)
            t: Time steps (B,) in [0, 1], if None sample uniformly

        Returns:
            x_t: Flow path samples (B, H, A)
            u_t: Target velocity (B, H, A)
            t: Time steps (B,)
        """
        B, H, A = x_1.shape
        device = x_1.device

        # Sample source from standard normal
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # Sample time uniformly
        if t is None:
            t = torch.rand(B, device=device)

        # Reshape t for broadcasting
        t = t.view(-1, 1, 1)  # (B, 1, 1)

        # Optimal Transport path (linear interpolation)
        x_t = t * x_1 + (1 - t) * x_0

        # Target velocity (constant for OT)
        u_t = x_1 - x_0

        return x_t, u_t, t.squeeze(-1).squeeze(-1)

    def sample_ode(
        self,
        velocity_model: nn.Module,
        x_0: torch.Tensor,
        num_steps: int = 10,
        method: str = 'euler',
        **model_kwargs
    ) -> torch.Tensor:
        """
        Sample from the flow by solving the ODE: dx/dt = v(x, t)

        Args:
            velocity_model: Neural network that predicts v(x, t)
            x_0: Initial noise samples (B, H, A)
            num_steps: Number of integration steps
            method: 'euler' or 'rk4'
            **model_kwargs: Additional arguments for velocity_model

        Returns:
            x_1: Final samples (B, H, A)
        """
        x = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)

            if method == 'euler':
                # Euler method: x_{i+1} = x_i + dt * v(x_i, t_i)
                with torch.no_grad():
                    v = velocity_model(x, t, **model_kwargs)
                x = x + dt * v

            elif method == 'rk4':
                # 4th order Runge-Kutta
                with torch.no_grad():
                    k1 = velocity_model(x, t, **model_kwargs)
                    k2 = velocity_model(x + 0.5 * dt * k1, t + 0.5 * dt, **model_kwargs)
                    k3 = velocity_model(x + 0.5 * dt * k2, t + 0.5 * dt, **model_kwargs)
                    k4 = velocity_model(x + dt * k3, t + dt, **model_kwargs)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown solver method: {method}")

        return x


# =====================================
# 2️⃣ Flow Matching Action Expert
# =====================================

class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching-based Action Expert with Sensor Fusion

    Learns to predict velocity field v(x, t) for action generation.
    Much faster inference than diffusion (10 steps vs 100 steps).

    Architecture:
    - Condition encoder: Fuses VL + sensor + robot_state features
    - Time embedding: Sinusoidal encoding of t
    - Velocity network: Transformer that predicts v(x_t, t)
    """

    def __init__(
        self,
        vl_dim: int = 3072,
        sensor_dim: int = 6144,  # sensor + robot_state
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 1024,
        nhead: int = 8,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        fusion_strategy: str = 'concat',
        dropout: float = 0.1,
        sigma_min: float = 1e-4
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.fusion_strategy = fusion_strategy

        # Flow matching
        self.flow = OptimalTransportConditionalFlowMatching(sigma_min=sigma_min)

        # Time embedding (sinusoidal)
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition fusion (same as regression/diffusion)
        # Note: sensor_dim already includes sensor + robot_state if both enabled
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.fusion_proj = nn.Linear(fused_dim, hidden_dim)
            print(f"   Fusion proj expects: vl_dim({vl_dim}) + sensor_dim({sensor_dim}) = {fused_dim}")
        elif fusion_strategy == 'cross_attention':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'gated':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'none':
            self.fusion_proj = nn.Linear(vl_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Velocity network (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.velocity_net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head (predict velocity)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

        print(f"✅ FlowMatchingActionExpert initialized")
        print(f"   Fusion: {fusion_strategy}")
        print(f"   OT-CFM with {num_layers} layers")

    def sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding (similar to positional encoding)

        Args:
            t: (B,) time steps in [0, 1]
        Returns:
            embed: (B, time_embed_dim)
        """
        half_dim = self.time_embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        vl_tokens: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
        robot_state_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t) given noisy actions and conditions

        Args:
            x_t: Noisy actions (B, H, A)
            t: Time steps (B,) in [0, 1]
            vl_tokens: Vision-language features (B, seq_len, vl_dim)
            sensor_features: Sensor features (B, sensor_dim)
            robot_state_features: Robot state features (B, robot_state_dim)

        Returns:
            velocity: Predicted velocity (B, H, A)
        """
        B, H, A = x_t.shape

        # Time embedding
        t_embed = self.sinusoidal_time_embedding(t)  # (B, time_embed_dim)
        t_embed = self.time_mlp(t_embed)  # (B, hidden_dim)

        # Combine sensor and robot state features
        combined_features = None
        if sensor_features is not None and robot_state_features is not None:
            combined_features = torch.cat([sensor_features, robot_state_features], dim=-1)
        elif sensor_features is not None:
            combined_features = sensor_features
        elif robot_state_features is not None:
            combined_features = robot_state_features

        # Fuse VL and combined features
        if self.fusion_strategy == 'concat' and combined_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            fused = torch.cat([vl_pooled, combined_features], dim=-1)

            # Dynamic handling of dimension mismatch
            if fused.shape[-1] != self.fusion_proj.in_features:
                # Lazily create correct projection layer
                if not hasattr(self, '_dynamic_fusion_proj') or self._dynamic_fusion_proj.in_features != fused.shape[-1]:
                    self._dynamic_fusion_proj = nn.Linear(fused.shape[-1], self.fusion_proj.out_features).to(
                        device=fused.device, dtype=fused.dtype
                    )
                    print(f"⚠️ Created dynamic fusion_proj: {fused.shape[-1]} -> {self.fusion_proj.out_features}")
                cond = self._dynamic_fusion_proj(fused)
            else:
                cond = self.fusion_proj(fused)  # (B, hidden_dim)
        elif self.fusion_strategy == 'cross_attention' and combined_features is not None:
            vl_feat = self.vl_proj(vl_tokens)
            combined_feat = self.sensor_proj(combined_features).unsqueeze(1)
            attn_out, _ = self.cross_attn(combined_feat, vl_feat, vl_feat)
            cond = self.fusion_proj(attn_out.squeeze(1))
        elif self.fusion_strategy == 'gated' and combined_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            vl_feat = self.vl_proj(vl_pooled)
            combined_feat = self.sensor_proj(combined_features)
            gate = self.gate(torch.cat([vl_feat, combined_feat], dim=-1))
            fused = gate * vl_feat + (1 - gate) * combined_feat
            cond = self.fusion_proj(fused)
        else:  # 'none' or features not provided
            vl_pooled = vl_tokens.mean(dim=1)
            cond = self.fusion_proj(vl_pooled)

        # Action embedding
        x_embed = self.action_embed(x_t)  # (B, H, hidden_dim)

        # Add time and condition to each action step
        # cond: (B, hidden_dim) -> (B, 1, hidden_dim)
        # t_embed: (B, hidden_dim) -> (B, 1, hidden_dim)
        cond_expand = (cond + t_embed).unsqueeze(1).expand(-1, H, -1)  # (B, H, hidden_dim)
        x_cond = x_embed + cond_expand  # (B, H, hidden_dim)

        # Velocity network
        velocity_feat = self.velocity_net(x_cond)  # (B, H, hidden_dim)

        # Output velocity
        velocity = self.output_head(velocity_feat)  # (B, H, A)

        return velocity

    def compute_loss(
        self,
        actions: torch.Tensor,
        vl_tokens: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
        robot_state_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching training loss

        Args:
            actions: Ground truth actions (B, H, A)
            vl_tokens: Vision-language features (B, seq_len, vl_dim)
            sensor_features: Sensor features (B, sensor_dim)
            robot_state_features: Robot state features (B, robot_state_dim)

        Returns:
            loss: MSE between predicted and target velocity
        """
        # Compute flow path and target velocity
        x_t, u_t, t = self.flow.compute_flow_and_target(actions)

        # Predict velocity
        v_pred = self.forward(x_t, t, vl_tokens, sensor_features, robot_state_features)

        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, u_t)

        return loss

    @torch.no_grad()
    def sample(
        self,
        vl_tokens: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
        robot_state_features: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_steps: int = 10,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Sample actions by solving the flow ODE

        Args:
            vl_tokens: Vision-language features (B, seq_len, vl_dim)
            sensor_features: Sensor features (B, sensor_dim)
            robot_state_features: Robot state features (B, robot_state_dim)
            batch_size: Batch size (if None, use vl_tokens.shape[0])
            num_steps: Number of ODE integration steps
            method: 'euler' or 'rk4'

        Returns:
            actions: Sampled actions (B, H, A)
        """
        if batch_size is None:
            batch_size = vl_tokens.shape[0]

        device = vl_tokens.device

        # Start from noise
        x_0 = torch.randn(batch_size, self.horizon, self.action_dim, device=device)

        # Solve ODE
        def velocity_fn(x, t):
            return self.forward(x, t, vl_tokens, sensor_features, robot_state_features)

        actions = self.flow.sample_ode(
            velocity_fn,
            x_0,
            num_steps=num_steps,
            method=method
        )

        return actions
