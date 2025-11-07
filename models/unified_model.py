"""
Unified Vision-Language-Action Model with Sensor Integration

통합된 VLA 모델 파일로 Diffusion과 Regression 모두 지원

Key Components:
1. SensorEncoder: OCT/FPI 센서 데이터 처리 (650/65 timesteps)
2. ActionExpert (Regression): 직접 회귀 기반 행동 예측
3. DiffusionActionExpert: DDPM 기반 행동 예측
4. QwenVLAUnified: Diffusion/Regression을 선택 가능한 통합 모델

Usage:
    # Regression model
    model = QwenVLAUnified(model_type='regression', sensor_enabled=True)

    # Diffusion model
    model = QwenVLAUnified(model_type='diffusion', diffusion_timesteps=100)
"""

import os
from pathlib import Path
import hashlib
import fcntl
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

# Import Flow Matching
from models.flow_matching import FlowMatchingActionExpert





class RobotStateEncoder(nn.Module):
    """
    Transformer-based encoder for robot state sequences (joint angles + pose).
    Now includes temporal pooling and projection for a fixed-size output, similar to SensorEncoder.
    """
    def __init__(self, 
                 input_dim: int = 12, 
                 model_dim: int = 256,
                 output_dim: int = 2048,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 temporal_length: int = 60,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim

        # 1. Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)

        # 2. Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, temporal_length, model_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True, # Important
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Temporal Pooling and Projection Head
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Robot state sequence, shape (B, T, D_in)
            return_sequence (bool): If True, return the full output sequence from the transformer.
                                    Otherwise, return the pooled and projected feature vector.
        
        Returns:
            torch.Tensor: Encoded features. Shape is (B, D_out) if return_sequence is False,
                          and (B, T, model_dim) if return_sequence is True.
        """
        # Project input and add positional encoding
        x = self.input_proj(src) # (B, T, model_dim)
        x = x + self.pos_encoder
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer_encoder(x) # (B, T, model_dim)

        # Conditionally return the sequence for MAE pre-training
        if return_sequence:
            return x

        # Pool and project for downstream tasks
        pooled_x = x.transpose(1, 2) # (B, model_dim, T)
        pooled_x = self.temporal_pool(pooled_x).squeeze(-1) # (B, model_dim)
        output_features = self.projection(pooled_x) # (B, output_dim)
        
        return output_features



# =====================================
# 2️⃣ Sensor Encoder Module
# =====================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.size(-1) ** 0.5)
        return self.weight * (x / (rms + self.eps))

class ResidualDownsample1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, dropout=0.1, bn_fp32=True):
        super().__init__()
        self.bn_fp32 = bn_fp32

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.GELU()
        self.do1   = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.GELU()

        self.skip  = nn.Identity() if (in_ch == out_ch and stride == 1) else \
                     nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

        # ▼ BN을 FP32로 고정
        if self.bn_fp32:
            self.bn1.float()
            self.bn2.float()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.do1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        s = self.skip(x)
        return y + s


def force_bn_fp32_(module: torch.nn.Module):
    """Cast all BatchNorm params/buffers in `module` to float32."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()  # weights/bias & running stats 모두 FP32로


# =====================================
# Force-Aware Sensor Encoder
# =====================================
class ForceAwareSensorEncoder(nn.Module):
    """
    Sensor encoder that processes 'distance' (1025 channels) and 'force' (1 channel)
    features separately to give more weight to the force data.

    Architecture:
    1.  The first 1025 channels ('distance') are processed by a standard SensorEncoder.
    2.  The last 1 channel ('force') is processed by a dedicated MLP.
    3.  The outputs from both are concatenated and projected to the final output dimension.
    """
    def __init__(self,
                 dist_channels=1025,
                 force_channels=1,
                 temporal_length=65,
                 dist_hidden_dim=512,
                 force_hidden_dim=128,
                 output_dim=3072,
                 **kwargs):
        super().__init__()
        self.input_channels = dist_channels + force_channels

        print(f"🚀 Initializing ForceAwareSensorEncoder:")
        print(f"   - Distance features (1-{dist_channels}) processed by a full ConvFormer.")
        print(f"   - Force features ({dist_channels+1}) processed by a dedicated MLP.")

        # Encoder for the main 'distance' features
        self.dist_encoder = SensorEncoder(
            input_channels=dist_channels,
            temporal_length=temporal_length,
            hidden_dim=dist_hidden_dim,
            output_dim=output_dim - force_hidden_dim, # Allocate part of the output space
            **kwargs
        )
        # Ensure BatchNorm layers within dist_encoder are float32
        force_bn_fp32_(self.dist_encoder)

        # A smaller, dedicated MLP for the 'force' feature
        self.force_encoder = nn.Sequential(
            nn.Linear(force_channels, force_hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(force_hidden_dim // 2),
            nn.Linear(force_hidden_dim // 2, force_hidden_dim)
        )
        self.force_pool = nn.AdaptiveAvgPool1d(1)

        print(f"   - Final output: Concatenated and projected to {output_dim} dims.")


    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (B, T, C) where C is self.input_channels (1026)
        Returns:
            features: (B, output_dim)
        """
        B, T, C = sensor_data.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {C}")

        # Split data into distance and force
        dist_data = sensor_data[..., :-1]  # (B, T, 1025)
        force_data = sensor_data[..., -1:] # (B, T, 1)

        # 1. Process distance data through the main encoder
        dist_features = self.dist_encoder(dist_data) # (B, output_dim - force_hidden_dim)

        # 2. Process force data through the dedicated MLP
        #    Input to MLP is (B, T, 1)
        force_features_temporal = self.force_encoder(force_data) # (B, T, force_hidden_dim)
        #    Pool across time dimension
        force_features_pooled = self.force_pool(force_features_temporal.transpose(1, 2)).squeeze(-1) # (B, force_hidden_dim)

        # 3. Concatenate and return
        combined_features = torch.cat([dist_features, force_features_pooled], dim=-1)

        return combined_features


# =====================================
# Improved Sensor Encoder (Temporal ConvFormer)
# =====================================
class SensorEncoder(nn.Module):
    def __init__(self,
                 input_channels=1026,
                 temporal_length=650,
                 hidden_dim=512,
                 output_dim=3072,
                 num_conv_layers=4,
                 use_transformer=True,
                 num_transformer_layers=2,
                 nhead=8,
                 dropout=0.1,
                 gradient_checkpointing=False,
                 interpolation_mode='linear'):
        super().__init__()
        self.input_channels = input_channels
        self.temporal_length = temporal_length
        self.output_dim = output_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.interpolation_mode = interpolation_mode

        # ▼ 잔차 다운샘플 블록 스택
        chs = [input_channels]
        conv_blocks = []
        for i in range(num_conv_layers):
            out_ch = hidden_dim if i == 0 else hidden_dim * 2  # 1026→512→1024→1024 …
            conv_blocks.append(ResidualDownsample1d(
                in_ch=chs[-1], out_ch=out_ch, stride=2, dropout=dropout, bn_fp32=True
            ))
            chs.append(out_ch)
        self.conv_backbone = nn.ModuleList(conv_blocks)
        self.final_channels = chs[-1]

        # 최종 길이(대략 반씩 줄어듦)
        self.final_temporal_length = (temporal_length + (1 << num_conv_layers) - 1) // (1 << num_conv_layers)

        self.use_transformer = use_transformer
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.final_channels, nhead=nhead,
                dim_feedforward=self.final_channels * 4,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(self.final_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        # sensor_data: (B,T,C)
        B, T, C = sensor_data.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {C}")

        # 비동기 길이 보정 (interpolate)
        if T != self.temporal_length:
            x = sensor_data.transpose(1, 2)  # (B,C,T)
            mode = 'linear' if self.interpolation_mode == 'cubic' and T < 4 else self.interpolation_mode
            x = F.interpolate(x, size=self.temporal_length, mode=mode,
                              align_corners=False if mode in ('linear', 'cubic') else None)
        else:
            x = sensor_data.transpose(1, 2)  # (B,C,T)

        # Conv 스택
        for block in self.conv_backbone:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)   # (B, ch, T/2)

        # Transformer (시간축 first로 바꿔 처리)
        if self.use_transformer:
            x = x.transpose(1, 2)  # (B, T', ch)
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)
            x = x.transpose(1, 2)  # (B, ch, T')

        # 시계열 풀링 → 투영
        x = self.temporal_pool(x).squeeze(-1)        # (B, ch)
        sensor_features = self.projection(x)         # (B, output_dim)
        return sensor_features

# =====================================
# 2️⃣ Flow Matching Components
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
        velocity_model,
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
                v = velocity_model(x, t, **model_kwargs)
                x = x + dt * v

            elif method == 'rk4':
                # 4th order Runge-Kutta
                k1 = velocity_model(x, t, **model_kwargs)
                k2 = velocity_model(x + 0.5 * dt * k1, t + 0.5 * dt, **model_kwargs)
                k3 = velocity_model(x + 0.5 * dt * k2, t + 0.5 * dt, **model_kwargs)
                k4 = velocity_model(x + dt * k3, t + dt, **model_kwargs)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown solver method: {method}")

        return x


class DiffusionSchedule:
    """Manages diffusion noise schedule and sampling"""
    def __init__(self, timesteps=100, schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device

        # Compute betas and alphas
        betas = cosine_beta_schedule(timesteps) if schedule == 'cosine' else None
        if betas is None:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Forward diffusion
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Reverse diffusion
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def p_mean_variance(self, x_t, t, eps_pred):
        """Compute mean and variance for p(x_{t-1} | x_t)"""
        pred_x0 = self.predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        model_mean = (
            self.sqrt_recip_alphas[t].view(-1, 1, 1) *
            (x_t - self.betas[t].view(-1, 1, 1) * eps_pred /
             self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1))
        )

        model_variance = self.posterior_variance[t].view(-1, 1, 1)
        model_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1)

        return model_mean, model_variance, model_log_variance, pred_x0


class DiffusionActionExpert(nn.Module):
    """
    Diffusion-based Action Expert with Sensor Fusion

    Architecture:
    - Condition encoder: Fuses VL + sensor features
    - Noise predictor: Temporal transformer model
    - Training: Predict noise at random timestep
    - Inference: Iterative denoising (DDPM/DDIM)
    """
    def __init__(self,
                 vl_dim=3072,
                 sensor_dim=3072,
                 action_dim=7,
                 horizon=8,
                 hidden_dim=512,
                 timesteps=100,
                 fusion_strategy='concat',
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.timesteps = timesteps
        self.fusion_strategy = fusion_strategy

        # Diffusion schedule
        self.diffusion = DiffusionSchedule(timesteps=timesteps, schedule='cosine')

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition encoder (VL + Sensor fusion)
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.cond_proj = nn.Linear(fused_dim, hidden_dim)
        elif fusion_strategy == 'cross_attention':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'gated':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'none':
            self.cond_proj = nn.Linear(vl_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head (predict noise)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # === Dual output heads (ε_pred) ===
        self.trans_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)  # dx, dy, dz
        )
        self.rot_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)  # dα, dβ, dγ
        )
        self.grip_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # gripper
        )

        print(f"✅ DiffusionActionExpert initialized:")
        print(f"   Timesteps: {timesteps}, Fusion: {fusion_strategy}")
        print(f"   Action shape: (B, {horizon}, {action_dim})")

    def forward(self, noisy_actions, timesteps, vl_tokens, sensor_features=None):
        """
        Forward pass for training

        Args:
            noisy_actions: (B, H, A) - actions with noise
            timesteps: (B,) - diffusion timestep
            vl_tokens: (B, seq_len, vl_dim) or (B, 1, vl_dim)
            sensor_features: (B, sensor_dim) - optional

        Returns:
            eps_pred: (B, H, A) - predicted noise
        """
        B, H, A = noisy_actions.shape

        # Timestep embedding
        t_embed = self.timestep_embedding(timesteps, dtype=noisy_actions.dtype)
        t_embed = self.time_embed(t_embed)
        t_embed = t_embed.unsqueeze(1).expand(-1, H, -1)

        # Condition embedding
        cond_embed = self._encode_condition(vl_tokens, sensor_features)
        cond_embed = cond_embed.unsqueeze(1).expand(-1, H, -1)

        # Action embedding
        action_embed = self.action_embed(noisy_actions)

        # Combine embeddings
        x = t_embed + cond_embed + action_embed

        # Temporal processing
        x = self.temporal_encoder(x)

        # Predict noise
        eps_trans = self.trans_head(x)
        eps_rot = self.rot_head(x)
        eps_grip = self.grip_head(x)

        eps_pred = torch.cat([eps_trans, eps_rot, eps_grip], dim=-1)

        return eps_pred

    def _encode_condition(self, vl_tokens, sensor_features):
        """Encode and fuse VL + Sensor features"""
        if self.fusion_strategy == 'concat' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            fused = torch.cat([vl_pooled, sensor_features], dim=-1)
            cond = self.cond_proj(fused)

        elif self.fusion_strategy == 'cross_attention' and sensor_features is not None:
            vl_feat = self.vl_proj(vl_tokens)
            sensor_feat = self.sensor_proj(sensor_features).unsqueeze(1)
            attn_out, _ = self.cross_attn(sensor_feat, vl_feat, vl_feat)
            cond = self.cond_proj(attn_out.squeeze(1))

        elif self.fusion_strategy == 'gated' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            vl_feat = self.vl_proj(vl_pooled)
            sensor_feat = self.sensor_proj(sensor_features)
            gate = self.gate(torch.cat([vl_feat, sensor_feat], dim=-1))
            fused = gate * vl_feat + (1 - gate) * sensor_feat
            cond = self.cond_proj(fused)

        else:  # 'none' or sensor not provided
            vl_pooled = vl_tokens.mean(dim=1)
            cond = self.cond_proj(vl_pooled)

        return cond

    def timestep_embedding(self, timesteps, dim=128, dtype=torch.float32):
        """Sinusoidal timestep embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.to(dtype=dtype)

    @torch.no_grad()
    def sample(self, vl_tokens, sensor_features=None, batch_size=1, ddim_steps=None):
        """
        Sample actions via iterative denoising (DDPM or DDIM)

        Args:
            vl_tokens: (B, seq_len, vl_dim)
            sensor_features: (B, sensor_dim)
            batch_size: Number of samples
            ddim_steps: If provided, use DDIM sampling (faster)

        Returns:
            actions: (B, H, A) - denoised actions
        """
        device = vl_tokens.device
        dtype = vl_tokens.dtype
        H, A = self.horizon, self.action_dim

        # Start from pure noise
        x = torch.randn(batch_size, H, A, device=device, dtype=dtype)

        # DDIM sampling (faster)
        if ddim_steps is not None:
            return self._ddim_sample(x, vl_tokens, sensor_features, ddim_steps)

        # DDPM sampling (full)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)
            mean, variance, log_variance, _ = self.diffusion.p_mean_variance(x, t_batch, eps_pred)

            noise = torch.randn_like(x) if t > 0 else 0.0
            x = mean + torch.sqrt(variance) * noise

        return x

    @torch.no_grad()
    def _ddim_sample(self, x, vl_tokens, sensor_features, ddim_steps):
        """DDIM sampling (faster, deterministic)"""
        device = x.device
        batch_size = x.shape[0]

        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[:ddim_steps]
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)
            pred_x0 = self.diffusion.predict_x0_from_eps(x, t_batch, eps_pred)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.diffusion.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)

            alpha_t = self.diffusion.alphas_cumprod[t]

            sigma_t = 0.0  # eta=0 for deterministic
            x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next - sigma_t**2) * eps_pred

        return x


# =====================================
# 3️⃣ Regression Action Expert
# =====================================
class RegressionActionExpert(nn.Module):
    """
    Regression-based Action Expert with Sensor Fusion

    Fusion Strategies:
    - 'concat': Concatenate VL and sensor features
    - 'cross_attention': Cross-attention between VL and sensor
    - 'gated': Gated fusion with learned gates
    - 'none': Use only VL features
    """
    def __init__(self,
                 vl_dim=3072,
                 sensor_dim=3072,
                 action_dim=7,
                 horizon=8,
                 hidden_dim=1024,
                 nhead=8,
                 num_layers=4,
                 fusion_strategy='concat',
                 dropout=0.1):
        super().__init__()

        self.horizon = horizon
        self.fusion_strategy = fusion_strategy

        # Fusion layer
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.fusion_proj = nn.Linear(fused_dim, hidden_dim)
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

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        # Temporal decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        self.trans_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)  # dx, dy, dz
        )
        self.rot_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)  # dα, dβ, dγ
        )
        self.grip_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # gripper
        )

        print(f"✅ RegressionActionExpert initialized with fusion: {fusion_strategy}")

    def forward(self, vl_tokens, z_chunk, sensor_features=None, robot_state_features=None):
        """
        Args:
            vl_tokens: (B, 1, vl_dim) or (B, seq_len, vl_dim)
            z_chunk: (B, H, action_dim) - action chunks
            sensor_features: (B, sensor_dim) - optional
            robot_state_features: (B, robot_state_dim) - optional

        Returns:
            pred_actions: (B, H, action_dim)
            delta: (B, H, action_dim)
        """
        B, H, A = z_chunk.shape

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
            cond = self.fusion_proj(fused).unsqueeze(1)

        elif self.fusion_strategy == 'cross_attention' and combined_features is not None:
            vl_feat = self.vl_proj(vl_tokens)
            combined_feat = self.sensor_proj(combined_features).unsqueeze(1)
            attn_out, _ = self.cross_attn(combined_feat, vl_feat, vl_feat)
            cond = self.fusion_proj(attn_out)

        elif self.fusion_strategy == 'gated' and combined_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            vl_feat = self.vl_proj(vl_pooled)
            combined_feat = self.sensor_proj(combined_features)
            gate = self.gate(torch.cat([vl_feat, combined_feat], dim=-1))
            fused = gate * vl_feat + (1 - gate) * combined_feat
            cond = self.fusion_proj(fused).unsqueeze(1)

        else:  # 'none' or features not provided
            vl_pooled = vl_tokens.mean(dim=1, keepdim=True)
            cond = self.fusion_proj(vl_pooled)

        # Temporal decoding
        tgt = self.pos_embed.repeat(B, 1, 1)
        decoded = self.temporal_decoder(tgt, cond)

        # Predict action deltas
        delta_trans = self.trans_head(decoded)
        delta_rot = self.rot_head(decoded)
        delta_grip = self.grip_head(decoded)
        delta = torch.cat([delta_trans, delta_rot, delta_grip], dim=-1)
        pred_actions = z_chunk + delta

        return pred_actions, delta


# =====================================
# 4️⃣ Unified VLA Model
# =====================================
class QwenVLAUnified(nn.Module):
    """
    통합 Vision-Language-Action 모델

    Features:
    - Frozen Qwen-VL backbone with caching
    - Trainable Sensor Encoder (optional)
    - Trainable Action Expert (Diffusion or Regression)
    - Supports LoRA fine-tuning for VL model

    Args:
        model_type: 'diffusion', 'regression', or 'flow_matching'
        vl_model_name: Qwen model name
        action_dim: Action dimension (7)
        horizon: Action prediction horizon (8)
        hidden_dim: Hidden dimension for action expert

        # Sensor params
        sensor_enabled: Enable sensor encoder
        sensor_temporal_length: 650 (full) or 65 (async)
        fusion_strategy: 'concat', 'cross_attention', 'gated', 'none'

        # Diffusion params (only for model_type='diffusion')
        diffusion_timesteps: Number of diffusion steps

        # Flow matching params (only for model_type='flow_matching')
        flow_steps: Number of ODE integration steps (default: 10)

        # LoRA params (optional VL fine-tuning)
        finetune_vl: 'none', 'lora', 'full'
        lora_r: LoRA rank

        # Image resize (for faster inference)
        image_resize_height: e.g., 360
        image_resize_width: e.g., 640
    """
    def __init__(self,
                # Model type
                model_type: Literal['diffusion', 'regression', 'flow_matching'] = 'flow_matching',

                # Base params
                vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                action_dim=7,
                horizon=8,
                hidden_dim=1024,
                cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",

                # Sensor encoder params
                sensor_enabled=True,
                sensor_encoder_type: Literal['default', 'force_aware'] = 'default',
                sensor_input_channels=1026,
                sensor_temporal_length=650,
                sensor_hidden_dim=512,
                sensor_output_dim=3072,

                # Robot state encoder params
                robot_state_enabled=True,  # Enable robot state input (joint + pose)
                robot_state_temporal_length=100,  # Temporal window for robot state (100 samples @ 100Hz = 1s)

                # Fusion params
                fusion_strategy='concat',

                # Diffusion params
                diffusion_timesteps=100,

                # Flow matching params
                flow_steps=10,
                flow_solver='euler',  # 'euler' or 'rk4'

                # LoRA params
                finetune_vl='none',
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,

                # Image resize params
                image_resize_height=None,
                image_resize_width=None,

                # VL optimization params
                parallel_view_encoding=False,  # Enable multi-view parallel encoding
                view_aggregation='mean',  # 'mean', 'max', or 'attention'

                # Device map
                device_map=None):
        super().__init__()

        if model_type not in ['diffusion', 'regression', 'flow_matching']:
            raise ValueError(f"model_type must be 'diffusion', 'regression', or 'flow_matching', got {model_type}")

        self.model_type = model_type
        self.sensor_enabled = sensor_enabled
        self.robot_state_enabled = robot_state_enabled
        self.flow_steps = flow_steps
        self.flow_solver = flow_solver
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True
        self.cache_limit_gb = 20.0
        self.strict_cache = False
        self.action_dim = action_dim
        self.horizon = horizon
        self.parallel_view_encoding = parallel_view_encoding  # ✅ New optimization flag
        self.view_aggregation = view_aggregation

        print(f"🚀 Loading QwenVLA Unified Model")
        print(f"   Model Type: {model_type.upper()}")
        print(f"   Sensor Enabled: {sensor_enabled}")
        print(f"   Robot State Enabled: {robot_state_enabled}")
        print(f"   Fusion Strategy: {fusion_strategy}")
        if model_type == 'diffusion':
            print(f"   Diffusion Timesteps: {diffusion_timesteps}")
        elif model_type == 'flow_matching':
            print(f"   Flow Steps: {flow_steps}, Solver: {flow_solver}")

        # VL Model
        self.processor = AutoProcessor.from_pretrained(vl_model_name)

        # Image resize configuration
        if image_resize_height and image_resize_width:
            target_pixels = image_resize_height * image_resize_width
            self.processor.image_processor.min_pixels = target_pixels
            self.processor.image_processor.max_pixels = target_pixels
            print(f"   Image resize: {image_resize_width}x{image_resize_height}")

        # VL optimization configuration
        if parallel_view_encoding:
            print(f"   ⚡ Parallel View Encoding: ENABLED (aggregation={view_aggregation})")
        else:
            print(f"   Sequential View Encoding (default)")

        self.vl_model = self._load_qwen_with_fallback(vl_model_name, device_map)

        # Print actual VL hidden size
        print(f"   VL Model hidden_size: {self.vl_model.config.hidden_size}")

        # Apply LoRA or freeze VL model
        if finetune_vl == 'lora':
            print(f"🔧 Applying LoRA to VL model (r={lora_r})...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.vl_model = get_peft_model(self.vl_model, lora_config)
            print("✅ LoRA applied. LoRA parameters are trainable.")
        elif finetune_vl == 'none':
            print("🧊 Freezing VL model parameters...")
            for p in self.vl_model.parameters():
                p.requires_grad = False
            print("✅ VL Model frozen.")
        else:  # 'full'
            print("🔥 VL model is fully trainable.")

        # Sensor Encoder
        if sensor_enabled:
            if sensor_encoder_type == 'force_aware':
                print("   Sensor Encoder Type: Force-Aware")
                self.sensor_encoder = ForceAwareSensorEncoder(
                    dist_channels=sensor_input_channels - 1,
                    force_channels=1,
                    temporal_length=sensor_temporal_length,
                    dist_hidden_dim=sensor_hidden_dim,
                    force_hidden_dim=128,
                    output_dim=sensor_output_dim,
                    use_transformer=True,
                    num_transformer_layers=2
                ).to(dtype=torch.bfloat16, device="cuda")
            else:  # 'default'
                print("   Sensor Encoder Type: Default")
                self.sensor_encoder = SensorEncoder(
                    input_channels=sensor_input_channels,
                    temporal_length=sensor_temporal_length,
                    hidden_dim=sensor_hidden_dim,
                    output_dim=sensor_output_dim,
                    use_transformer=True,
                    num_transformer_layers=2
                ).to(dtype=torch.bfloat16, device="cuda")

            force_bn_fp32_(self.sensor_encoder)

        else:
            self.sensor_encoder = None

        # Robot State Encoder (joint + pose: 12 dims)
        # Uses dedicated MLP-based encoder optimized for robot state
        if self.robot_state_enabled:
            self.robot_state_encoder = RobotStateEncoder(
                input_dim=12,  # 6 joints + 6 poses
                temporal_length=robot_state_temporal_length,  # Use robot state temporal length
                model_dim=256,  # Smaller than sensor (robot state is simpler)
                output_dim=sensor_output_dim,  # Same output dim as sensor for fusion
                num_layers=3,
                num_heads=8,
                dropout=0.1
            ).to(dtype=torch.bfloat16, device="cuda")
        else:
            self.robot_state_encoder = None

        # Action Expert (Diffusion, Regression, or Flow Matching)
        # Combined feature dim: sensor + robot_state (both use same output_dim)
        combined_feature_dim = sensor_output_dim * 2 if (sensor_enabled and self.robot_state_enabled) else sensor_output_dim if sensor_enabled else 0

        if model_type == 'diffusion':
            raise ValueError(
                "Diffusion model is deprecated. Please use 'flow_matching' or 'regression' instead.\n"
                "Flow matching provides faster inference (10 steps vs 100 steps) and better performance."
            )
        elif model_type == 'flow_matching':
            self.action_expert = FlowMatchingActionExpert(
                vl_dim=self.vl_model.config.hidden_size,
                sensor_dim=combined_feature_dim,  # Now includes sensor + robot_state
                action_dim=action_dim,
                horizon=horizon,
                hidden_dim=hidden_dim,
                fusion_strategy=fusion_strategy if sensor_enabled else 'none'
            ).to(dtype=torch.bfloat16, device="cuda")
        else:  # regression
            self.action_expert = RegressionActionExpert(
                vl_dim=self.vl_model.config.hidden_size,
                sensor_dim=combined_feature_dim,  # Now includes sensor + robot_state
                action_dim=action_dim,
                horizon=horizon,
                hidden_dim=hidden_dim,
                fusion_strategy=fusion_strategy if sensor_enabled else 'none'
            ).to(dtype=torch.bfloat16, device="cuda")

        print("✅ Model initialization complete!")

    def _load_qwen_with_fallback(self, vl_model_name, device_map):
        """Load Qwen-VL with fallback"""
        dtype_candidates = [torch.bfloat16, torch.float16]
        attn_candidates = ["flash_attention_2", "sdpa"]

        for dtype in dtype_candidates:
            for impl in attn_candidates:
                try:
                    print(f"🧠 Trying attn={impl}, dtype={dtype}...")
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vl_model_name,
                        torch_dtype=dtype,
                        attn_implementation=impl,
                        device_map=device_map or "cuda",
                        low_cpu_mem_usage=True,
                    )
                    print(f"✅ Loaded with {impl} ({dtype})")
                    self.attn_backend = impl
                    self.model_dtype = dtype
                    return model
                except Exception as e:
                    print(f"⚠️ {impl} ({dtype}) failed: {e}")

        # Final fallback
        for dtype in dtype_candidates:
            try:
                print(f"🧠 Trying default attention, dtype={dtype}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vl_model_name,
                    torch_dtype=dtype,
                    device_map=device_map or "cuda",
                    low_cpu_mem_usage=True,
                )
                print(f"✅ Loaded with default attention ({dtype})")
                self.attn_backend = "default"
                self.model_dtype = dtype
                return model
            except Exception as e:
                print(f"⚠️ Default ({dtype}) failed: {e}")

        raise RuntimeError("❌ All fallback attempts failed.")

    def set_cache(self, enabled: bool = True):
        self.cache_enabled = enabled

    def set_strict_cache(self, enabled: bool = True):
        self.strict_cache = enabled

    def set_cache_limit(self, limit_gb: float):
        self.cache_limit_gb = float(limit_gb)

    def _cache_path(self, key: str, txt: str, views: list) -> Path:
        vlist = [v for v in views if v is not None] if views is not None else []
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    def _prepare_cached_vl_tokens(self, cached_batch, device: torch.device):
        """
        Convert dataloader-provided VL cache tensors (list or tensor) into a single batch tensor.
        Returns None if any item is missing so the caller can fall back to live encoding.
        """
        if cached_batch is None:
            return None

        if isinstance(cached_batch, torch.Tensor):
            tensors = [cached_batch]
        elif isinstance(cached_batch, (list, tuple)):
            tensors = []
            for item in cached_batch:
                if item is None:
                    return None
                if not isinstance(item, torch.Tensor):
                    return None
                tensors.append(item)
        else:
            return None

        if not tensors:
            return None

        target_dtype = getattr(self, "model_dtype", torch.bfloat16)
        prepared = []
        for tensor in tensors:
            t = tensor
            if t.ndim == 1:
                t = t.view(1, 1, -1)
            elif t.ndim == 2:
                t = t.unsqueeze(0)
            prepared.append(t.to(dtype=target_dtype))

        batch_tensor = torch.cat(prepared, dim=0)
        return batch_tensor.to(device=device, non_blocking=True)

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
        tmp = path.with_suffix(".pt.tmp")
        with open(str(path) + ".lock", "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                if path.exists():
                    return
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self):
        limit_gb = getattr(self, "cache_limit_gb", 20.0)
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > limit_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > limit_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)

    def encode_vision(self, text_inputs, image_inputs, cache_keys, cache: bool = True):
        """Encode vision features (for cache building)"""
        device = next(self.vl_model.parameters()).device
        dummy = torch.zeros(len(text_inputs), self.horizon, self.action_dim,
                          device=device, dtype=self.model_dtype)

        with torch.inference_mode():
            if self.model_type == 'diffusion':
                raise ValueError("Diffusion model is deprecated. Use 'flow_matching' or 'regression' instead.")
            elif self.model_type in ['flow_matching', 'regression']:
                # For flow_matching/regression, pass actions=None for inference mode
                self(text_inputs=text_inputs,
                     image_inputs=image_inputs,
                     actions=None,
                     sensor_data=None,
                     cache_keys=cache_keys,
                     cache=cache)
            else:
                self(text_inputs=text_inputs,
                     image_inputs=image_inputs,
                     z_chunk=dummy,
                     sensor_data=None,
                     cache_keys=cache_keys,
                     cache=cache)

    def forward(self,
                text_inputs,
                image_inputs,
                actions=None,  # For diffusion training
                z_chunk=None,  # For regression
                sensor_data=None,
                robot_states=None,  # NEW: Robot state data (joint + pose)
                cache_keys=None,
                cache: bool = True,
                vl_cache_tokens=None):
        """
        Forward pass for both diffusion and regression

        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths
            actions: (B, H, A) - for diffusion training
            z_chunk: (B, H, A) - for regression
            sensor_data: (B, T, C) - sensor data
            robot_states: (B, T, 12) - robot state data (6 joints + 6 poses)
            cache_keys: Cache keys for VL features
            cache: Whether to use caching

        Returns:
            Diffusion training: eps_pred, eps_target, timesteps
            Diffusion inference: sampled_actions
            Regression: pred_actions, delta
        """
        device = next(self.parameters()).device

        # Try to leverage dataloader-provided VL cache tensors first
        vl_tokens = self._prepare_cached_vl_tokens(vl_cache_tokens, device)
        if vl_tokens is not None and not hasattr(self, "_external_cache_confirmed"):
            print("💾 Using dataloader-provided VL cache tensors.")
            self._external_cache_confirmed = True

        if vl_tokens is None:
            use_cache = bool(cache and self.cache_enabled)
            if cache_keys is None:
                cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

            # Encode VL features (live path)
            vl_tokens = self._encode_vision_features(text_inputs, image_inputs, cache_keys, use_cache, device)

        # Encode sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)

        # Encode robot state features
        robot_state_features = None
        if self.robot_state_enabled and robot_states is not None:
            robot_states = robot_states.to(device=device, dtype=torch.bfloat16)
            robot_state_features = self.robot_state_encoder(robot_states)
        elif self.robot_state_enabled and robot_states is None:
            # Debug: robot states expected but not provided
            import warnings
            warnings.warn("robot_state_enabled=True but robot_states is None!")

        # Model-specific forward
        if self.model_type == 'diffusion':
            raise ValueError("Diffusion model is deprecated. Use 'flow_matching' or 'regression' instead.")

        elif self.model_type == 'flow_matching':
            # Training mode: compute flow matching loss
            if actions is not None and self.training:
                actions = actions.to(device=device, dtype=vl_tokens.dtype)

                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = self.action_expert.compute_loss(
                        actions, vl_tokens, sensor_features, robot_state_features
                    )

                # Return loss in same format as diffusion for compatibility
                # We'll handle this specially in training loop
                return loss, None, None

            # Inference mode: sample actions via ODE
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    sampled_actions = self.action_expert.sample(
                        vl_tokens,
                        sensor_features,
                        robot_state_features,
                        batch_size=vl_tokens.shape[0],
                        num_steps=self.flow_steps,
                        method=self.flow_solver
                    )
                # Return in same format as training for compatibility with validation
                return sampled_actions, None, None

        else:  # regression
            z_chunk = z_chunk.to(device=device, dtype=vl_tokens.dtype)
            pred_actions, delta = self.action_expert(vl_tokens, z_chunk, sensor_features, robot_state_features)
            return pred_actions, delta
        
    def _encode_vision_features(self, text_inputs, image_inputs, cache_keys, use_cache, device):
        """
        Encode VL features with optional parallel encoding

        Routes to parallel or sequential encoding based on self.parallel_view_encoding
        """
        # Use parallel encoding if enabled AND not using cache
        if self.parallel_view_encoding and not use_cache:
            # Debug: print once to confirm parallel encoding is used
            if not hasattr(self, '_parallel_encoding_confirmed'):
                print("🚀 Using PARALLEL view encoding for faster VL updates!")
                self._parallel_encoding_confirmed = True
            return self._encode_vision_features_parallel(text_inputs, image_inputs, device)

        # Otherwise use sequential encoding (with caching support)
        if not hasattr(self, '_sequential_encoding_confirmed'):
            print("📝 Using SEQUENTIAL view encoding")
            self._sequential_encoding_confirmed = True
        return self._encode_vision_features_sequential(text_inputs, image_inputs, cache_keys, use_cache, device)

    def _encode_vision_features_sequential(self, text_inputs, image_inputs, cache_keys, use_cache, device):
        """Sequential encoding (robust version with None handling)"""
        pooled_vl_tokens_dict = {}
        miss_items = []

        # ✅ 1. None-safe handling
        if text_inputs is None:
            text_inputs = []
        if image_inputs is None:
            image_inputs = [None] * len(text_inputs)
        if cache_keys is None:
            cache_keys = [f"seq_{i}" for i in range(len(text_inputs))]

        n = min(len(text_inputs), len(image_inputs), len(cache_keys))
        text_inputs = text_inputs[:n]
        image_inputs = image_inputs[:n]
        cache_keys = cache_keys[:n]

        # ✅ 2. Cache handling
        if use_cache:
            for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
                cache_path = self._cache_path(key, txt, views)
                if cache_path.exists():
                    pooled = torch.load(cache_path, map_location="cpu")
                    pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                    pooled_vl_tokens_dict[key] = pooled
                else:
                    miss_items.append((txt, views, key))
        else:
            miss_items = list(zip(text_inputs, image_inputs, cache_keys))

        def preprocess_message(args):
            txt, views, key = args
            msg_content = [{"type": "image", "image": v} for v in (views or []) if v is not None]
            msg_content.append({"type": "text", "text": txt})
            messages = [{"role": "user", "content": msg_content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            vision_inputs, video_inputs = process_vision_info(messages)
            return key, txt, views, text, vision_inputs, video_inputs

        if miss_items and use_cache and getattr(self, "strict_cache", False):
            missing_keys = [key for _, _, key in miss_items]
            raise FileNotFoundError(f"Missing cached features for keys: {missing_keys}")

        if miss_items:
            with ThreadPoolExecutor(max_workers=24) as executor:
                results = list(executor.map(preprocess_message, miss_items))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, text, vision_inputs, video_inputs in results:
                    if use_cache:
                        cache_path = self._cache_path(key, txt, views)
                        if cache_path.exists():
                            pooled = torch.load(cache_path, map_location="cpu")
                            pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                            pooled_vl_tokens_dict[key] = pooled
                            continue

                    # ✅ text-only safe input
                    inputs = self.processor(
                        text=[text],
                        images=vision_inputs if vision_inputs else None,
                        videos=video_inputs if video_inputs else None,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    if use_cache:
                        cache_path = self._cache_path(key, txt, views)
                        self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), cache_path)
                        self._enforce_cache_limit()

                    pooled_vl_tokens_dict[key] = pooled.to(dtype=torch.bfloat16)

        if not pooled_vl_tokens_dict:
            raise RuntimeError("⚠️ No valid VL tokens could be generated or loaded.")

        pooled_vl_tokens = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(pooled_vl_tokens, dim=0)
        return vl_tokens


    def _encode_vision_features_parallel(self, text_inputs, image_inputs, device):
        """Parallel multi-view encoding (robust version)"""
        batch_vl_features = []

        # ✅ None-safe setup
        if text_inputs is None:
            text_inputs = []
        if image_inputs is None:
            image_inputs = [None] * len(text_inputs)

        n = min(len(text_inputs), len(image_inputs))
        text_inputs = text_inputs[:n]
        image_inputs = image_inputs[:n]

        for txt, views in zip(text_inputs, image_inputs):
            if not views:
                # text-only
                messages = [{"role": "user", "content": [{"type": "text", "text": txt}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = self.processor(text=[text], images=None, padding=True, return_tensors="pt").to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)
                    batch_vl_features.append(pooled)
            else:
                # multi-view
                batch_msgs, batch_texts, batch_imgs = [], [], []
                for img in views:
                    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
                    text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    vision_inputs, _ = process_vision_info(msg)
                    batch_texts.append(text)
                    batch_imgs.append(vision_inputs)

                inputs = self.processor(
                    text=batch_texts, images=batch_imgs, padding=True, return_tensors="pt"
                ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    vl_tokens = outputs.hidden_states[-1]
                    view_features = vl_tokens.mean(dim=1, keepdim=True)

                    if self.view_aggregation == "mean":
                        aggregated = view_features.mean(dim=0, keepdim=True)
                    elif self.view_aggregation == "max":
                        aggregated, _ = view_features.max(dim=0, keepdim=True)
                    else:
                        aggregated = view_features.mean(dim=0, keepdim=True)

                    batch_vl_features.append(aggregated)

        if not batch_vl_features:
            raise RuntimeError("⚠️ No VL features could be encoded (check inputs).")

        return torch.cat(batch_vl_features, dim=0)

    
    # def _encode_vision_features(self, text_inputs, image_inputs, cache_keys, use_cache, device):
    #     """
    #     Encode VL features with caching

    #     Routes to parallel or sequential encoding based on self.parallel_view_encoding
    #     """
    #     if self.parallel_view_encoding and not use_cache:
    #         # Use parallel encoding for real-time inference (no caching)
    #         return self._encode_vision_features_parallel(text_inputs, image_inputs, device)
    #     else:
    #         # Use sequential encoding (default, with caching support)
    #         return self._encode_vision_features_sequential(text_inputs, image_inputs, cache_keys, use_cache, device)

    # def _encode_vision_features_sequential(self, text_inputs, image_inputs, cache_keys, use_cache, device):
    #     """Sequential encoding (original implementation with caching)"""
    #     pooled_vl_tokens_dict = {}
    #     miss_items = []

    #     if use_cache:
    #         for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
    #             cache_path = self._cache_path(key, txt, views)
    #             if cache_path.exists():
    #                 pooled = torch.load(cache_path, map_location="cpu")
    #                 pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
    #                 pooled_vl_tokens_dict[key] = pooled
    #             else:
    #                 miss_items.append((txt, views, key))
    #     else:
    #         miss_items = list(zip(text_inputs, image_inputs, cache_keys))

    #     def preprocess_message(args):
    #         txt, views, key = args
    #         msg_content = [{"type": "image", "image": v} for v in views if v is not None] if views is not None else []
    #         msg_content.append({"type": "text", "text": txt})
    #         messages = [{"role": "user", "content": msg_content}]
    #         text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #         vision_inputs, video_inputs = process_vision_info(messages)
    #         return key, txt, views, text, vision_inputs, video_inputs

    #     if miss_items and use_cache and getattr(self, "strict_cache", False):
    #         missing_keys = [key for _, _, key in miss_items]
    #         raise FileNotFoundError(f"Missing cached features for keys: {missing_keys}")

    #     if miss_items:
    #         with ThreadPoolExecutor(max_workers=24) as executor:
    #             results = list(executor.map(preprocess_message, miss_items))

    #         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #             for key, txt, views, text, vision_inputs, video_inputs in results:
    #                 if use_cache:
    #                     cache_path = self._cache_path(key, txt, views)
    #                     if cache_path.exists():
    #                         pooled = torch.load(cache_path, map_location="cpu")
    #                         pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
    #                         pooled_vl_tokens_dict[key] = pooled
    #                         continue

    #                 inputs = self.processor(
    #                     text=[text],
    #                     images=vision_inputs,
    #                     videos=video_inputs,
    #                     padding=True,
    #                     return_tensors="pt"
    #                 ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

    #                 outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
    #                 vl_tokens = outputs.hidden_states[-1]
    #                 pooled = vl_tokens.mean(dim=1, keepdim=True)

    #                 if use_cache:
    #                     cache_path = self._cache_path(key, txt, views)
    #                     self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), cache_path)
    #                     self._enforce_cache_limit()

    #                 pooled_vl_tokens_dict[key] = pooled.to(dtype=torch.bfloat16)

    #     pooled_vl_tokens = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
    #     vl_tokens = torch.cat(pooled_vl_tokens, dim=0)

    #     return vl_tokens

    # def _encode_vision_features_parallel(self, text_inputs, image_inputs, device):
    #     """
    #     Parallel multi-view encoding for faster inference

    #     Key idea: Process each view separately in a batch for shorter sequences
    #     - Sequential: 1 long sequence with 5 images (~5000 tokens) - O(n²) attention
    #     - Parallel: 5 short sequences with 1 image each (~1000 tokens each) - O(5×m²) where m << n

    #     Expected speedup: 2-3x due to reduced self-attention complexity
    #     """
    #     batch_vl_features = []

    #     for txt, views in zip(text_inputs, image_inputs):
    #         if views is None or len(views) == 0:
    #             # Text-only case
    #             messages = [{"role": "user", "content": [{"type": "text", "text": txt}]}]
    #             text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #             inputs = self.processor(
    #                 text=[text],
    #                 images=None,
    #                 padding=True,
    #                 return_tensors="pt"
    #             ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

    #             with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #                 outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    #                 vl_tokens = outputs.hidden_states[-1]
    #                 pooled = vl_tokens.mean(dim=1, keepdim=True)
    #                 batch_vl_features.append(pooled)

    #         else:
    #             # Multi-view case: process each view separately
    #             view_features_list = []

    #             # Create batch of single-view messages
    #             batch_messages = []
    #             batch_vision_inputs = []

    #             for img in views:
    #                 msg = [{"role": "user", "content": [
    #                     {"type": "image", "image": img},
    #                     {"type": "text", "text": txt}
    #                 ]}]
    #                 batch_messages.append(msg)

    #                 # Preprocess each message
    #                 text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    #                 vision_inputs, _ = process_vision_info(msg)
    #                 batch_vision_inputs.append((text, vision_inputs))

    #             # Process all views in a single batch
    #             texts = [item[0] for item in batch_vision_inputs]
    #             images = [item[1] for item in batch_vision_inputs]

    #             inputs = self.processor(
    #                 text=texts,  # List[str] with len = num_views
    #                 images=images,  # List[images] with len = num_views
    #                 padding=True,
    #                 return_tensors="pt"
    #             ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

    #             with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #                 # Single forward pass for all views
    #                 outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    #                 vl_tokens = outputs.hidden_states[-1]  # (num_views, seq_len, hidden_dim)

    #                 # Pool each view separately
    #                 view_features = vl_tokens.mean(dim=1, keepdim=True)  # (num_views, 1, hidden_dim)

    #                 # Aggregate views
    #                 if self.view_aggregation == 'mean':
    #                     # Average pooling across views
    #                     aggregated = view_features.mean(dim=0, keepdim=True)  # (1, 1, hidden_dim)
    #                 elif self.view_aggregation == 'max':
    #                     # Max pooling across views
    #                     aggregated, _ = view_features.max(dim=0, keepdim=True)  # (1, 1, hidden_dim)
    #                 elif self.view_aggregation == 'attention':
    #                     # Learnable attention aggregation (TODO: implement)
    #                     # For now, fallback to mean
    #                     aggregated = view_features.mean(dim=0, keepdim=True)
    #                 else:
    #                     aggregated = view_features.mean(dim=0, keepdim=True)

    #                 batch_vl_features.append(aggregated)

    #     # Concatenate batch
    #     vl_tokens = torch.cat(batch_vl_features, dim=0)  # (B, 1, hidden_dim)

    #     return vl_tokens

    @torch.no_grad()
    def predict_action(self, text_inputs, image_inputs, sensor_data, cache_keys, **kwargs):
        """
        Inference-only wrapper

        For diffusion: calls forward with actions=None
        For regression: needs z_chunk
        """
        self.eval()

        if self.model_type == 'diffusion':
            return self.forward(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                actions=None,
                sensor_data=sensor_data,
                cache_keys=cache_keys
            )
        else:  # regression
            # For regression, you need to provide z_chunk
            # This is typically used in iterative action prediction
            raise NotImplementedError(
                "predict_action for regression requires z_chunk. "
                "Use forward() directly with z_chunk parameter."
            )


# Backward compatibility aliases
QwenVLAWithSensorDiffusion = lambda **kwargs: QwenVLAUnified(model_type='diffusion', **kwargs)
QwenVLAWithSensor = lambda **kwargs: QwenVLAUnified(model_type='regression', **kwargs)
Not_freeze_QwenVLAWithSensor = QwenVLAWithSensor  # Alias for old name


if __name__ == "__main__":
    print("🧪 Testing Unified VLA Model...")

    # Test regression
    print("\n=== Testing Regression Model ===")
    model_reg = QwenVLAUnified(
        model_type='regression',
        sensor_enabled=True,
        finetune_vl='none'
    )
    print(f"✅ Regression model created")

    # Test flow matching
    print("\n=== Testing Flow Matching Model ===")
    model_flow = QwenVLAUnified(
        model_type='flow_matching',
        sensor_enabled=True,
        robot_state_enabled=True,
        finetune_vl='none'
    )
    print(f"✅ Flow matching model created")

    print("\n✅ All tests passed!")
