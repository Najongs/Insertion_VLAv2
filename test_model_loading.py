#!/usr/bin/env python3
"""
Test that the unified model loads correctly with flow matching
"""

import torch
import sys

print("🧪 Testing Model Loading...")
print()

# Test 1: Import check
print("1️⃣ Testing imports...")
try:
    from models.unified_model import QwenVLAUnified, RobotStateEncoder
    from models.flow_matching import FlowMatchingActionExpert, OptimalTransportConditionalFlowMatching
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Flow matching model creation
print("2️⃣ Testing flow matching model creation...")
try:
    model = QwenVLAUnified(
        model_type='flow_matching',
        sensor_enabled=True,
        robot_state_enabled=True,
        sensor_input_channels=1026,
        sensor_temporal_length=650,
        sensor_output_dim=2048,
        fusion_strategy='concat',
        finetune_vl='none',
        image_resize_height=360,
        image_resize_width=640,
    )
    print("   ✅ Flow matching model created successfully")
    print(f"   Model type: {model.model_type}")
    print(f"   Sensor enabled: {model.sensor_enabled}")
    print(f"   Robot state enabled: {model.robot_state_enabled}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Check encoders
print("3️⃣ Checking encoders...")
try:
    assert model.sensor_encoder is not None, "Sensor encoder is None"
    assert model.robot_state_encoder is not None, "Robot state encoder is None"
    assert isinstance(model.robot_state_encoder, RobotStateEncoder), "Wrong robot state encoder type"
    print("   ✅ Sensor encoder: Present")
    print("   ✅ Robot state encoder: Present (RobotStateEncoder)")
except Exception as e:
    print(f"   ❌ Encoder check failed: {e}")
    sys.exit(1)

print()

# Test 4: Check action expert
print("4️⃣ Checking action expert...")
try:
    assert isinstance(model.action_expert, FlowMatchingActionExpert), "Wrong action expert type"
    print("   ✅ Action expert type: FlowMatchingActionExpert")
    print(f"   Horizon: {model.action_expert.horizon}")
    print(f"   Action dim: {model.action_expert.action_dim}")
except Exception as e:
    print(f"   ❌ Action expert check failed: {e}")
    sys.exit(1)

print()

# Test 5: Try creating diffusion model (should fail)
print("5️⃣ Testing diffusion deprecation...")
try:
    model_diff = QwenVLAUnified(
        model_type='diffusion',
        sensor_enabled=True,
        finetune_vl='none'
    )
    print("   ❌ Diffusion model should have raised ValueError!")
    sys.exit(1)
except ValueError as e:
    if "deprecated" in str(e).lower():
        print("   ✅ Diffusion correctly deprecated")
        print(f"   Error message: {str(e).split(chr(10))[0]}")
    else:
        print(f"   ❌ Wrong error: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")
    sys.exit(1)

print()

# Test 6: Regression model (should still work)
print("6️⃣ Testing regression model...")
try:
    model_reg = QwenVLAUnified(
        model_type='regression',
        sensor_enabled=True,
        robot_state_enabled=True,
        finetune_vl='none'
    )
    print("   ✅ Regression model created successfully")
except Exception as e:
    print(f"   ❌ Regression model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ All tests passed!")
print()
print("Summary:")
print("  ✅ Flow matching model works")
print("  ✅ RobotStateEncoder integrated")
print("  ✅ Diffusion model deprecated")
print("  ✅ Regression model still works")
print()
print("Ready for training with:")
print("  bash run_flow_matching.sh")
