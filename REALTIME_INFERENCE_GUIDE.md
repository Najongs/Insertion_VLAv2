# Real-time Inference Without VL Caching

실시간 추론 시 VL 캐싱 없이 작동하는 방법을 설명합니다.

## 🎯 핵심 차이점

### 학습 시 (VL Caching 사용)
```python
# 1. 사전에 VL features를 캐시로 저장
python Make_VL_cache.py

# 2. 학습 시 캐시 로드
vl_features = load_from_cache(cache_key)  # 디스크에서 로드
actions = model(vl_features, sensor, robot_states)
```

**장점:**
- VL encoding을 한 번만 수행 (빠름)
- VLM reuse로 효율적

**단점:**
- 실시간 추론에 사용 불가 (새로운 이미지 처리 불가)

---

### 실시간 추론 (VL Caching 없음)
```python
# 매 프레임 VL encoding 수행
for frame in camera_stream:
    # 1. VL encoding (실시간)
    vl_features = vl_model(frame, text)  # 매번 새로 encoding

    # 2. Sensor & Robot state encoding
    sensor_features = sensor_encoder(sensor_data)
    robot_features = robot_encoder(robot_states)

    # 3. Action prediction
    actions = action_expert(vl_features, sensor_features, robot_features)
```

**장점:**
- 실시간 추론 가능
- 새로운 이미지 처리 가능

**단점:**
- VL encoding이 병목 (전체의 90% 시간)
- 비동기 처리 필수

---

## 🔧 코드 수정 방법

### 1. Model Forward 수정

```python
# models/unified_model.py - QwenVLAUnified.forward()

def forward(self, text_inputs, image_inputs, ...):
    # ❌ 기존: 캐시 사용
    vl_tokens = self._encode_vision_features(
        text_inputs, image_inputs, cache_keys, use_cache=True
    )

    # ✅ 실시간: 캐시 미사용
    vl_tokens = self._encode_vision_features_realtime(
        text_inputs, image_inputs
    )
```

### 2. Real-time VL Encoding 구현

```python
def _encode_vision_features_realtime(self, text_inputs, image_inputs):
    """Encode VL features without caching (real-time inference)"""

    # Prepare messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img}
            for img in image_inputs
        ] + [{"type": "text", "text": text_inputs[0]}]
    }]

    # Process
    text = self.processor.apply_chat_template(messages, ...)
    vision_inputs, _ = process_vision_info(messages)

    inputs = self.processor(
        text=[text],
        images=vision_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device='cuda', dtype=torch.bfloat16)

    # VL model forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = self.vl_model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,  # ✅ KV cache도 비활성화
            return_dict=True
        )
        vl_tokens = outputs.hidden_states[-1]  # (B, seq_len, 2048)

        # ✅ Mean pooling (학습 시 캐시와 동일)
        vl_features = vl_tokens.mean(dim=1, keepdim=True)  # (B, 1, 2048)

    return vl_features
```

### 3. Dataset 수정 (실시간용)

```python
# vla_datasets/unified_dataset.py

class RealtimeVLADataset(UnifiedVLADataset):
    """Dataset for real-time inference without VL caching"""

    def __getitem__(self, idx):
        # ❌ VL cache 로드하지 않음
        # cache_key = ...
        # vl_features = load_cache(cache_key)

        # ✅ 이미지 경로와 텍스트만 반환 (모델에서 encoding)
        return {
            "images": image_paths,  # List[str]
            "instruction": instruction,  # str
            "sensor_data": sensor_data,
            "robot_states": robot_states,
            "actions": actions,
        }
```

---

## ⚡ 비동기 처리 전략

VL encoding이 느리므로 비동기 처리가 필수입니다.

### VLM Reuse Pattern

```python
import threading
import queue

class AsyncVLEncoder:
    def __init__(self, vl_model, processor):
        self.vl_model = vl_model
        self.processor = processor
        self.vl_queue = queue.Queue(maxsize=3)  # VLM reuse count
        self.running = True

        # VL encoding 스레드 시작
        self.thread = threading.Thread(target=self._encode_loop, daemon=True)
        self.thread.start()

    def _encode_loop(self):
        """Background thread for VL encoding"""
        while self.running:
            try:
                # 새 프레임 가져오기
                frame, text = camera.get_frame(), get_instruction()

                # VL encoding (느림: ~300ms)
                vl_features = self._encode(frame, text)

                # Queue에 저장 (reuse를 위해)
                self.vl_queue.put(vl_features)

            except Exception as e:
                print(f"VL encoding error: {e}")

    def get_features(self):
        """Get VL features from queue (fast)"""
        return self.vl_queue.get(timeout=0.5)
```

### Main Loop

```python
# Main control loop (10Hz)
async_encoder = AsyncVLEncoder(vl_model, processor)

while True:
    start = time.time()

    # 1. VL features 가져오기 (비동기, 빠름)
    vl_features = async_encoder.get_features()

    # 2. Sensor & Robot encoding (빠름: ~10ms)
    sensor_features = sensor_encoder(sensor_data)
    robot_features = robot_encoder(robot_states)

    # 3. Action prediction (빠름: ~15ms)
    actions = action_expert(vl_features, sensor_features, robot_features)

    # 4. Execute actions
    robot.execute(actions[0])

    # 10Hz 유지
    elapsed = time.time() - start
    time.sleep(max(0, 0.1 - elapsed))
```

---

## 📊 성능 비교

| 방식 | VL Encoding | Action Prediction | 총 시간 | FPS |
|------|-------------|-------------------|---------|-----|
| 동기 (캐시 없음) | 300ms | 25ms | 325ms | 3.1 FPS |
| 비동기 (VLM reuse=3) | 300ms (백그라운드) | 25ms | 25ms | **40 FPS** |

**VLM Reuse=3 의미:**
- VL encoding을 300ms마다 1번 수행
- Action prediction은 100ms(10Hz)마다 수행
- 같은 VL features를 3번 재사용

---

## 🚀 Quick Start

### 1. 벤치마크 실행
```bash
# VL 캐싱 없이 실시간 추론 벤치마크
bash benchmark_quick_test.sh
```

### 2. 실시간 추론 테스트
```python
# test_realtime_inference.py
from models.unified_model import QwenVLAUnified

model = QwenVLAUnified(
    model_type='regression',
    sensor_enabled=True,
    robot_state_enabled=True,
)

# Disable cache for real-time
model.cache_enabled = False

# Test inference
vl_features = model._encode_vision_features_realtime(
    text_inputs=["Pick up the blue object"],
    image_inputs=[img1, img2, img3, img4, img5]
)

actions = model.action_expert(vl_features, sensor, robot)
```

---

## 📝 체크리스트

실시간 추론을 위한 확인 사항:

- [ ] `model.cache_enabled = False` 설정
- [ ] VL encoding에서 `use_cache=False` 사용
- [ ] Mean pooling 적용 (학습 시 캐시와 동일)
- [ ] 비동기 VL encoding 구현
- [ ] VLM reuse pattern 적용
- [ ] 10Hz 제어 주기 달성 확인

---

**작성일:** 2025-01-04
**버전:** v1.0 - Real-time Inference without Caching
