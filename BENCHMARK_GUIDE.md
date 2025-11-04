# Real-time Inference Benchmark Guide

실시간 추론 성능을 측정하고 분석하기 위한 벤치마크 도구입니다.

## 📋 목차

1. [개요](#개요)
2. [측정 항목](#측정-항목)
3. [사용 방법](#사용-방법)
4. [벤치마크 시나리오](#벤치마크-시나리오)
5. [결과 해석](#결과-해석)

---

## 개요

### 목적
- VL 모델, Sensor Encoder, Action Expert의 **개별 추론 시간** 측정
- Regression vs Flow Matching 모델 비교
- 카메라 view 개수에 따른 성능 변화 분석
- Sensor/Robot States 입력 유무에 따른 성능 영향 분석

### 특징
- ✅ **VL 캐시 완전 비활성화** (실제 실시간 추론 환경 시뮬레이션)
- ✅ **컴포넌트별 시간 측정** (VL, Sensor, Action 분리)
- ✅ **GPU Synchronization** (정확한 시간 측정)
- ✅ **Warmup + 반복 측정** (안정적인 결과)
- ✅ **비동기 모델 지원** (VLM reuse pattern)
- ✅ **시각화 자동 생성** (그래프 및 표)

### ⚠️ 중요: VL 캐시 비활성화
실시간 추론에서는 매 프레임 새로운 이미지가 입력되므로 **VL 캐시를 사용할 수 없습니다**.
따라서 벤치마크에서도 `cache_enabled=False`로 설정하여 실제 환경을 정확히 시뮬레이션합니다.

```python
# 벤치마크 초기화 시 자동으로 설정됨
self.model.cache_enabled = False  # ✅ 실시간 추론 환경
```

학습 시에는 VLM reuse (예: 3회)를 사용하여 VL encoding을 절약하지만,
**벤치마크는 매번 새로운 VL encoding을 수행**하여 worst-case 시나리오를 측정합니다.

---

## 측정 항목

### 1. VL Encoding Time
**측정 내용:**
- Text + Image → Vision-Language features 생성 시간
- **캐시 없이 매번 새로 encoding** (실시간 환경)

**포함 작업:**
1. Image loading & preprocessing
2. Tokenization
3. Qwen2.5-VL forward pass (3B parameters)
4. Feature extraction (hidden states)
5. **Mean pooling** (sequence → single vector)

**VL Processing Pipeline:**
```python
# Step 1-3: VL model forward
vl_outputs = vl_model(**inputs, output_hidden_states=True, use_cache=False)
vl_tokens = vl_outputs.hidden_states[-1]  # (B, seq_len, 3072)

# Step 4-5: Pool to match training format
vl_features = vl_tokens.mean(dim=1, keepdim=True)  # (B, 1, 3072)
```

**예상 시간:**
- 1 view: ~150-250ms
- 3 views: ~250-350ms
- 5 views: ~350-500ms

**주의:**
- VL encoding은 **가장 큰 병목** (전체의 90% 이상)
- 실시간 추론에서는 비동기 처리 필수
- VLM reuse로 overhead 분산 가능
- **Pooling 방식이 학습 시 캐시와 동일해야 함**

---

### 2. Sensor Encoding Time
**측정 내용:**
- Sensor data + Robot states → Sensor features 생성 시간

**포함 작업:**
- Sensor data preprocessing
- Robot states encoding
- Temporal 1D CNN encoding
- Feature pooling

**예상 시간:** ~5-15ms

---

### 3. Action Prediction Time
**측정 내용:**
- VL features + Sensor features → Action sequence 생성 시간

**포함 작업:**
- Feature fusion (concat/cross-attention)
- Action expert forward pass
- **Regression**: Direct prediction
- **Flow Matching**: ODE sampling

**예상 시간:**
- Regression: ~10-20ms
- Flow Matching: ~30-50ms

---

### 4. End-to-End Time
**측정 내용:**
- 전체 추론 시간 (VL + Sensor + Action)

**예상 시간:** ~250-500ms (4-2 FPS)

---

## 사용 방법

### 1. 기본 사용법

#### A. Regression vs Flow Matching 비교
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --checkpoint-flow ./checkpoints/flow_matching_best.pt \
    --dataset-dir /path/to/episode_dir \
    --num-iterations 10
```

#### B. Regression만 테스트
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --dataset-dir /path/to/episode_dir
```

#### C. Flow Matching만 테스트
```bash
python benchmark_realtime_inference.py \
    --checkpoint-flow ./checkpoints/flow_matching_best.pt \
    --dataset-dir /path/to/episode_dir
```

---

### 2. 고급 옵션

#### A. 반복 횟수 조정
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --num-iterations 20  # 더 정확한 측정
```

#### B. 카메라 view 개수 조정
```bash
# 3개 view만 사용 (속도 향상)
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --num-views 3
```

#### C. Sensor 비활성화
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --disable-sensor
```

#### D. Robot States 비활성화
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --disable-robot-states
```

#### E. GPU 선택
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --device cuda:1  # GPU 1 사용
```

---

### 3. 비교 모드

#### A. View 개수 비교 (1-5 views)
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --compare-views
```

**결과:**
- 각 view 개수별 성능 측정
- View 개수에 따른 latency/throughput 그래프 생성

#### B. Sensor 유무 비교
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --compare-sensors
```

**결과:**
- Sensor 사용/미사용 성능 비교
- Sensor overhead 측정

---

### 4. 일괄 실행 스크립트

**모든 벤치마크를 한번에 실행:**
```bash
bash run_benchmark.sh
```

**포함 테스트:**
1. Regression vs Flow Matching
2. View 개수 비교 (1-5)
3. Sensor 유무 비교
4. 직접 Sensor 비교

---

## 벤치마크 시나리오

### 시나리오 1: 모델 비교
**목적:** Regression vs Flow Matching 성능 비교

**명령어:**
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --checkpoint-flow ./checkpoints/flow_matching_best.pt \
    --num-iterations 20 \
    --output-dir ./benchmark_results/model_comparison
```

**예상 결과:**
- Regression이 Flow Matching보다 **2-3배 빠름**
- Flow Matching은 ODE sampling으로 인한 overhead

---

### 시나리오 2: 실시간 요구사항 확인
**목적:** 10Hz 제어 주기 달성 가능 여부 확인

**요구사항:**
- 제어 주기: 100ms (10Hz)
- VLM reuse: 3회 (300ms마다 VL encoding)

**명령어:**
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --num-views 3 \
    --num-iterations 20
```

**판단 기준:**
- Action Prediction < 50ms: ✅ 가능
- Action Prediction > 100ms: ❌ 불가능
- VL Encoding은 비동기 처리 (별도 스레드)

---

### 시나리오 3: 최적 View 개수 찾기
**목적:** 성능과 정확도 trade-off 분석

**명령어:**
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --compare-views \
    --num-iterations 20 \
    --output-dir ./benchmark_results/view_optimization
```

**분석:**
1. 1 view: 가장 빠름 (하지만 spatial info 부족)
2. 3 views: 균형점 (성능 + 정확도)
3. 5 views: 가장 느림 (하지만 최고 정확도)

---

### 시나리오 4: Sensor Impact 분석
**목적:** Sensor가 성능에 미치는 영향 측정

**명령어:**
```bash
python benchmark_realtime_inference.py \
    --checkpoint-regression ./checkpoints/regression_best.pt \
    --compare-sensors \
    --num-iterations 20 \
    --output-dir ./benchmark_results/sensor_impact
```

**분석:**
- Sensor overhead: ~5-15ms
- Sensor가 정확도 향상에 기여하는지 확인 필요

---

## 결과 해석

### 출력 형식

**터미널 출력 예시:**
```
============================================================
Results: Regression (regression)
============================================================

📊 Timing Breakdown:
  VL Encoding:       287.34 ± 12.45 ms
  Sensor Encoding:   8.72 ± 1.23 ms
  Action Prediction: 15.67 ± 2.11 ms
  ────────────────────────────────────────
  Total (E2E):       311.73 ± 13.89 ms
  Throughput:        3.21 FPS

📈 Component Breakdown:
  VL Encoding:       92.2%
  Sensor Encoding:   2.8%
  Action Prediction: 5.0%
```

**해석:**
- VL Encoding이 **전체 시간의 92%** 차지 → 병목 지점
- Action Prediction은 매우 빠름 (15ms)
- 비동기 처리로 VL Encoding을 숨길 수 있음

---

### 저장 파일

**1. JSON 결과 파일**
- `regression_results.json`: Regression 상세 결과
- `flow_matching_results.json`: Flow Matching 상세 결과

**내용:**
```json
{
  "model_name": "Regression",
  "model_type": "regression",
  "vl_encoding": {
    "mean": 0.28734,
    "std": 0.01245,
    "min": 0.27123,
    "max": 0.31456
  },
  "total": {
    "mean": 0.31173,
    "fps": 3.21
  },
  "raw_results": [...]
}
```

**2. CSV 비교 표**
- `comparison.csv`: 모델 간 비교 표

**내용:**
```csv
Model,Type,VL (ms),Sensor (ms),Action (ms),Total (ms),FPS
Regression,regression,287.34,8.72,15.67,311.73,3.21
Flow Matching,flow_matching,289.12,8.94,42.35,340.41,2.94
```

**3. 시각화 그래프**
- `comparison.png`: 성능 비교 그래프
  - 왼쪽: 컴포넌트별 시간 분해 (bar chart)
  - 오른쪽: 전체 시간 + FPS (bar + line plot)

---

### 병목 지점 분석

**Case 1: VL Encoding이 90% 이상**
```
VL Encoding:   92.2%
Sensor:        2.8%
Action:        5.0%
```

**해결책:**
- ✅ VLM reuse count 증가 (3 → 5)
- ✅ 비동기 VL encoding (별도 스레드)
- ✅ View 개수 감소 (5 → 3)
- ✅ 이미지 해상도 감소

---

**Case 2: Action Prediction이 30% 이상**
```
VL Encoding:   60.0%
Sensor:        10.0%
Action:        30.0%  ← 병목
```

**해결책:**
- ✅ Flow Matching → Regression 전환
- ✅ Action expert hidden dim 감소
- ✅ torch.compile 적용

---

**Case 3: Sensor Encoding이 20% 이상**
```
VL Encoding:   70.0%
Sensor:        20.0%  ← 병목
Action:        10.0%
```

**해결책:**
- ✅ Sensor window size 감소 (650 → 65)
- ✅ 1D CNN depth 감소
- ✅ Sensor를 비활성화하고 성능 확인

---

### 실시간 요구사항 체크리스트

#### 10Hz 제어 주기 (100ms)
- [ ] Action Prediction < 50ms
- [ ] VL Encoding 비동기 처리
- [ ] Sensor Encoding < 10ms
- [ ] Total overhead < 70ms

#### 30Hz 제어 주기 (33ms)
- [ ] Action Prediction < 15ms
- [ ] Sensor Encoding < 5ms
- [ ] VL Encoding 백그라운드 처리
- [ ] Total overhead < 25ms

---

## 최적화 권장 사항

### 1. VL Encoding 최적화
```python
# 이미지 해상도 감소
--image_resize_height 270  # 360 → 270
--image_resize_width 480   # 640 → 480

# View 개수 감소
--num-views 3  # 5 → 3

# VLM reuse 증가 (accuracy vs latency trade-off)
--vlm-reuse-count 5  # 3 → 5
```

### 2. Model Optimization
```python
# torch.compile 적용 (10-20% speedup)
model = torch.compile(model, mode='max-autotune')

# FlashAttention-2 확인
# 이미 사용 중: attn_implementation="flash_attention_2"
```

### 3. Sensor Optimization
```python
# Sensor window 크기 감소
--sensor-window-size 32  # 65 → 32

# Sensor 비활성화 (정확도 확인 필요)
--disable-sensor
```

---

## FAQ

**Q1: FPS가 너무 낮습니다 (< 2 FPS). 어떻게 해야 하나요?**

A: VL Encoding이 병목일 가능성이 높습니다.
1. View 개수 감소 (5 → 3)
2. 이미지 해상도 감소
3. VLM reuse count 증가
4. 비동기 VL encoding 구현

---

**Q2: Flow Matching이 Regression보다 얼마나 느린가요?**

A: Action Prediction 시간 기준:
- Regression: ~10-20ms
- Flow Matching: ~30-50ms (2-3배 차이)

하지만 VL Encoding을 비동기로 처리하면 전체 E2E 차이는 작음.

---

**Q3: 실시간 10Hz 제어가 가능한가요?**

A: VL Encoding을 비동기로 처리하면 가능합니다.
- VL Encoding: 별도 스레드에서 300ms마다 실행 (reuse=3)
- Action Prediction: 메인 루프에서 100ms마다 실행

---

**Q4: Sensor가 성능에 미치는 영향은?**

A: Overhead는 작지만 (5-15ms), 정확도 향상 효과를 확인해야 합니다.
`--compare-sensors` 옵션으로 비교하세요.

---

**Q5: 여러 GPU에서 동시에 테스트하려면?**

A:
```bash
# GPU 0
python benchmark_realtime_inference.py --device cuda:0 &

# GPU 1
python benchmark_realtime_inference.py --device cuda:1 &

wait
```

---

## 다음 단계

1. **벤치마크 실행**
   ```bash
   bash run_benchmark.sh
   ```

2. **결과 분석**
   - `benchmark_results/` 폴더 확인
   - 그래프 및 CSV 파일 검토

3. **최적화 적용**
   - 병목 지점 파악
   - 최적화 권장 사항 적용

4. **실제 환경 테스트**
   - Real robot에서 실시간 추론 테스트
   - Latency 모니터링

---

**작성일:** 2025-01-04
**버전:** v1.0
