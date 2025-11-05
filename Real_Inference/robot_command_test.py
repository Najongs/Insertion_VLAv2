# ============================================================
# delta_action_sender.py
# ============================================================
import zmq
import struct
import time
import numpy as np

# =========================
# Configuration
# =========================
ZMQ_PUB_ADDRESS = "tcp://127.0.0.1:5557"  # robot_control.py의 Subscriber 주소
ZMQ_TOPIC = b"robot_cmd"                   # 토픽 이름 (subscriber와 동일해야 함)
SEND_RATE_HZ = 10                          # 10Hz로 전송
DT = 1.0 / SEND_RATE_HZ

# =========================
# ZMQ 초기화
# =========================
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.connect(ZMQ_PUB_ADDRESS)

print(f"✅ Delta Action Sender connected to {ZMQ_PUB_ADDRESS}")
print(f"   Topic: '{ZMQ_TOPIC.decode()}', rate: {SEND_RATE_HZ} Hz")

# =========================
# 예시 Delta Action 생성
# =========================
# 예: 5초간 X축 +1mm 이동, 이후 복귀
pattern = [
    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),   # +X
    np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # -X
]

try:
    print("▶️ Sending ΔEE commands...")
    start_time = time.time()
    t0 = start_time

    while time.time() - start_time < 10:  # 총 10초간 전송
        elapsed = time.time() - t0
        if elapsed < 5:
            delta = pattern[0]  # +X 방향
        else:
            delta = pattern[1]  # -X 방향

        payload = struct.pack("<6f", *delta)
        pub.send_multipart([ZMQ_TOPIC, payload])
        print(f"Sent ΔEE: {delta.tolist()}")
        time.sleep(DT)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    pub.close()
    ctx.term()
    print("✅ Sender terminated cleanly.")
