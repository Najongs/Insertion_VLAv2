# sender.py
import socket
import struct
import time

UDP_IP = "192.168.0.101"   # 로봇 제어 PC(IP) 또는 수신자 IP
UDP_PORT = 10000           # 수신 측 포트 (임의로 지정)
RATE_HZ = 10               # 10 Hz 주기
DT = 1.0 / RATE_HZ

# payload format: 6 floats (x, y, z, a, b, c)
FMT = "<6f"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

start_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end_pose = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
steps = 100
for i in range(steps):
    ratio = i / steps
    pose = [(1 - ratio) * s + ratio * e for s, e in zip(start_pose, end_pose)]
    payload = struct.pack(FMT, *pose)
    sock.sendto(payload, (UDP_IP, UDP_PORT))
    print(f"Sent target: {pose}")
    time.sleep(DT)

sock.close()
print("✅ Sender finished.")
