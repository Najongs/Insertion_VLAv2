# receiver_robot.py
import socket
import struct
import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer

# UDP 설정
UDP_IP = "0.0.0.0"
UDP_PORT = 10000
FMT = "<6f"   # 6 floats
SIZE = struct.calcsize(FMT)

# Meca500 설정
ROBOT_IP = "192.168.0.100"

def main():
    print("✅ Initializing Meca500 robot...")
    robot = initializer.RobotWithTools()
    robot.Connect(ROBOT_IP)
    robot.ActivateRobot()
    robot.Home()
    robot.WaitHomed()
    robot.SetRealTimeMode(1)
    robot.SetCartLinVel(50)
    robot.SetCartLinAccel(200)

    print("✅ Meca500 ready in RT mode.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)

    try:
        while True:
            try:
                data, addr = sock.recvfrom(SIZE)
                if len(data) != SIZE:
                    continue
                x, y, z, a, b, c = struct.unpack(FMT, data)
                cmd = f"SetRealtimeTarget({x:.3f},{y:.3f},{z:.3f},{a:.3f},{b:.3f},{c:.3f})"
                robot.SendCustomCommand(cmd)
                print(f"[RT] Received target from {addr}: {cmd}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[WARN] {e}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        robot.SetRealTimeMode(0)
        robot.DeactivateRobot()
        robot.Disconnect()
        sock.close()
        print("✅ Receiver stopped cleanly.")

if __name__ == "__main__":
    main()
