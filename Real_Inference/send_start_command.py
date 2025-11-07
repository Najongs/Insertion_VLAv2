#!/usr/bin/env python3
"""
Send START command to robot_command_receiver.py

Usage:
    python send_start_command.py --robot-ip 127.0.0.1
"""
import zmq
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Send START command to robot')
    parser.add_argument('--robot-ip', type=str, default='127.0.0.1',
                       help='IP address of robot_command_receiver (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Command port (default: 5000)')
    parser.add_argument('--joints', type=float, nargs=6,
                       default=[0, 0, 0, 0, 0, 0],
                       help='Start joint positions (default: [0,0,0,0,0,0])')
    args = parser.parse_args()

    # Create ZMQ PUSH socket
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{args.robot_ip}:{args.port}")

    print(f"Connecting to tcp://{args.robot_ip}:{args.port}...")
    time.sleep(0.5)  # Give time for connection

    # Send start command
    start_cmd = {
        "cmd": "start",
        "start_joints": list(args.joints),
        "lock_j6": False
    }

    print(f"Sending START command: {start_cmd}")
    sock.send_json(start_cmd)

    print("✅ START command sent!")
    print("\nRobot should now be ready to receive dpose commands.")
    print("Check robot_command_receiver.py logs for confirmation.")

    time.sleep(0.5)
    sock.close()
    ctx.term()

if __name__ == "__main__":
    main()
