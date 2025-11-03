#!/usr/bin/env python3
"""
Diagnostic tool for verifying the async VLA inference pipeline setup.

This script checks:
1. Network connectivity to data sources
2. ZMQ/UDP port availability
3. Data reception from each source
4. Model checkpoint existence
5. GPU availability and memory

Usage:
    python verify_inference_pipeline.py --robot-ip 10.130.41.110 --jetson-ip 10.130.41.111
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("⚠️  WARNING: ZMQ not available. Install with: pip install pyzmq")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  WARNING: PyTorch not available. Install with: pip install torch")


def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}")


def print_section(text):
    """Print a section header"""
    print(f"\n{text}")
    print(f"{'-'*80}")


def check_network_connectivity(ip_address, port, timeout=2):
    """Check if a host:port is reachable"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip_address, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False


def check_udp_port_available(port):
    """Check if a UDP port is available for binding"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True, "Available"
    except OSError as e:
        if "Address already in use" in str(e):
            return False, "In use (sender may be running)"
        return False, str(e)


def check_tcp_port_available(port):
    """Check if a TCP port is available for binding"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True, "Available"
    except OSError as e:
        if "Address already in use" in str(e):
            return False, "In use (sender may be running)"
        return False, str(e)


def test_robot_data_reception(robot_ip, robot_port, timeout=5):
    """Test receiving robot data via ZMQ SUB"""
    if not ZMQ_AVAILABLE:
        return False, "ZMQ not available"

    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        socket.connect(f'tcp://{robot_ip}:{robot_port}')
        socket.subscribe(b'robot_state')

        print(f"   Waiting for robot data (timeout: {timeout}s)...")

        try:
            topic, payload = socket.recv_multipart()
            ts, send_ts, force, *joints_pose = struct.unpack('<ddf12f', payload)

            socket.close()
            context.term()

            return True, f"Received data: timestamp={ts:.3f}, joints={joints_pose[:6]}"
        except zmq.Again:
            socket.close()
            context.term()
            return False, "Timeout - no data received"

    except Exception as e:
        return False, f"Error: {e}"


def test_camera_data_reception(camera_port, timeout=5):
    """Test receiving camera data via ZMQ PULL"""
    if not ZMQ_AVAILABLE:
        return False, "ZMQ not available"

    try:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        socket.bind(f'tcp://*:{camera_port}')

        print(f"   Waiting for camera data (timeout: {timeout}s)...")

        try:
            metadata_bytes = socket.recv()
            timestamp, view_count = struct.unpack('<dI', metadata_bytes[:12])

            view_sizes = []
            for _ in range(view_count):
                jpg_data = socket.recv()
                view_sizes.append(len(jpg_data))

            socket.close()
            context.term()

            return True, f"Received {view_count} views: {view_sizes} bytes, timestamp={timestamp:.3f}"
        except zmq.Again:
            socket.close()
            context.term()
            return False, "Timeout - no data received"

    except Exception as e:
        return False, f"Error: {e}"


def test_sensor_data_reception(sensor_port, timeout=5):
    """Test receiving sensor data via UDP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', sensor_port))
        sock.settimeout(timeout)

        print(f"   Waiting for sensor data (timeout: {timeout}s)...")

        count = 0
        start_time = time.time()

        # Collect data for up to timeout seconds
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(8192)
                count += 1
                if count >= 10:
                    break
            except socket.timeout:
                break

        sock.close()

        if count > 0:
            elapsed = time.time() - start_time
            rate = count / elapsed
            return True, f"Received {count} packets in {elapsed:.1f}s (~{rate:.1f} Hz)"
        else:
            return False, "Timeout - no data received"

    except Exception as e:
        return False, f"Error: {e}"


def check_checkpoint_exists(checkpoint_path):
    """Check if model checkpoint exists and is valid"""
    path = Path(checkpoint_path)

    if not path.exists():
        return False, "File not found"

    if not TORCH_AVAILABLE:
        return None, "PyTorch not available (cannot verify contents)"

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = []
        if 'epoch' in checkpoint:
            info.append(f"epoch={checkpoint['epoch']}")
        if 'model_state_dict' in checkpoint:
            info.append("model_state_dict=OK")
        if 'optimizer_state_dict' in checkpoint:
            info.append("optimizer_state_dict=OK")

        return True, ", ".join(info)
    except Exception as e:
        return False, f"Failed to load: {e}"


def check_gpu_status():
    """Check GPU availability and memory"""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not available"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        device_count = torch.cuda.device_count()
        devices_info = []

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / 1024**3

            # Try to get current memory usage
            try:
                torch.cuda.set_device(i)
                allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(i) / 1024**3
                free_gb = total_mem_gb - reserved_gb

                devices_info.append(
                    f"GPU {i}: {name} | "
                    f"Total: {total_mem_gb:.1f}GB | "
                    f"Free: {free_gb:.1f}GB | "
                    f"Used: {allocated_gb:.1f}GB"
                )
            except:
                devices_info.append(
                    f"GPU {i}: {name} | Total: {total_mem_gb:.1f}GB"
                )

        return True, "\n      " + "\n      ".join(devices_info)
    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description='Verify async VLA inference pipeline setup')
    parser.add_argument('--robot-ip', default='10.130.41.110', help='Robot PC IP address')
    parser.add_argument('--robot-port', type=int, default=5556, help='Robot ZMQ port')
    parser.add_argument('--jetson-ip', default='10.130.41.111', help='Jetson IP address')
    parser.add_argument('--camera-port', type=int, default=5555, help='Camera ZMQ port')
    parser.add_argument('--sensor-port', type=int, default=9999, help='Sensor UDP port')
    parser.add_argument('--checkpoint', default='./checkpoints/qwen_vla_sensor_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test-data-reception', action='store_true',
                        help='Test actual data reception from sources (requires senders running)')
    args = parser.parse_args()

    print_header("Async VLA Inference Pipeline Verification")

    # Track overall status
    all_checks = []

    # ========================================
    # 1. Check Dependencies
    # ========================================
    print_section("1. Checking Dependencies")

    print(f"   Python version: {sys.version.split()[0]}")

    if ZMQ_AVAILABLE:
        print(f"   ✅ ZMQ: Available")
        all_checks.append(True)
    else:
        print(f"   ❌ ZMQ: Not available")
        all_checks.append(False)

    if TORCH_AVAILABLE:
        print(f"   ✅ PyTorch: Available (version {torch.__version__})")
        all_checks.append(True)
    else:
        print(f"   ❌ PyTorch: Not available")
        all_checks.append(False)

    # ========================================
    # 2. Check Network Connectivity
    # ========================================
    print_section("2. Checking Network Connectivity")

    # Ping robot PC
    print(f"   Robot PC ({args.robot_ip})...")
    reachable = check_network_connectivity(args.robot_ip, args.robot_port)
    if reachable:
        print(f"      ✅ Reachable on port {args.robot_port}")
        all_checks.append(True)
    else:
        print(f"      ⚠️  Not reachable on port {args.robot_port} (sender may not be running)")
        all_checks.append(None)  # Warning, not error

    # Ping Jetson
    print(f"   Jetson ({args.jetson_ip})...")
    # Note: Can't directly test PUSH socket, just check if host is up
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        # Try to connect to any common port to see if host is up
        result = sock.connect_ex((args.jetson_ip, 22))  # SSH port
        sock.close()
        if result == 0:
            print(f"      ✅ Host is reachable")
            all_checks.append(True)
        else:
            print(f"      ⚠️  Host may not be reachable")
            all_checks.append(None)
    except:
        print(f"      ⚠️  Cannot verify host connectivity")
        all_checks.append(None)

    # ========================================
    # 3. Check Port Availability
    # ========================================
    print_section("3. Checking Port Availability")

    # Camera port (TCP PULL)
    available, msg = check_tcp_port_available(args.camera_port)
    if available:
        print(f"   ✅ Camera port {args.camera_port}: {msg}")
        all_checks.append(True)
    else:
        print(f"   ⚠️  Camera port {args.camera_port}: {msg}")
        all_checks.append(None)

    # Sensor port (UDP)
    available, msg = check_udp_port_available(args.sensor_port)
    if available:
        print(f"   ✅ Sensor port {args.sensor_port}: {msg}")
        all_checks.append(True)
    else:
        print(f"   ⚠️  Sensor port {args.sensor_port}: {msg}")
        all_checks.append(None)

    # ========================================
    # 4. Check Model Checkpoint
    # ========================================
    print_section("4. Checking Model Checkpoint")

    exists, msg = check_checkpoint_exists(args.checkpoint)
    if exists:
        print(f"   ✅ Checkpoint: {msg}")
        all_checks.append(True)
    elif exists is None:
        print(f"   ⚠️  Checkpoint: {msg}")
        all_checks.append(None)
    else:
        print(f"   ❌ Checkpoint: {msg}")
        all_checks.append(False)

    # ========================================
    # 5. Check GPU Status
    # ========================================
    print_section("5. Checking GPU Status")

    available, msg = check_gpu_status()
    if available:
        print(f"   ✅ GPU: {msg}")
        all_checks.append(True)
    else:
        print(f"   ❌ GPU: {msg}")
        all_checks.append(False)

    # ========================================
    # 6. Test Data Reception (Optional)
    # ========================================
    if args.test_data_reception:
        print_section("6. Testing Data Reception (requires senders running)")

        print("\n   Testing Robot Data...")
        success, msg = test_robot_data_reception(args.robot_ip, args.robot_port)
        if success:
            print(f"   ✅ Robot data: {msg}")
            all_checks.append(True)
        else:
            print(f"   ❌ Robot data: {msg}")
            all_checks.append(False)

        print("\n   Testing Camera Data...")
        success, msg = test_camera_data_reception(args.camera_port)
        if success:
            print(f"   ✅ Camera data: {msg}")
            all_checks.append(True)
        else:
            print(f"   ❌ Camera data: {msg}")
            all_checks.append(False)

        print("\n   Testing Sensor Data...")
        success, msg = test_sensor_data_reception(args.sensor_port)
        if success:
            print(f"   ✅ Sensor data: {msg}")
            all_checks.append(True)
        else:
            print(f"   ❌ Sensor data: {msg}")
            all_checks.append(False)
    else:
        print_section("6. Data Reception Tests")
        print("   ⏭  Skipped (use --test-data-reception to enable)")
        print("   Note: This requires all senders to be running")

    # ========================================
    # Summary
    # ========================================
    print_header("Summary")

    # Count results
    passed = sum(1 for x in all_checks if x is True)
    failed = sum(1 for x in all_checks if x is False)
    warnings = sum(1 for x in all_checks if x is None)
    total = len(all_checks)

    print(f"\n   Total checks: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Warnings: {warnings}")

    if failed == 0 and warnings == 0:
        print("\n   🎉 All checks passed! System is ready for inference.")
        return 0
    elif failed == 0:
        print("\n   ⚠️  Some warnings, but system should work.")
        print("   Review the warnings above for potential issues.")
        return 0
    else:
        print("\n   ❌ Some checks failed. Please fix the issues above before running inference.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
