# robot_command_receiver.py (rev. stable-10hz-delta + abs pose)
import socket, json, time, threading, atexit, signal, sys
from math import copysign

ROBOT_IP, CTRL_PORT = "192.168.0.100", 10000
LISTEN_PORT = 5000

# === 주기/워치독 ===
CTRL_HZ   = 10.0
CTRL_DT   = 1.0 / CTRL_HZ
WATCHDOG_T = 5.0
KEEPALIVE_PERIOD = WATCHDOG_T * 0.5
VEL_TIMEOUT = 0.40  # (10Hz보다 크게)

# === 시작 이동 속도 제한(요청) ===
CART_LIN_V0 = 5     # mm/s
CART_ANG_V0 = 10    # deg/s

# === 1틱당 Δpose 클램프(안전) ===
CLAMP_POS_MM  = 1.0   # mm
CLAMP_ANG_DEG = 2.0   # deg

robot_sock = None
_keepalive_run = False

# 누적 Δpose 버퍼
_dp_lock = threading.Lock()
_dp_acc = [0.0]*6
_started = False
_stop_flag = False

# 파일 상단 전역
_lock_j6 = False
_j6_target = None  # deg

shutdown_done = False

# ---------- TCP helpers ----------
def send_cmd(sock, cmd: str):
    sock.sendall((cmd + "\0").encode("ascii"))

def recv_line(sock, timeout=2.0):
    sock.settimeout(timeout)
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("robot closed")
        if b == b"\x00":
            break
        buf.extend(b)
    return buf.decode("ascii", "ignore")

def try_recv_line(sock, timeout=0.5):
    try:
        return recv_line(sock, timeout=timeout)
    except Exception:
        return None

def send_and_get(sock, cmd, timeout=2.0):
    send_cmd(sock, cmd)
    return recv_line(sock, timeout=timeout)

# ---------- Watchdog ----------
def watchdog_keepalive(sock):
    global _keepalive_run
    while _keepalive_run:
        try:
            send_cmd(sock, f"ConnectionWatchdog({WATCHDOG_T})")
            _ = try_recv_line(sock, timeout=0.5)
        except Exception:
            break
        time.sleep(KEEPALIVE_PERIOD)

# ---------- Status / helpers ----------
def _parse_status(line):
    # [2007][as,hs,sm,es,pm,eob,eom]
    fields = line.split("[", 2)[2].split("]")[0].split(",")
    as_, hs, sm, es, pm, eob, eom = [int(x.strip()) for x in fields]
    return {"as": as_, "hs": hs, "sm": sm, "es": es, "pm": pm, "eob": eob, "eom": eom}

def get_status():
    try:
        send_cmd(robot_sock, "GetStatusRobot()")
        line = try_recv_line(robot_sock, timeout=0.5)
        if not line or not line.startswith("[2007]"):
            return None
        return _parse_status(line)
    except Exception:
        return None

def wait_until(pred, timeout=60.0, poll=0.1, label=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = get_status()
        if st and pred(st):
            return True
        time.sleep(poll)
    print(f" -> [wait_until timeout] {label}")
    return False

def _clamp(v, lim):
    return v if abs(v) <= lim else copysign(lim, v)

def _parse_pose_line(line):
    # [2053][x,y,z,a,b,g] 또는 숫자 쉼표 6개
    try:
        payload = line.split("[", 2)[2].split("]")[0]
        vals = [float(v.strip()) for v in payload.split(",")]
        if len(vals) == 6:
            return vals
    except Exception:
        pass
    return None

# ---------- Robot session ----------
def robot_connect():
    print("[RCV] Connecting to robot...")
    s = socket.create_connection((ROBOT_IP, CTRL_PORT), timeout=3)
    print("[RCV] Connected.")
    banner = try_recv_line(s, timeout=1.0)
    if banner:
        print(" ->", banner)
    return s

def robot_init():
    global robot_sock, _keepalive_run
    print("[RCV] Robot init...")

    # 워치독 + keepalive
    print(" ->", send_and_get(robot_sock, f"ConnectionWatchdog({WATCHDOG_T})", 1.0))
    _keepalive_run = True
    threading.Thread(target=watchdog_keepalive, args=(robot_sock,), daemon=True).start()

    # 에러 있으면 해제
    st = get_status()
    if st and st["es"] != 0:
        print(" ->", send_and_get(robot_sock, "ResetError()", 1.0))
        print(" ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
        wait_until(lambda s: s["es"] == 0, 5, label="error clear")

    # Activate
    st = get_status()
    if not st or st["as"] == 0:
        rep = send_and_get(robot_sock, "ActivateRobot()", 2.0)
        print(" ->", rep)
        if rep.startswith("[1013]"):
            print(" ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
            print(" ->", send_and_get(robot_sock, "ActivateRobot()", 2.0))
    wait_until(lambda s: s and s["as"] == 1 and s["es"] == 0, 10, label="activate")

    # Home
    st = get_status()
    if st and st["hs"] == 0:
        rep = send_and_get(robot_sock, "Home()", 4.0); print(" ->", rep)
        ok = wait_until(lambda s: s["hs"] == 1 and s["es"] == 0, 60, label="home")
        if not ok:
            print(" ->", send_and_get(robot_sock, "ResetError()", 1.0))
            print(" ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
            print(" ->", send_and_get(robot_sock, "Home()", 4.0))
            wait_until(lambda s: s["hs"] == 1 and s["es"] == 0, 60, label="home retry")

    # 완료 이벤트 on + 속도 제한 + 자동 구성
    print(" ->", send_and_get(robot_sock, "SetEom(1)", 1.0))
    print(" ->", send_and_get(robot_sock, f"SetCartLinVel({CART_LIN_V0})", 1.0))
    print(" ->", send_and_get(robot_sock, f"SetCartAngVel({CART_ANG_V0})", 1.0))
    print(" ->", send_and_get(robot_sock, "SetAutoConf(1)", 1.0))
    print(" ->", send_and_get(robot_sock, "SetAutoConfTurn(1)", 1.0))
    print(" ->", send_and_get(robot_sock, f"SetVelTimeout({VEL_TIMEOUT})", 1.0))

    print("[RCV] Robot ready: homed & no error")

# ---------- 10 Hz 실행 루프 ----------
def exec_loop_10hz():
    global _dp_acc, _stop_flag, _started
    t0 = time.time()
    last_status = t0
    while not _stop_flag:
        t_next = t0 + CTRL_DT
        t0 = t_next

        # 1) 누적 Δ 가져오고 즉시 초기화
        with _dp_lock:
            dp = _dp_acc[:]
            _dp_acc = [0.0] * 6

        # 2) 안전 클램프
        dp[0] = _clamp(dp[0], CLAMP_POS_MM)
        dp[1] = _clamp(dp[1], CLAMP_POS_MM)
        dp[2] = _clamp(dp[2], CLAMP_POS_MM)
        dp[3] = _clamp(dp[3], CLAMP_ANG_DEG)
        dp[4] = _clamp(dp[4], CLAMP_ANG_DEG)
        dp[5] = _clamp(dp[5], CLAMP_ANG_DEG)

        # 3) 상태 확인 후 실행
        st = get_status()
        if st and st["es"] == 0 and _started and any(abs(v) > 1e-9 for v in dp):
            try:
                send_cmd(robot_sock, f"MoveLinRelWrf({dp[0]},{dp[1]},{dp[2]},{dp[3]},{dp[4]},{dp[5]})")
                _ = try_recv_line(robot_sock, timeout=0.02)
            except Exception as e:
                print("[RCV] MoveLinRelWrf error:", e)

        # 1초 주기 상태 출력
        now = time.time()
        if now - last_status >= 1.0:
            if st:
                print(f"[RCV] as={st['as']} hs={st['hs']} es={st['es']} sm={st['sm']} eom={st['eom']} | lastΣΔ={dp}")
            last_status = now

        # 4) 타이밍 정렬
        dt = t_next - time.time()
        if dt > 0:
            time.sleep(dt)
        else:
            time.sleep(0.001)

# ---------- Sender handling ----------
def handle_sender(conn):
    global _started, _stop_flag
    f = conn.makefile("r", encoding="utf-8")
    for line in f:
        if not line.strip():
            continue
        msg = json.loads(line)
        cmd = msg.get("cmd")

        if cmd == "start":
            # a) 조인트/포즈로 시작점 이동
            if "start_joints" in msg:
                j = msg["start_joints"]
                print("[RCV] MoveJoints:", j)
                send_cmd(robot_sock, f"MoveJoints({j[0]},{j[1]},{j[2]},{j[3]},{j[4]},{j[5]})")
                ok = wait_until(lambda s: s["sm"] == 0 and s["eom"] == 1 and s["es"] == 0,
                                120, label="start joints")
                print("[RCV] MoveJoints idle:", ok)
            elif "start_pose" in msg:
                x, y, z, a, b, g = msg["start_pose"]
                print("[RCV] MovePose(abs):", msg["start_pose"])
                send_cmd(robot_sock, f"MovePose({x},{y},{z},{a},{b},{g})")
                ok = wait_until(lambda s: s["sm"] == 0 and s["eom"] == 1 and s["es"] == 0,
                                120, label="start pose")
                print("[RCV] MovePose idle:", ok)
            else:
                conn.sendall((json.dumps({"rcv": "error",
                                          "msg": "start requires start_joints or start_pose"}) + "\n").encode("utf-8"))
                continue

            # b) 컨피그/턴 고정 + AutoConf Off (고정)
            # conf = msg.get("conf")
            # if conf:
            #     cs = int(conf.get("cs", 1)); ce = int(conf.get("ce", 1)); cw = int(conf.get("cw", 1)); ct = int(conf.get("ct", 0))
            #     print(f"[RCV] Lock Conf: cs={cs} ce={ce} cw={cw} | Turn ct={ct}")
            #     print(" ->", send_and_get(robot_sock, f"SetConf({cs},{ce},{cw})", 1.0))
            #     print(" ->", send_and_get(robot_sock, f"SetConfTurn({ct})", 1.0))
            # # 어떤 경우든 AutoConf Off로 잠금
            # print(" ->", send_and_get(robot_sock, "SetAutoConf(0)", 1.0))
            # print(" ->", send_and_get(robot_sock, "SetAutoConfTurn(0)", 1.0))
            # _ = try_recv_line(robot_sock, timeout=0.1)  # 드레인

            # c) 실제 시작 포즈 계산 → ready 응답
            pose_line = send_and_get(robot_sock, "GetPose()", 1.0)
            start_pose_actual = _parse_pose_line(pose_line)
            conn.sendall((json.dumps({"rcv": "ready", "pose": start_pose_actual}) + "\n").encode("utf-8"))
            _started = True

        elif cmd == "dpose":
            if not _started:
                continue
            dp = msg.get("dp", [0, 0, 0, 0, 0, 0])
            with _dp_lock:
                for i in range(6):
                    _dp_acc[i] += float(dp[i])

        elif cmd == "apose":
            if not _started: continue
            x,y,z,a,b,g = msg["pose"]

            # --- J6 lock이 켜져 있으면 현재 J6 읽고 γ 보정 ---
            if _lock_j6 and _j6_target is not None:
                try:
                    send_cmd(robot_sock, "GetJoints()")
                    line = try_recv_line(robot_sock, timeout=0.2)  # 예: [2027][j1,j2,j3,j4,j5,j6]
                    if line and line.startswith("[2027]"):
                        payload = line.split("[",2)[2].split("]")[0]
                        j = [float(v) for v in payload.split(",")]
                        if len(j) == 6:
                            delta = j[5] - _j6_target  # 현재J6 - 목표J6
                            g = g - delta              # 툴 yaw를 반대로 보정
                except Exception as e:
                    print("[RCV] J6 lock read/adjust note:", e)

            try:
                send_cmd(robot_sock, f"MovePose({x},{y},{z},{a},{b},{g})")
                _ = try_recv_line(robot_sock, timeout=0.02)
            except Exception as e:
                print("[RCV] MovePose error:", e)


        elif cmd == "stop":
            print("[RCV] STOP")
            _stop_flag = True
            with _dp_lock:
                for i in range(6):
                    _dp_acc[i] = 0.0
            break

# ---------- Shutdown ----------
def graceful_shutdown():
    global robot_sock, _keepalive_run, _stop_flag, shutdown_done
    if shutdown_done:
        return
    shutdown_done = True
    print("\n[RCV] Graceful shutdown...")
    _stop_flag = True
    _keepalive_run = False
    time.sleep(0.1)

    es = 0
    try:
        st = get_status()
        if st:
            es = st["es"]
    except Exception:
        pass

    try:
        if es == 0:
            try:
                send_cmd(robot_sock, "MoveLinVelWrf(0,0,0,0,0,0)")
                _ = try_recv_line(robot_sock, timeout=0.3)
            except Exception as e:
                print(" -> MoveLinVelWrf(0) note:", e)
            for cmd in ("ClearMotion()", "ResumeMotion()"):
                try:
                    print(" ->", send_and_get(robot_sock, cmd, 0.5))
                except Exception as e:
                    print(f" -> {cmd} note:", e)
        else:
            try:
                print(" ->", send_and_get(robot_sock, "ResetError()", 0.5))
            except Exception as e:
                print(" -> ResetError() note:", e)

        try:
            print(" ->", send_and_get(robot_sock, "DeactivateRobot()", 0.5))
        except Exception as e:
            print(" -> DeactivateRobot() note:", e)
        try:
            print(" ->", send_and_get(robot_sock, "ConnectionWatchdog(0)", 0.5))
        except Exception as e:
            print(" -> ConnectionWatchdog(0) note:", e)
    finally:
        try:
            robot_sock.close()
        except Exception:
            pass
        print("[RCV] Shutdown complete.")

atexit.register(graceful_shutdown)
def _sig(signum, frame):
    graceful_shutdown(); sys.exit(0)
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

# ---------- Main ----------
def main():
    global robot_sock
    robot_sock = robot_connect()
    robot_init()

    # 10 Hz 실행 스레드 시작
    th = threading.Thread(target=exec_loop_10hz, daemon=True)
    th.start()

    # Sender 대기
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", LISTEN_PORT))
    srv.listen(1)
    print(f"[RCV] Waiting for sender on port {LISTEN_PORT}...")
    conn, addr = srv.accept()
    print("[RCV] Sender connected:", addr)
    try:
        handle_sender(conn)
    finally:
        try: conn.close()
        except: pass
        try: srv.close()
        except: pass
        graceful_shutdown()

if __name__ == "__main__":
    main()
