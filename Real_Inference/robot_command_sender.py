# robot_command_sender_lock_j6.py
import socket, json, time, math

RX_HOST, RX_PORT = "127.0.0.1", 5000

start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 30.0]  # J6 기준각
conf = {"cs": 1, "ce": 1, "cw": 1, "ct": 0}

# 시작/끝 포즈 (deg는 a,b,g = Rx,Ry,Rz)
start_pose = [190.0, 0.0, 308.0, 0.0, 90.0, 0.0]
end_pose   = [154.597847, 6.505294, 238.40975, 178.160805, 16.948613, 144.213525]

HZ       = 10.0
PERIOD   = 1.0 / HZ
DURATION = 10.0
N        = int(DURATION * HZ)

# ---------- rot helpers (XYZ euler: roll=a, pitch=b, yaw=g) ----------
def deg2rad(d): return d*math.pi/180.0
def rad2deg(r): return r*180.0/math.pi

def eul_xyz_to_R(a,b,g):
    ar, br, gr = map(deg2rad, (a,b,g))
    ca,sa = math.cos(ar), math.sin(ar)
    cb,sb = math.cos(br), math.sin(br)
    cg,sg = math.cos(gr), math.sin(gr)
    # R = Rz(g)*Ry(b)*Rx(a)
    Rz = [[cg,-sg,0],[sg,cg,0],[0,0,1]]
    Ry = [[cb,0,sb],[0,1,0],[-sb,0,cb]]
    Rx = [[1,0,0],[0,ca,-sa],[0,sa,ca]]
    def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mm(mm(Rz,Ry),Rx)

def R_to_quat(R):
    t = R[0][0]+R[1][1]+R[2][2]
    if t>0:
        s = math.sqrt(t+1.0)*2; w=0.25*s
        x=(R[2][1]-R[1][2])/s; y=(R[0][2]-R[2][0])/s; z=(R[1][0]-R[0][1])/s
    elif R[0][0]>R[1][1] and R[0][0]>R[2][2]:
        s = math.sqrt(1.0+R[0][0]-R[1][1]-R[2][2])*2; w=(R[2][1]-R[1][2])/s
        x=0.25*s; y=(R[0][1]+R[1][0])/s; z=(R[0][2]+R[2][0])/s
    elif R[1][1]>R[2][2]:
        s = math.sqrt(1.0+R[1][1]-R[0][0]-R[2][2])*2; w=(R[0][2]-R[2][0])/s
        x=(R[0][1]+R[1][0])/s; y=0.25*s; z=(R[1][2]+R[2][1])/s
    else:
        s = math.sqrt(1.0+R[2][2]-R[0][0]-R[1][1])*2; w=(R[1][0]-R[0][1])/s
        x=(R[0][2]+R[2][0])/s; y=(R[1][2]+R[2][1])/s; z=0.25*s
    return (w,x,y,z)

def quat_normalize(q):
    w,x,y,z = q; n = math.sqrt(w*w+x*x+y*y+z*z)
    return (w/n, x/n, y/n, z/n)

def quat_slerp(q0,q1,t):
    w0,x0,y0,z0 = q0; w1,x1,y1,z1 = q1
    dot = w0*w1 + x0*x1 + y0*y1 + z0*z1
    if dot < 0.0:
        w1,x1,y1,z1 = -w1,-x1,-y1,-z1; dot = -dot
    if dot > 0.9995:
        w = w0 + t*(w1-w0); x = x0 + t*(x1-x0); y = y0 + t*(y1-y0); z = z0 + t*(z1-z0)
        return quat_normalize((w,x,y,z))
    theta0 = math.acos(dot)
    s0 = math.sin((1.0-t)*theta0)/math.sin(theta0)
    s1 = math.sin(t*theta0)/math.sin(theta0)
    return (s0*w0 + s1*w1, s0*x0 + s1*x1, s0*y0 + s1*y1, s0*z0 + s1*z1)

def quat_to_eul_xyz(q):
    w,x,y,z = q
    R = [[1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
         [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
         [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]]
    pitch = math.asin(max(-1.0, min(1.0, R[0][2])))
    roll  = math.atan2(-R[1][2], R[2][2])
    yaw   = math.atan2(-R[0][1], R[0][0])
    return [rad2deg(roll), rad2deg(pitch), rad2deg(yaw)]

# 준비: 쿼터니언
R0 = eul_xyz_to_R(start_pose[3], start_pose[4], start_pose[5])
R1 = eul_xyz_to_R(end_pose[3],   end_pose[4],   end_pose[5])
q0 = quat_normalize(R_to_quat(R0))
q1 = quat_normalize(R_to_quat(R1))

def send_line(sock, obj): sock.sendall((json.dumps(obj)+"\n").encode("utf-8"))
def recv_json_line(sock, timeout=180.0):
    sock.settimeout(timeout); buf=b""
    while True:
        b = sock.recv(1)
        if not b: return None
        if b == b"\n":
            try: return json.loads(buf.decode("utf-8"))
            except: return None
        buf += b

with socket.create_connection((RX_HOST, RX_PORT)) as s:
    print("[SND] Connected to receiver")
    # J6 고정 요청
    send_line(s, {"cmd":"start",
                  "start_joints": start_joints,
                  "conf": conf,
                  "lock_j6": True})   # ★★★

    print("[SND] Waiting for ready...")
    ready = recv_json_line(s)
    if not ready or ready.get("rcv") != "ready":
        raise RuntimeError("Receiver not ready")

    # Receiver가 보내준 실제 시작 포즈(가능하면 사용)
    pose_from_rcv = ready.get("pose")
    if isinstance(pose_from_rcv, list) and len(pose_from_rcv) == 6:
        x0,y0,z0,a0,b0,g0 = pose_from_rcv
        # slerp 시작기준도 갱신(선택사항): 여기선 위치만 x0,y0,z0로 교체
        start_pose = [x0,y0,z0,start_pose[3],start_pose[4],start_pose[5]]
    else:
        x0,y0,z0,a0,b0,g0 = start_pose

    print(f"[SND] start_pose used: {[x0,y0,z0,a0,b0,g0]}")

    print("[SND] Streaming ABS pose @10Hz (pos linear + rot slerp)")
    t0 = time.time()
    for k in range(1, N+1):
        tau = k/float(N)
        x = x0 + tau*(end_pose[0]-x0)
        y = y0 + tau*(end_pose[1]-y0)
        z = z0 + tau*(end_pose[2]-z0)
        q = quat_slerp(q0, q1, tau)
        a,b,g = quat_to_eul_xyz(q)

        apose = [x,y,z,a,b,g]
        print(f"[SND] apose[{k}/{N}]: {apose}")
        send_line(s, {"cmd":"apose", "pose": apose})

        t_next = t0 + k*PERIOD
        dt = t_next - time.time()
        if dt>0: time.sleep(dt)

    send_line(s, {"cmd":"stop"})
    print("[SND] Done")
