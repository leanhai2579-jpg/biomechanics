from flask import Flask, request, jsonify, send_from_directory, Response, render_template
import json, os, cv2, time
import mediapipe as mp
import numpy as np
import threading
import datetime
import shutil

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Cấu hình 3 file lịch sử riêng biệt cho 3 bài tập cụ thể
HISTORY_SHOULDER_FILE = 'smartarm_history_shoulder.json'
HISTORY_ELBOW_FILE = 'smartarm_history_elbow.json'
HISTORY_HAND_FILE = 'smartarm_history_hand.json'

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='.')
# Tạo 2 danh sách toàn cục để lưu tiến trình trận tập
python_collected_angles = []
python_collected_times = []
python_actual_reps = 0
python_target_reps = 9          # <-- THÊM DÒNG NÀY
python_calculated_compliance = 0 # <-- THÊM DÒNG NÀY
start_exercise_time = None
USER_DATA_FILE = 'users.json'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_FILE = os.path.join(BASE_DIR, 'users.json')

video_writer = None
current_video_filename = ""
current_guidance_text = ""

class VideoRecorder:
    def __init__(self, filename, frame_size, fps=20):
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, self.fourcc, fps, frame_size)
        self.lock = threading.Lock()

    def write(self, frame):
        with self.lock:
            if self.out:
                self.out.write(frame)

    def stop(self):
        with self.lock:
            if self.out:
                self.out.release()
                self.out = None

# --- KHỞI TẠO MEDIAPIPE TOÀN CỤC ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- QUẢN LÝ TRẠNG THÁI ---
counters = {'LEFT': 0, 'RIGHT': 0}
stages = {'LEFT': None, 'RIGHT': None} 

hand_data = {
    "Left":  {"counter": 0, "stage": None, "color": (255, 255, 255)},
    "Right": {"counter": 0, "stage": None, "color": (255, 255, 255)}
}

elbow_counters = {'LEFT': 0, 'RIGHT': 0}
elbow_stages = {'LEFT': None, 'RIGHT': None}



error_frames_counter = 0

class KalmanFilter1D:
    def __init__(self, Q=0.01, R=0.1, P=1.0, X=0.0):
        self.Q = Q  # Nhiễu hệ thống
        self.R = R  # Nhiễu đo lường
        self.P = P  # Lỗi ước tính ban đầu
        self.x = X  # Giá trị ước tính ban đầu (Trạng thái lọc)

    def update(self, measurement):
        if self.x is None or self.x == 0.0:
            self.x = measurement
            return self.x
        self.P = self.P + self.Q
        kalman_gain = self.P / (self.P + self.R)
        self.x = self.x + kalman_gain * (measurement - self.x)
        self.P = (1 - kalman_gain) * self.P
        return self.x

# Cấu hình tham số Q, R mịn màng dựa theo đặc trưng biến động của từng bài tập
kf_shoulder_torso = KalmanFilter1D(Q=0.005, R=0.05)
kf_elbow_torso = KalmanFilter1D(Q=0.005, R=0.05)
kf_hand_ratio_l = KalmanFilter1D(Q=0.005, R=0.05)
kf_hand_ratio_r = KalmanFilter1D(Q=0.005, R=0.05)


def check_torso_rotation(landmarks, exercise_active=False, holding_high=False, is_in_rest_zone=False):
    l_sh = [landmarks[11].x, landmarks[11].y]
    r_sh = [landmarks[12].x, landmarks[12].y]
    l_hip = [landmarks[23].x, landmarks[23].y]
    r_hip = [landmarks[24].x, landmarks[24].y]

    # Tính toán trung điểm vai (midSh) và trung điểm hông (midHip)
    mid_sh = [(l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2]
    mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

    # 1. Tính góc giữa trục vai và trục cột sống
    sh_spine_angle = calculate_angle_between_vectors(l_sh, r_sh, mid_sh, mid_hip)
    if abs(sh_spine_angle - 90) > 5:
        return False, "VAI DANG BI LECH"

    # 2.  Tính góc nghiêng của cột sống bằng arctan2(x, y)
    spine_vec_x = mid_hip[0] - mid_sh[0]
    spine_vec_y = mid_hip[1] - mid_sh[1]
    lean_angle = np.degrees(np.arctan2(spine_vec_x, spine_vec_y))

    allowed_spine_dev = 12.0 if is_in_rest_zone else 5.0
    allowed_lean_dev = 10.0 if is_in_rest_zone else 5.0

    if abs(sh_spine_angle - 90) > allowed_spine_dev:
        return False, "VAI DANG BI LECH"

    if abs(lean_angle) > allowed_lean_dev:
        return False, "NGUOI DANG BI NGHIENG"

    return True, "GOOD"

def calculate_angle_between_vectors(p1, p2, p3, p4):
    # Vector 1: Nối từ p1 đến p2 (Trục vai)
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    # Vector 2: Nối từ p3 đến p4 (Trục cột sống)
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 90.0
        
    cosine_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_position_guidance(landmarks, w, h):
    # Tọa độ các điểm chính
    # Tọa độ các điểm chính
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    
    torso_height = abs(l_sh.y - l_hip.y)
    if torso_height > 0.35: 
        return "HAY LUI LAI, QUA GAN ROI", (0, 0, 255)
    if torso_height < 0.20: 
        return "HAY TIEN LEN", (0, 0, 255)

    # 🌟 TẦNG 2: CHECK TRÁI / PHẢI
    center_body_x = (l_sh.x + r_sh.x) / 2
    if center_body_x < 0.40: 
        return "HAY DI CHUYEN SANG PHAI", (0, 165, 255)
    if center_body_x > 0.60: 
        return "HAY DI CHUYEN SANG TRAI", (0, 165, 255)

    # 🌟 TẦNG 3: CHECK KHOẢNG TRỐNG TRÊN ĐẦU (Bài vai bắt buộc phải có để không mất mốc tay)
    space_above = min(l_sh.y, r_sh.y) 
    if space_above < (torso_height * 1.2):
        return "LUI LAI, CAN KHOANG TRONG DE DUOI TAY", (0, 0, 255)

    # 🌟 TẦNG 4: CHECK THẲNG CAMERA (Chỉ check khi các tầng trên đã chuẩn)
    is_aligned_hip, _ = check_torso_rotation(landmarks)
    is_level_shoulder, _ = check_shoulder_tilt(landmarks)
    if not is_aligned_hip or not is_level_shoulder:
        return "GIU TU THE THANG VOI CAMERA", (0, 165, 255)

    return "PERFECT! START EXERCISE", (0, 255, 0)

def check_shoulder_tilt(landmarks):
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    
    # Tính độ chênh lệch chiều dọc (y) giữa 2 vai
    delta_y = abs(l_sh.y - r_sh.y)
    # Tính khoảng cách chiều ngang (x) giữa 2 vai
    shoulder_width = abs(l_sh.x - r_sh.x)
    
    if shoulder_width == 0: return True, 0
    tilt_ratio = delta_y / shoulder_width
    
    # Ngưỡng 0.12 tương đương khoảng 7-8 độ lệch
    is_level = tilt_ratio < 0.07
    return is_level, tilt_ratio

def get_position_guidance_elbow(landmarks, w, h, exercise_active=False):
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    
    torso_height = abs(l_sh.y - l_hip.y)
    if torso_height > 0.40:  
        return "HAY LUI LAI, QUA GAN ROI", (0, 0, 255)
    if torso_height < 0.30:  
        return "HAY TIEN LEN", (0, 0, 255)

    # 🌟 TẦNG 2: CHECK TRÁI / PHẢI
    center_body_x = (l_sh.x + r_sh.x) / 2
    if center_body_x < 0.38: 
        return "HAY DI CHUYEN SANG PHAI", (0, 165, 255)
    if center_body_x > 0.62: 
        return "HAY DI CHUYEN SANG TRAI", (0, 165, 255)

    # 🌟 TẦNG 3: CHECK THẲNG CAMERA
    is_aligned, _ = check_torso_rotation(landmarks, exercise_active=exercise_active)
    if not is_aligned:
        return "GIU TU THE THANG VOI CAMERA", (0, 165, 255)

    return "PERFECT! START EXERCISE", (0, 255, 0)

def check_hand_alignment(hand_landmarks, is_open=True):
    # 5: Gốc ngón trỏ, 17: Gốc ngón út
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]

    delta_z = abs(index_mcp.z - pinky_mcp.z)
    hand_width_x = abs(index_mcp.x - pinky_mcp.x)
    
    if hand_width_x == 0: return True
    rotation_ratio = delta_z / hand_width_x

    # NẾU ĐANG XÒE TAY (Chuẩn bị): Ngưỡng 0.4 (Khắt khe)
    # NẾU ĐANG CO TAY (Thực hiện): Ngưỡng 0.65 (Nới lỏng để không bị ngắt Rep)
    #threshold = 0.4 if is_open else 0.65
    threshold = 0.3 if is_open else 0.4
    
    
    return rotation_ratio < threshold

def check_hand_facing(hand_landmarks, label):
    # Lấy điểm Cổ tay (0), Gốc trỏ (5), Gốc út (17)
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    index = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
    pinky = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])

    # Tính Vector pháp tuyến của mặt phẳng bàn tay
    v1 = index - wrist
    v2 = pinky - wrist
    normal = np.cross(v1, v2) 

    # Dựa vào nhãn tay (Left/Right) để biết hướng lòng bàn tay
    # Với tay phải (Right), lòng bàn tay chuẩn sẽ có normal.z < 0 (do flip frame)
    # Với tay trái (Left), lòng bàn tay chuẩn sẽ có normal.z > 0
    if label == "Right":
        return normal[2] < 0
    else:
        return normal[2] > 0
    
def get_position_guidance_hand(landmarks, hand_results, w, h):
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    
    # 🌟 TẦNG 1: CHECK VỊ TRÍ ĐỨNG XA / GẦN CỦA THÂN NGƯỜI (Ép bệnh nhân phải đứng sát camera hơn bài vai)
    torso_height = abs(l_sh.y - l_hip.y)
    if torso_height > 0.70:   
        return "HAY LUI LAI, QUA GAN ROI", (0, 0, 255)
    if torso_height < 0.50:   
        return "HAY TIEN LEN", (0, 0, 255)

    # 🌟 TẦNG 2: CHECK VỊ TRÍ ĐỨNG TRÁI / PHẢI
    center_body_x = (l_sh.x + r_sh.x) / 2
    if center_body_x < 0.38: 
        return "HAY DI CHUYEN SANG PHAI", (0, 165, 255)
    if center_body_x > 0.62: 
        return "HAY DI CHUYEN SANG TRAI", (0, 165, 255)

    # 🌟 TẦNG 3: CHECK XEM ĐÃ GIƠ BÀN TAY LÊN VÙNG QUÉT CHƯA
    if not hand_results or not hand_results.multi_hand_landmarks:
        return "HUONG LONG BAN TAY THANG VAO CAMERA", (0, 165, 255)

    # TẦNG 3: ĐÃ GIƠ TAY -> KIỂM TRA KHOẢNG CÁCH XA GẦN CỦA RIÊNG BÀN TAY
    for hand_lms in hand_results.multi_hand_landmarks:
        wrist = hand_lms.landmark[0]
        mcp = hand_lms.landmark[9]
        
        hand_size = np.sqrt(((mcp.x - wrist.x) * w)**2 + ((mcp.y - wrist.y) * h)**2)
        
        if hand_size > 145:
            return "HAY DUA BAN TAY RA XA CAMERA MOT CHUT", (0, 0, 255)
        if hand_size < 90:
            return "HAY DUA BAN TAY LAI GAN CAMERA MOT CHUT", (0, 0, 255)
    
    return "PERFECT! START EXERCISE", (0, 255, 0)
    
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def get_shoulder_eval(angle, side):
    global counters, stages
    UP_THRESHOLD, DOWN_THRESHOLD = 140, 40
    if angle > UP_THRESHOLD:
        stages[side] = "up"
        return "DANG VAI TOI DA", (0, 255, 0)
    elif angle < DOWN_THRESHOLD:
        if stages[side] == 'up': 
            counters[side] += 1
            pass
        stages[side] = "down"
        return "TAY KHEP SAT THAN", (200, 200, 200)
    return "LIFTING...", (0, 165, 255)

def get_eval_elbow(angle):
    if angle <= 50: return "GAP TAY TOT", (0, 255, 0)
    elif angle >= 150: return "DUOI THANG TOT", (0, 255, 0)
    return "DANG CO DUOI...", (255, 255, 255)



# --- LUỒNG XỬ LÝ 1: VAI (SHOULDER) ---
def gen_shoulder(target_sets, target_reps, target_angle=None):
    def get_shoulder_eval(angle, side):
        global counters, stages
        if target_angle:
            UP_THRESHOLD = target_angle
            DOWN_THRESHOLD = 40
        else:
            UP_THRESHOLD, DOWN_THRESHOLD = 140, 40
        if angle > UP_THRESHOLD:
            stages[side] = "up"
            return "DANG VAI TOI DA", (0, 255, 0)
        elif angle < DOWN_THRESHOLD:
            if stages[side] == 'up': 
                counters[side] += 1
                pass
            stages[side] = "down"
            return "TAY KHEP SAT THAN", (200, 200, 200)
        return "LIFTING...", (0, 165, 255)
    global counters, stages
    global video_writer, current_video_filename
    
    # --- PHẦN QUẢN LÝ SET/REP ---
    current_rep_in_set = 0
    current_set = 1

    # Thêm biến để tạo độ trễ khi hạ tay xong mới ẩn UI
    hide_delay_counter = 0

    waiting_for_correct_posture = True
    exercise_finished = False
    # ----------------------------
    position_confirmed = False
    counters['LEFT'] = 0
    counters['RIGHT'] = 0

    error_frames_counter = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    shoulder_error_counter = 0
    guidance = ""
    color = (255, 255, 255)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while True:
            success, frame = cap.read()
            if not success: break

            # --- ĐOẠN DÁN CODE LƯU VIDEO ---
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                current_video_filename = f"video_vai_{current_time}.mp4"
                video_writer = cv2.VideoWriter(current_video_filename, fourcc, 20.0, (1280, 720))
            
            video_writer.write(frame)
            # -------------------------------

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

               # --- 1. VẼ KHUNG THÂN NGƯỜI (Dùng code của bạn) ---
                l_s = tuple(np.multiply([landmarks[11].x, landmarks[11].y], [w, h]).astype(int))
                r_s = tuple(np.multiply([landmarks[12].x, landmarks[12].y], [w, h]).astype(int))
                l_h = tuple(np.multiply([landmarks[23].x, landmarks[23].y], [w, h]).astype(int))
                r_h = tuple(np.multiply([landmarks[24].x, landmarks[24].y], [w, h]).astype(int))
                
                cv2.line(frame, l_s, r_s, (255, 255, 255), 2)
                cv2.line(frame, r_s, r_h, (255, 255, 255), 2)
                cv2.line(frame, r_h, l_h, (255, 255, 255), 2)
                cv2.line(frame, l_h, l_s, (255, 255, 255), 2)
 
                is_holding_high = False
                try:
                    for ids in [[24,12,14], [23,11,13]]: 
                        pts = [[landmarks[i].x, landmarks[i].y] for i in ids]
                        tmp_angle = calculate_angle(pts[0], pts[1], pts[2])
                        if tmp_angle > 140: 
                            is_holding_high = True
                except: pass

                # --- 2. LOGIC TÁCH THỜI ĐIỂM CHECK VỊ TRÍ ---

                # Tính toán torso_height thô để cập nhật bộ lọc Kalman
                raw_torso_height = abs(landmarks[11].y - landmarks[23].y)
                smooth_torso_height = kf_shoulder_torso.update(raw_torso_height)
                
                # Cập nhật lại trục Y của Hông theo giá trị đã lọc để đồng bộ dữ liệu cho các hàm định vị
                if landmarks[23].y > landmarks[11].y:
                    landmarks[23].y = landmarks[11].y + smooth_torso_height
                    landmarks[24].y = landmarks[12].y + smooth_torso_height
                else:
                    landmarks[23].y = landmarks[11].y - smooth_torso_height
                    landmarks[24].y = landmarks[12].y - smooth_torso_height

                is_in_rest_zone = True
                try:
                    hip_l, sho_l, elb_l = [landmarks[i] for i in [24, 12, 14]]
                    hip_r, sho_r, elb_r = [landmarks[i] for i in [23, 11, 13]]
                    ang_l = calculate_angle([hip_l.x, hip_l.y], [sho_l.x, sho_l.y], [elb_l.x, elb_l.y])
                    ang_r = calculate_angle([hip_r.x, hip_r.y], [sho_r.x, sho_r.y], [elb_r.x, elb_r.y])
                    
                    # Nếu có bất kỳ tay nào nâng lên cao hơn 35 độ -> Thoát vùng nghỉ
                    if ang_l > 35 or ang_r > 35:
                        is_in_rest_zone = False
                except:
                    is_in_rest_zone = False

                is_level_sh, _ = check_shoulder_tilt(landmarks)
                is_aligned_hip, _ = check_torso_rotation(landmarks, is_in_rest_zone=is_in_rest_zone)
                raw_guidance, raw_color = get_position_guidance(landmarks, w, h)
                #is_posture_ok = is_level_sh and is_aligned_hip and raw_guidance == "PERFECT! START EXERCISE"
                is_posture_ok_with_angle, error_msg = check_torso_rotation(landmarks, is_in_rest_zone=is_in_rest_zone)
                
                is_posture_ok = is_level_sh and is_posture_ok_with_angle and raw_guidance == "PERFECT! START EXERCISE"

                if exercise_finished:
                    raw_guidance, raw_color = "DA HOAN THANH BAI TAP!", (0, 255, 0)
                elif position_confirmed and not is_in_rest_zone:
                    # Khi đang nâng tay tập, khóa cứng thông báo để không bị nháy chữ do che khuất mốc
                    raw_guidance, raw_color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                else:
                    is_rest_flexible = (ang_l < 30 or ang_r < 30)
                    # TẦNG 1: KIỂM TRA KHOẢNG CÁCH XA / GẦN TRƯỚC TIÊN
                    if "LUI LAI" in raw_guidance or "TIEN LEN" in raw_guidance or "KHOANG TRONG" in raw_guidance:
                        raw_guidance, raw_color = raw_guidance, raw_color
                    
                    # TẦNG 2: KIỂM TRA VỊ TRÍ TRÁI / PHẢI (Chỉ check khi khoảng cách đã chuẩn)
                    elif "SANG TRAI" in raw_guidance or "SANG PHAI" in raw_guidance:
                        raw_guidance, raw_color = raw_guidance, raw_color
                    
                    # TẦNG 3: KIỂM TRA THẲNG CAMERA (LỆCH VAI / NGHIÊNG NGƯỜI)
                    elif not is_level_sh or not is_posture_ok_with_angle:
                        raw_guidance, raw_color = "GIU TU THE THANG VOI CAMERA", (0, 0, 255)
                    
                    # TẦNG CUỐI: TẤT CẢ ĐỀU ĐẠT CHUẨN
                    else:
                        raw_guidance, raw_color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                        if not position_confirmed and is_posture_ok:
                            position_confirmed = True

                # 🌟 BỘ LỌC ĐỘ TRỄ KHUNG HÌNH (DEBOUNCE): Phải lỗi liên tiếp 10 frames mới hiện chữ đỏ
                if raw_guidance != "VI TRI CHUAN! BAT DAU TAP" and raw_guidance != "DA HOAN THANH BAI TAP!":
                    shoulder_error_counter += 1
                    # Nếu đứng sai liên tục đủ 35 frames (~1.5 giây) thì mới công nhận lỗi và đổi chữ đỏ
                    if shoulder_error_counter >= 35:
                        guidance = raw_guidance
                        color = raw_color
                else:
                    # Khi đo đạc báo chuẩn, ta cũng bắt phải chuẩn liên tiếp ít nhất 10 frames 
                    # thì mới tin tưởng xóa lỗi cũ và reset bộ đếm, tránh việc bị reset bậy giữa chừng
                    shoulder_error_counter -= 2  # Trừ lùi dần thay vì xóa sạch về 0
                    if shoulder_error_counter <= 0:
                        shoulder_error_counter = 0
                        guidance = raw_guidance
                        color = raw_color

                
                # Vẽ bảng thông báo phía trên khung thân
                cv2.rectangle(frame, (w//2-250, 20), (w//2+250, 80), (0,0,0), -1)
                cv2.putText(frame, guidance, (w//2-230, 60), 1, 1.8, color, 3)

                global current_guidance_text
                current_guidance_text = guidance

                # --- 3. VÒNG LẶP HIỂN THỊ VAI VÀ ĐẾM REPS ---
                if position_confirmed and (guidance == "VI TRI CHUAN! BAT DAU TAP"): 
                    for side, ids in [('LEFT', [24,12,14,16]), ('RIGHT', [23,11,13,15])]:
                        try:
                            hip, sho, elb, wri = [[landmarks[i].x, landmarks[i].y] for i in ids]
                            angle = np.clip(0.9955 * calculate_angle(hip, sho, elb) + 2.36, 0, 180)
                           
                            #---Lưu dữ liệu đồ thị vai
                            global python_collected_times, start_exercise_time
                            if start_exercise_time is None:
                                 start_exercise_time = time.time()
                            elapsed = round(time.time() - start_exercise_time, 1)
                            if len(python_collected_times) == 0 or elapsed - float(python_collected_times[-1].replace('s','')) >= 1.0:
                                 python_collected_times.append(f"{int(elapsed)}s")
                                 python_collected_angles.append(int(angle))
                            real_side = 'RIGHT' if side == 'LEFT' else 'LEFT'

                            old_rep_count = counters[real_side]
                            # Lấy đánh giá từ hàm gốc (tự động cập nhật counters[real_side])
                            eval_t, eval_c = get_shoulder_eval(angle, real_side)
                            

                            if counters[real_side] > old_rep_count:
                                global python_actual_reps
                                python_actual_reps += 1
                                waiting_for_correct_posture = True
                                position_confirmed = False
                                hide_delay_counter = 1
                            # LOGIC VÒNG LẶP: Check vị trí sau mỗi Rep
                            if not exercise_finished and not waiting_for_correct_posture:
                                # Nếu bên đang tập vừa nhảy số (Sử dụng stages để biết vừa hạ tay)
                                if stages[real_side] == "down" and angle < 5:
                                   waiting_for_correct_posture = True
                                   position_confirmed = False

                                   if angle < 5: # Đảm bảo tay đã hạ xuống sát người
                                        hide_delay_counter += 1
                                    
                                    # Đợi khoảng 10 frame (~0.5s) ở tư thế hạ tay rồi mới reset
                                   if hide_delay_counter > 35:
                                        waiting_for_correct_posture = True
                                        
                                        hide_delay_counter = 0
                                        position_confirmed = False
                                   # Kiểm tra chuyển Set
                                   if counters[real_side] >= target_reps:
                                        if current_set >= target_sets:
                                           exercise_finished = True
                                        else:
                                           current_set += 1
                                           counters['LEFT'] = 0
                                           counters['RIGHT'] = 0

                            max_allowed_reps_in_this_set = current_set * target_reps
                            actual_side_reps = min(counters[real_side], max_allowed_reps_in_this_set)

                            # Chia lấy dư để Rep hiển thị luôn reset từ 0 -> target_reps
                            side_rep_in_set = actual_side_reps % target_reps
                            if side_rep_in_set == 0 and actual_side_reps > 0:
                                side_rep_in_set = target_reps

                            # Vẽ UI thông số (Chỉ hiện khi đang trong phase tập)
                            s_p, e_p, w_p = [tuple(np.multiply(c, [w, h]).astype(int)) for c in [sho, elb, wri]]
                            cv2.line(frame, s_p, e_p, (255, 255, 255), 4)
                            cv2.line(frame, e_p, w_p, (255, 255, 255), 4)
                            cv2.circle(frame, s_p, 10, eval_c, -1)

                            pos_y = 60 if side == 'LEFT' else 200
                            box_color = (245, 117, 16) if side == 'LEFT' else (117, 66, 245)
                            cv2.rectangle(frame, (0, pos_y-45), (280, pos_y+85), box_color, -1)
                            cv2.putText(frame, f"{side}: {int(angle)} deg", (10, pos_y), 1, 1.2, (255,255,255), 2)
                            cv2.putText(frame, eval_t, (10, pos_y+35), 1, 1.2, eval_c, 2)
                            cv2.putText(frame, f"SET:{current_set} REP:{side_rep_in_set}", (10, pos_y+75), 1, 1.8, (0,255,255), 3)
                        except Exception as e:
                            pass

                        left_reps = counters['LEFT']
                        right_reps = counters['RIGHT']
                        common_reps = min(left_reps, right_reps)
                    

                        if common_reps > 0:
                            calculated_set = ((common_reps - 1) // target_reps) + 1
                        else:
                            calculated_set = 1

                    # Tính toán Set tịnh tiến đồng bộ đưa lên đầu chu kỳ
                        #calculated_set = (common_reps // target_reps) + 1
                    
                    # CHẶN LUỒNG NGẦM GHI ĐÈ: Chỉ cập nhật khi chưa hoàn thành bài tập và common_reps > 0
                        if not exercise_finished and common_reps > 0:
                            current_set = min(calculated_set, target_sets)
                            python_actual_reps = common_reps  

                        if common_reps >= (target_sets * target_reps):
                            exercise_finished = True
                            guidance = "DA HOAN THANH BAI TAP!"
                            color = (0, 255, 0)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- LUỒNG XỬ LÝ 2: KHUỶU TAY (ELBOW) ---
def gen_elbow(target_sets, target_reps, target_angle=None):
    def get_eval_elbow(angle):
        global elbow_counters, elbow_stages
        if target_angle:
            UP_THRESHOLD = target_angle
            DOWN_THRESHOLD = 160
        else:
            UP_THRESHOLD, DOWN_THRESHOLD = 50, 160
        if angle <= UP_THRESHOLD: return "GAP TAY TOT", (0, 255, 0)
        elif angle >= DOWN_THRESHOLD: return "DUOI THANG TOT", (0, 255, 0)
        return "DANG CO DUOI...", (255, 255, 255)
        
    global elbow_counters, elbow_stages
    global video_writer, current_video_filename
    # --- PHẦN QUẢN LÝ SET/REP ---
    current_rep_in_set = 0
    current_set = 1

    # Thêm biến để tạo độ trễ khi hạ tay xong mới ẩn UI
    hide_delay_counter = 0

    waiting_for_correct_posture = True
    exercise_finished = False
    # ----------------------------
    position_confirmed = False
    elbow_counters['LEFT'] = 0
    elbow_counters['RIGHT'] = 0
    elbow_stages['LEFT'] = "down"
    elbow_stages['RIGHT'] = "down"
    
    guidance = ""
    color = (255, 255, 255)
    global python_actual_reps
    python_actual_reps = 0

    error_frames_counter = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elbow_error_counter = 0
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while True:
            success, frame = cap.read()
            if not success: break

            # --- ĐOẠN DÁN CODE LƯU VIDEO ---
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                current_video_filename = f"video_khuyutay_{current_time}.mp4" 
                video_writer = cv2.VideoWriter(current_video_filename, fourcc, 20.0, (1280, 720))
            
            video_writer.write(frame)
            # -------------------------------

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # --- 1. LẤY TỌA ĐỘ KHUNG THÂN NGƯỜI ---
                l_s = tuple(np.multiply([landmarks[11].x, landmarks[11].y], [w, h]).astype(int))
                r_s = tuple(np.multiply([landmarks[12].x, landmarks[12].y], [w, h]).astype(int))
                l_h = tuple(np.multiply([landmarks[23].x, landmarks[23].y], [w, h]).astype(int))
                r_h = tuple(np.multiply([landmarks[24].x, landmarks[24].y], [w, h]).astype(int))
                
                
                raw_guidance, raw_color = get_position_guidance_elbow(landmarks, w, h)

                # Vẽ khung thân người (Đổi màu theo trạng thái)
                box_color = (255, 255, 255) if guidance != "PERFECT! START EXERCISE" else (0, 255, 0)
                cv2.line(frame, l_s, r_s, box_color, 2)
                cv2.line(frame, r_s, r_h, box_color, 2)
                cv2.line(frame, r_h, l_h, box_color, 2)
                cv2.line(frame, l_h, l_s, box_color, 2)

                is_holding_high = False
                try:
                    for ids in [[12,14,16], [11,13,15]]: 
                        pts = [[landmarks[i].x, landmarks[i].y] for i in ids]
                        tmp_angle = calculate_angle(pts[0], pts[1], pts[2])
                        if tmp_angle > 140: 
                            is_holding_high = True
                except: pass

                # --- 2. LOGIC TÁCH THỜI ĐIỂM CHECK VỊ TRÍ ---
                raw_torso_height = abs(landmarks[11].y - landmarks[23].y)
                smooth_torso_height = kf_elbow_torso.update(raw_torso_height)
                
                if landmarks[23].y > landmarks[11].y:
                    landmarks[23].y = landmarks[11].y + smooth_torso_height
                    landmarks[24].y = landmarks[12].y + smooth_torso_height
                else:
                    landmarks[23].y = landmarks[11].y - smooth_torso_height
                    landmarks[24].y = landmarks[12].y - smooth_torso_height

                is_in_rest_zone = True
                try:
                    hip_l, sho_l, elb_l = [landmarks[i] for i in [24, 12, 14]]
                    hip_r, sho_r, elb_r = [landmarks[i] for i in [23, 11, 13]]
                    ang_l = calculate_angle([hip_l.x, hip_l.y], [sho_l.x, sho_l.y], [elb_l.x, elb_l.y])
                    ang_r = calculate_angle([hip_r.x, hip_r.y], [sho_r.x, sho_r.y], [elb_r.x, elb_r.y])
                    
                    # Nếu có bất kỳ tay nào nâng lên cao hơn 35 độ -> Thoát vùng nghỉ
                    if ang_l > 35 or ang_r > 35:
                        is_in_rest_zone = False
                except:
                    is_in_rest_zone = False

                is_level_sh, _ = check_shoulder_tilt(landmarks)
                is_posture_ok_with_angle, error_msg = check_torso_rotation(landmarks, is_in_rest_zone=is_in_rest_zone)
                raw_guidance, raw_color = get_position_guidance_elbow(landmarks, w, h)
                
                is_posture_ok = is_level_sh and is_posture_ok_with_angle and (raw_guidance == "PERFECT! START EXERCISE" or raw_guidance == "VI TRI CHUAN! BAT DAU TAP")

                if exercise_finished:
                    raw_guidance, raw_color = "DA HOAN THANH BAI TAP!", (0, 255, 0)
                elif position_confirmed and not is_in_rest_zone:
                    # Khi đang nâng tay tập, khóa cứng thông báo để không bị nháy chữ do che khuất mốc
                    raw_guidance, raw_color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                else:
                    is_rest_flexible = (ang_l < 30 or ang_r < 30)
                    # TẦNG 1: KIỂM TRA KHOẢNG CÁCH XA / GẦN TRƯỚC TIÊN
                    if "LUI LAI" in raw_guidance or "TIEN LEN" in raw_guidance or "KHOANG TRONG" in raw_guidance:
                        raw_guidance, raw_color = raw_guidance, raw_color
                    
                    # TẦNG 2: KIỂM TRA VỊ TRÍ TRÁI / PHẢI (Chỉ check khi khoảng cách đã chuẩn)
                    elif "SANG TRAI" in raw_guidance or "SANG PHAI" in raw_guidance:
                        raw_guidance, raw_color = raw_guidance, raw_color
                    
                    # TẦNG 3: KIỂM TRA THẲNG CAMERA (LỆCH VAI / NGHIÊNG NGƯỜI)
                    elif not is_level_sh or not is_posture_ok_with_angle:
                        raw_guidance, raw_color = "GIU TU THE THANG VOI CAMERA", (0, 0, 255)
                    
                    # TẦNG CUỐI: TẤT CẢ ĐỀU ĐẠT CHUẨN
                    else:
                        raw_guidance, raw_color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                        if not position_confirmed and is_posture_ok:
                            position_confirmed = True

                # 🌟 BỘ LỌC ĐỘ TRỄ KHUNG HÌNH CHO BÀI KHUỶU TAY (Đồng bộ theo biến gốc)
                if raw_guidance != "VI TRI CHUAN! BAT DAU TAP" and raw_guidance != " DA HOAN THANH BAI TAP!":
                    elbow_error_counter += 1
                    if elbow_error_counter >= 35:
                        guidance = raw_guidance
                        color = raw_color
                else:
                    # Trừ lùi bộ đếm lỗi để kiểm tra độ ổn định thực tế của tư thế chuẩn
                    elbow_error_counter -= 2
                    if elbow_error_counter <= 0:
                        elbow_error_counter = 0
                        guidance = raw_guidance
                        color = raw_color

                # Hiển thị thông báo hướng dẫn ra màn hình
                cv2.rectangle(frame, (w//2-250, 15), (w//2+250, 75), (0,0,0), -1)
                cv2.putText(frame, guidance, (w//2-230, 55), 1, 1.8, color, 3)
                
                global current_guidance_text
                current_guidance_text = guidance
                # --- 3. VÒNG LẶP HIỂN THỊ VÀ TÍNH TOÁN REPS KHUỶU TAY ---
                if position_confirmed and (guidance == "VI TRI CHUAN! BAT DAU TAP"):
                    for side, ids in [('LEFT', [12, 14, 16]), ('RIGHT', [11, 13, 15])]:
                        try:
                            s_r, e_r, w_r = [[landmarks[i].x, landmarks[i].y] for i in ids]
                            angle = np.clip(0.962 * calculate_angle(s_r, e_r, w_r) + 4.25, 0, 180) 
                            
                            # --- LƯU DỮ LIỆU ĐỒ THỊ BÀI KHUỶU TAY ---
                            global python_collected_angles, python_collected_times, start_exercise_time
                            if start_exercise_time is None:
                                start_exercise_time = time.time()
                            elapsed = round(time.time() - start_exercise_time, 1)
                            if len(python_collected_times) == 0 or elapsed - float(python_collected_times[-1].replace('s','')) >= 1.0:
                                python_collected_times.append(f"{int(elapsed)}s")
                                python_collected_angles.append(int(angle))

                            real_side = 'RIGHT' if side == 'LEFT' else 'LEFT'
                            old_rep_count = elbow_counters[real_side]
                            eval_t, eval_c = get_eval_elbow(angle)

                            if angle < 50: 
                                elbow_stages[real_side] = "up"
                            if angle > 150 and elbow_stages[real_side] == "up":
                                elbow_stages[real_side] = "down"
                                elbow_counters[real_side] += 1

                            if elbow_counters[real_side] > old_rep_count:
                                
                                #python_actual_reps += 1
                                waiting_for_correct_posture = True
                                position_confirmed = False
                                hide_delay_counter = 1

                            # LOGIC VÒNG LẶP: Check vị trí sau mỗi Rep
                            if not exercise_finished and not waiting_for_correct_posture:
                                if elbow_stages[real_side] == "down" and angle > 175:
                                    if angle > 175: 
                                        hide_delay_counter += 1
                                    
                                    if hide_delay_counter > 35:
                                        waiting_for_correct_posture = True
                                        hide_delay_counter = 0
                                        position_confirmed = False

                            # KHÓA REPS: Tay nhanh không được vượt quá mốc của Set hiện tại khi tay chậm chưa đủ rep
                            max_allowed_reps_in_this_set = current_set * target_reps
                            actual_side_reps = min(elbow_counters[real_side], max_allowed_reps_in_this_set)

                            side_rep_in_set = actual_side_reps % target_reps
                            if side_rep_in_set == 0 and actual_side_reps > 0:
                                side_rep_in_set = target_reps

                            # Vẽ xương và khớp khuỷu tay
                            sp, ep, wp = [tuple(np.multiply(c, [w, h]).astype(int)) for c in [s_r, e_r, w_r]]
                            cv2.line(frame, sp, ep, (255, 255, 255), 3)
                            cv2.line(frame, ep, wp, (255, 255, 255), 3)
                            cv2.circle(frame, ep, 12, eval_c, -1)

                            # UI Khuỷu tay
                            pos_y = 50 if side == 'LEFT' else 170
                            box_color = (245, 117, 16) if side == 'LEFT' else (117, 66, 245)
                            cv2.rectangle(frame, (0, pos_y-35), (260, pos_y+75), box_color, -1)
                            cv2.putText(frame, f"{side} ELBOW: {int(angle)} deg", (10, pos_y), 1, 1.2, (255,255,255), 2)
                            cv2.putText(frame, eval_t, (10, pos_y+35), 1, 1.1, eval_c, 2)
                            cv2.putText(frame, f"SET:{current_set} REPS: {side_rep_in_set}", (10, pos_y+65), 1, 1.3, (0,255,255), 2)
                        except Exception as e:
                            pass

                    #left_reps = elbow_counters['LEFT']    # 🌟 SỬA TỪ counters THÀNH elbow_counters
                    #right_reps = elbow_counters['RIGHT']  # 🌟 SỬA TỪ counters THÀNH elbow_counters
                    #actual_reps = min(left_reps, right_reps)
                    
                    # Cập nhật python_actual_reps là số reps chung
                    #python_actual_reps = actual_reps
                    
                    # Vẽ thêm thông báo common reps lên màn hình
                    #cv2.rectangle(frame, (w//2 - 180, 130), (w//2 + 180, 180), (0, 0, 0), -1)
                    #cv2.putText(frame, f"ACTUAL REPS: {actual_reps}", (w//2 - 150, 165), 
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
                    common_reps = min(elbow_counters['LEFT'], elbow_counters['RIGHT'])
                    if common_reps > 0:
                        calculated_set = ((common_reps - 1) // target_reps) + 1
                    else:
                        calculated_set = 1
                    # CHẶN LUỒNG NGẦM GHI ĐÈ: Chỉ cập nhật tổng số reps tịnh tiến khi chưa hoàn thành
                    if not exercise_finished:
                        python_actual_reps = common_reps  

                    # Tính Set tịnh tiến đồng bộ từ số Rep chung của cả 2 tay
                    #calculated_set = (common_reps // target_reps) + 1
                    current_set = min(calculated_set, target_sets)

                    if common_reps >= (target_sets * target_reps):
                        exercise_finished = True
                        guidance = "DA HOAN THANH BAI TAP!"
                        color = (0, 255, 0)
                    

                    # Kiểm tra nếu actual_reps đạt target thì kết thúc
                    #if actual_reps >= (target_sets * target_reps):
                        #exercise_finished = True
                        #guidance = " DA HOAN THANH BAI TAP!"
                        #color = (0, 255, 0)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- LUỒNG XỬ LÝ 3: BÀN TAY (HAND) ---
# --- LUỒNG XỬ LÝ 3: BÀN TAY (HAND) ---
def gen_hand(target_sets, target_reps):
    global hand_data
    global video_writer, current_video_filename
    global python_collected_angles, python_collected_times, start_exercise_time
    global python_actual_reps  # THÊM DÒNG NÀY
    
    error_frames_counter = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_set = 1
    hide_delay_counter = 0
    waiting_for_correct_posture = True
    exercise_finished = False
    position_confirmed = False
    
    hand_data["Left"]["counter"] = 0
    hand_data["Left"]["stage"] = None
    hand_data["Left"]["delay_counter"] = 0 
    hand_data["Right"]["counter"] = 0
    hand_data["Right"]["stage"] = None
    hand_data["Right"]["delay_counter"] = 0 

    with mp_pose.Pose(min_detection_confidence=0.7) as pose, \
         mp_hands.Hands(
            static_image_mode=False,        
            max_num_hands=2,                
            model_complexity=1,             
            min_detection_confidence=0.8,   
            min_tracking_confidence=0.8     
         ) as hands:
        
        while True:
            success, frame = cap.read()
            if not success: break

            # --- ĐOẠN DÁN CODE LƯU VIDEO ---
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                current_video_filename = f"video_bantay_{current_time}.mp4" 
                video_writer = cv2.VideoWriter(current_video_filename, fourcc, 20.0, (1280, 720))
            
            video_writer.write(frame)

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            guidance = "WAITING FOR BODY..."
            color = (255, 255, 255)
            box_color = (255, 255, 255) 
            
            pose_results = pose.process(rgb_frame)
            hand_results = hands.process(rgb_frame) 

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                raw_guidance, raw_color = get_position_guidance_hand(landmarks, hand_results, w, h)
                
                l_s = tuple(np.multiply([landmarks[11].x, landmarks[11].y], [w, h]).astype(int))
                r_s = tuple(np.multiply([landmarks[12].x, landmarks[12].y], [w, h]).astype(int))
                l_h = tuple(np.multiply([landmarks[23].x, landmarks[23].y], [w, h]).astype(int))
                r_h = tuple(np.multiply([landmarks[24].x, landmarks[24].y], [w, h]).astype(int))
                
                is_level_sh, _ = check_shoulder_tilt(landmarks)
                is_aligned_hip, _ = check_torso_rotation(landmarks)
                is_posture_ok = is_level_sh and is_aligned_hip and raw_guidance == "PERFECT! START EXERCISE"

                if exercise_finished:
                    guidance, color = "HOAN THANH TAT CA BAI TAP!", (0, 255, 0)
                elif not position_confirmed:
                    cv2.line(frame, l_s, r_s, box_color, 2)
                    cv2.line(frame, r_s, r_h, box_color, 2)
                    cv2.line(frame, r_h, l_h, box_color, 2)
                    cv2.line(frame, l_h, l_s, box_color, 2)
    
                    if is_posture_ok:
                        guidance, color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                        position_confirmed = True 
                    else:
                        guidance, color = raw_guidance, raw_color
                else:
                    is_in_rest_zone = True 
                    if hand_results.multi_hand_landmarks:
                        for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                            label = hand_results.multi_handedness[idx].classification[0].label
                            wrist = hand_lms.landmark[0]
                            mcp = hand_lms.landmark[9]
                            tip = hand_lms.landmark[12]
                            base_dist = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
                            current_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                            raw_ratio = current_dist / base_dist if base_dist != 0 else 0

                            if label == "Left":
                                ratio = kf_hand_ratio_l.update(raw_ratio)
                            else:
                                ratio = kf_hand_ratio_r.update(raw_ratio)

                            if ratio < 1.3 or ratio > 1.6:
                                is_in_rest_zone = False
                                break 
                    
                    # LOGIC KHÓA UI CHẶT CHẼ KHI ĐỨNG SAI VỊ TRÍ
                    if raw_guidance != "PERFECT! START EXERCISE" and raw_guidance != "VI TRI CHUAN! BAT DAU TAP":
                        guidance, color = raw_guidance, raw_color
                        position_confirmed = False
                    elif not is_in_rest_zone or (hand_data["Left"]["stage"] is not None or hand_data["Right"]["stage"] is not None):
                        guidance, color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)
                    else:
                        if not is_level_sh or not is_aligned_hip:
                            guidance, color = "GIU TU THE THANG VOI CAMERA", (0, 0, 255) 
                            position_confirmed = False
                        else:
                            guidance, color = "VI TRI CHUAN! BAT DAU TAP", (0, 255, 0)

                # Hiển thị thông báo hướng dẫn
                cv2.rectangle(frame, (w//2-250, 15), (w//2+250, 75), (0,0,0), -1)
                cv2.putText(frame, guidance, (w//2-230, 55), 1, 1.8, color, 3)

                # --- 2. CHỈ TÍNH TOÁN BÀN TAY KHI VỊ TRÍ ĐÃ CHUẨN ---
                if position_confirmed:
                    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                        for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                            label = hand_results.multi_handedness[idx].classification[0].label 
                            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                            
                            curr = hand_data[label]
                            wrist = hand_lms.landmark[0]
                            mcp = hand_lms.landmark[9]
                            tip = hand_lms.landmark[12]
                            base_dist = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
                            current_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                            raw_ratio = current_dist / base_dist if base_dist != 0 else 0
                            
                            if label == "Left":
                                ratio = kf_hand_ratio_l.update(raw_ratio)
                                score = int(np.clip(np.interp(ratio, [0.75, 1.9], [0, 100]), 0, 100))
                            else:
                                ratio = kf_hand_ratio_r.update(raw_ratio)
                                score = int(np.clip(np.interp(ratio, [0.75, 1.9], [0, 100]), 0, 100))

                            # --- ĐOẠN ĐƯA GIÁ TRỊ SCORE (%) VÀO ĐỒ THỊ ---
                            if start_exercise_time is None:
                                start_exercise_time = time.time()
                            elapsed = round(time.time() - start_exercise_time, 1)
                            if len(python_collected_times) == 0 or elapsed - float(python_collected_times[-1].replace('s','')) >= 1.0:
                                python_collected_times.append(f"{int(elapsed)}s")
                                python_collected_angles.append(int(score))

                            is_currently_open = ratio > 1.6
                            is_straight = check_hand_alignment(hand_lms, is_open=is_currently_open)
                            
                            if not is_straight:
                                guidance = "HUONG LONG BAN TAY THANG VOI CAMERA!"
                                color = (0, 165, 255)
                            else:
                                old_rep_count = curr["counter"]
                                
                                if ratio < 1.3: 
                                    curr["stage"] = "Closed"  
                                if ratio > 1.7 and curr["stage"] == "Closed":
                                    curr["stage"] = "Open"    
                                    curr["counter"] += 1     

                                if curr["counter"] > old_rep_count:
                                    # KHÔNG tăng python_actual_reps ở đây nữa
                                    waiting_for_correct_posture = True
                                    position_confirmed = False
                                    curr["delay_counter"] = 1
                                
                                if not exercise_finished and not waiting_for_correct_posture:
                                    if curr["stage"] == "Open":
                                        if ratio > 1.65:
                                            curr["delay_counter"] += 1
                                        if hide_delay_counter > 35:
                                            waiting_for_correct_posture = True
                                            curr["delay_counter"] = 0
                                            position_confirmed = False

                                        if curr["counter"] >= target_reps:
                                            if current_set >= target_sets:
                                                exercise_finished = True
                                            else:
                                                current_set += 1
                                                hand_data["Left"]["counter"] = 0
                                                hand_data["Right"]["counter"] = 0
                                                hand_data["Left"]["stage"] = None
                                                hand_data["Right"]["stage"] = None

                            # Vẽ bảng thông số UI
                            ox = 30 if label == "Left" else w - 280
                            ui_color = (245, 117, 16) if label == "Left" else (117, 66, 245)
                            cv2.rectangle(frame, (ox, 20), (ox + 250, 160), ui_color, -1)
                            cv2.rectangle(frame, (ox, 20), (ox + 250, 160), (255, 255, 255), 2)
                            cv2.putText(frame, f"{label.upper()} HAND", (ox + 15, 55), 1, 1.4, (255, 255, 255), 2)
                            cv2.putText(frame, f"SET:{current_set} REP:{curr['counter']}", (ox + 15, 105), 1, 1.8, (0, 255, 255), 2)
                            cv2.putText(frame, f"QUALITY: {score}%", (ox + 15, 145), 1, 1.2, (255, 255, 255), 2)

                            # Vẽ thanh Progress Bar dọc cánh màn hình
                            bar_x = ox + 80 if label == "Left" else ox + 130
                            bar_start_y, bar_end_y = 190, 480
                            dynamic_color = (0, int(score * 2.55), int(255 - score * 2.55))
                            bar_height = int(np.interp(score, [0, 100], [bar_end_y, bar_start_y]))
                            
                            cv2.rectangle(frame, (bar_x, bar_start_y), (bar_x + 40, bar_end_y), (255, 255, 255), 3)
                            cv2.rectangle(frame, (bar_x, bar_height), (bar_x + 40, bar_end_y), dynamic_color, -1)
                            cv2.putText(frame, f"{score}%", (bar_x + 45, bar_height + 5), 1, 1.1, (255, 255, 255), 2)
                    
                    # =============================================
                    # TÍNH COMMON REPS CHO BÀI HAND (sau vòng lặp for)
                    # =============================================
                    left_reps = hand_data["Left"]["counter"]
                    right_reps = hand_data["Right"]["counter"]
                    actual_reps = min(left_reps, right_reps)
                    
                    # Cập nhật python_actual_reps là số reps chung
                    python_actual_reps = actual_reps
                    
                    # Vẽ thêm thông báo common reps lên màn hình
                    cv2.rectangle(frame, (w//2 - 180, 130), (w//2 + 180, 180), (0, 0, 0), -1)
                    cv2.putText(frame, f"ACTUAL REPS: {actual_reps}", (w//2 - 150, 165), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
                    # Kiểm tra nếu actual_reps đạt target thì kết thúc
                    if actual_reps >= (target_sets * target_reps):
                        exercise_finished = True
                        guidance = "HOAN THANH TAT CA BAI TAP!"
                        color = (0, 255, 0)

            global current_guidance_text
            current_guidance_text = guidance

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    cap.release()

def reset_exercise_data():
    global python_collected_angles, python_collected_times, start_exercise_time, python_actual_reps
    python_collected_angles = []
    python_collected_times = []
    python_actual_reps = 0
    start_exercise_time = None

def load_permanent_history(ex_type, email):
    """Đọc dữ liệu lịch sử của bệnh nhân từ file JSON"""
    if not email:
        print("WARNING: load_permanent_history called with empty email")
        return {}
    
    patient_dir = get_patient_dir(email)
    
    if ex_type == 'shoulder':
        file_path = os.path.join(patient_dir, 'smartarm_history_shoulder.json')
    elif ex_type == 'elbow':
        file_path = os.path.join(patient_dir, 'smartarm_history_elbow.json')
    else:
        file_path = os.path.join(patient_dir, 'smartarm_history_hand.json')
    
    print(f"=== DEBUG: Loading history from {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records")
            return data if data else {}
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return {}

def save_permanent_history(ex_type, email, data):
    """Ghi dữ liệu lịch sử của bệnh nhân xuống file JSON"""
    if not email:
        print("WARNING: save_permanent_history called with empty email")
        return
    
    patient_dir = get_patient_dir(email)
    
    if ex_type == 'shoulder':
        file_path = os.path.join(patient_dir, 'smartarm_history_shoulder.json')
    elif ex_type == 'elbow':
        file_path = os.path.join(patient_dir, 'smartarm_history_elbow.json')
    else:
        file_path = os.path.join(patient_dir, 'smartarm_history_hand.json')
    
    print(f"=== DEBUG: Saving history to {file_path} ===")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data if data is not None else {}, f, ensure_ascii=False, indent=4)
        print(f"Saved successfully!")
    except Exception as e:
        print(f"Lỗi ghi file: {e}")

def save_permanent_history(ex_type, email, data):
    """Ghi dữ liệu lịch sử của bệnh nhân xuống file JSON"""
    patient_dir = get_patient_dir(email)
    
    if ex_type == 'shoulder':
        file_path = os.path.join(patient_dir, 'smartarm_history_shoulder.json')
    elif ex_type == 'elbow':
        file_path = os.path.join(patient_dir, 'smartarm_history_elbow.json')
    else:
        file_path = os.path.join(patient_dir, 'smartarm_history_hand.json')
        
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data if data is not None else {}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Lỗi ghi file: {e}")
# --- TRONG HÀM DEF XỬ LÝ MEDIAPIPE CỦA BẠN ---
def your_mediapipe_frame_process():
    global python_collected_angles, python_collected_times, start_exercise_time
    # ... code đọc camera và tính góc của bạn ...
    # Giả sử biến tính góc bằng Python của bạn tên là: angle_deg
    angle_deg = 95 

    # Gom dữ liệu góc theo thời gian (ví dụ cứ mỗi frame hoặc tính theo giây)
    if start_exercise_time is not None:
        elapsed = round(time.time() - start_exercise_time, 1)
        
        # Để tránh mảng quá dày, cứ sau khoảng 1 giây ta lưu 1 điểm dữ liệu
        if len(python_collected_times) == 0 or elapsed - python_collected_times[-1] >= 1.0:
            python_collected_times.append(f"{int(elapsed)}s")
            python_collected_angles.append(int(angle_deg))



@app.route('/login-doctor', methods=['POST'])
def login_doctor():
    data = request.json

    email = data.get('email')
    password = data.get('password')

    users = load_users()

    if email in users and users[email]['password'] == password and users[email].get('role') == 'Doctor':
        # có thể thêm check role nếu sau này phân quyền
        return jsonify({
            "status": "success",
            "redirect": "/doctor-page"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Sai tài khoản bác sĩ!"
        })
    
@app.route('/doctor-page')
def doctor_page():
    return send_from_directory('.', 'index_ee.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    users = load_users()
    user = users.get(data['email'])
    if user and user['password'] == data['password'] and user.get('role') == 'Patient':
        return jsonify({"status": "success", "message": "Đăng nhập thành công!"})
    return jsonify({"status": "error", "message": "Sai mật khẩu!"})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'Patient')

        # BƯỚC 1: LUÔN ĐỌC FILE MỚI NHẤT TRƯỚC KHI KIỂM TRA
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                try:
                    users = json.load(f)
                except json.JSONDecodeError:
                    users = {} # Nếu file trống hoặc lỗi định dạng thì tạo mới dict
        else:
            users = {}

        # BƯỚC 2: KIỂM TRA EMAIL
        if email in users:
            return jsonify({"status": "error", "message": "Email đã tồn tại!"})

        # BƯỚC 3: THÊM DỮ LIỆU MỚI VÀO BIẾN TẠM
        users[email] = {
            "username": username,
            "password": password,
            "role": role
        }

        # BƯỚC 4: GHI ĐÈ LẠI FILE NGAY LẬP TỨC (TỰ ĐỘNG CẬP NHẬT)
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        # Thêm indent=4 vào cuối lệnh json.dump
            json.dump(users, f, ensure_ascii=False, indent=4)
        
        # BƯỚC 5: TẠO THƯ MỤC CHO BỆNH NHÂN MỚI (NẾU LÀ PATIENT)
        if role == 'Patient':
            try:
                patient_dir = get_patient_dir(email)
                print(f"Đã tạo thư mục cho bệnh nhân mới: {patient_dir}")
            except Exception as e:
                print(f"Lỗi tạo thư mục cho {email}: {e}")

        print(f"--- Đã cập nhật thành công user mới: {email} ---")
        return jsonify({"status": "success", "message": "Đăng ký thành công!"})

    except Exception as e:
        print(f"Lỗi hệ thống: {str(e)}")
        return jsonify({"status": "error", "message": "Có lỗi xảy ra trên server!"})

@app.route('/video_feed/<type>') # Đổi 'exercise_type' thành 'type' để khớp với các đường dẫn cũ của bạn
def video_feed(type):
    # Lấy thông số sets và reps từ URL (mặc định là 3 set 10 rep nếu không có)
    sets = int(request.args.get('sets', 3))
    reps = int(request.args.get('reps', 10))
    target_angle = request.args.get('target_angle')

    if target_angle and target_angle != 'null':
        target_angle = int(target_angle)
    
    if type == 'shoulder': 
        # Truyền sets, reps vào hàm gen_shoulder
        return Response(gen_shoulder(sets, reps), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif type == 'elbow': 
        return Response(gen_elbow(sets, reps), mimetype='multipart/x-mixed-replace; boundary=frame')
    else: 
        return Response(gen_hand(sets, reps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register-page')
def register_page(): return send_from_directory('.', 'index2.html')

@app.route('/selection')
def selection(): return send_from_directory('.', 'index4.html')

@app.route('/get-current-reps')
def get_current_reps():
    global python_actual_reps
    return jsonify({"actual_reps": python_actual_reps})
@app.route('/index5')
def view_chart_page():
    global python_collected_angles, python_collected_times
    global python_actual_reps
    
    ex_type = request.args.get('ex_type', 'shoulder')
    patient_email = request.args.get('email', '')  
    
    if not patient_email:
        return "Không tìm thấy email bệnh nhân!", 400
    
    target_sets = request.args.get('target_sets', '3')
    target_reps_per_set = request.args.get('target_reps_per_set', '10')
    target_angle = request.args.get('target_angle', None)  
    
    # 🌟 SỬA LỖI 2: Đưa xử lý target_angle lên ĐẦU HÀM để tránh NameError
    if target_angle and target_angle != '' and target_angle != 'None' and target_angle != 'null':
        try:
            target_angle_int = int(target_angle)
        except:
            target_angle_int = None
    else:
        target_angle_int = None

    try:
        TARGET_SETS = int(target_sets)
        TARGET_REPS_PER_SET = int(target_reps_per_set)
        TARGET_TOTAL_REPS = TARGET_SETS * TARGET_REPS_PER_SET
    except:
        TARGET_SETS = 3
        TARGET_REPS_PER_SET = 10
        TARGET_TOTAL_REPS = 30
    
    history_data = load_permanent_history(ex_type, patient_email)
    if history_data is None:
        history_data = {}
        
    today_key = datetime.datetime.now().strftime("%d/%m")
    
    if python_actual_reps > 0 or len(python_collected_angles) > 0:
        actual_reps = python_actual_reps
        print(f"=== DEBUG: Saving actual_reps = {actual_reps} ===")

        if TARGET_TOTAL_REPS > 0:
            calculated_compliance = int((actual_reps / TARGET_TOTAL_REPS) * 100)
        else:
            calculated_compliance = 0
            
        calculated_compliance = min(calculated_compliance, 100)
        
        history_data[today_key] = {
            "labels": python_collected_times,
            "angles": python_collected_angles,
            "actual_reps": actual_reps,
            "target_sets": TARGET_SETS,
            "target_reps_per_set": TARGET_REPS_PER_SET,
            "target_total_reps": TARGET_TOTAL_REPS,
            "compliance": calculated_compliance,
            "target_angle": target_angle_int # <-- Bây giờ biến này đã an toàn
        }
        
        save_permanent_history(ex_type, patient_email, history_data)
        
        current_labels = python_collected_times
        current_angles = python_collected_angles
        
        # Reset dữ liệu tạm
        python_collected_times = []
        python_collected_angles = []
        python_actual_reps = 0
        
    else:
        if history_data:
            def parse_key(k):
                if len(k) == 5 and '/' in k:
                    d, m = k.split('/')
                    return (int(m), int(d))
                return (0, 0)
            
            latest_key = sorted(history_data.keys(), key=parse_key)[-1] if history_data else None
            if latest_key:    
                current_labels = history_data[latest_key].get("labels", [])
                current_angles = history_data[latest_key].get("angles", [])
            else:
                current_labels, current_angles = [], []
        else:
            current_labels, current_angles = [], []

    return render_template(
        'index5.html',
        labels=json.dumps(current_labels),
        angles=json.dumps(current_angles),
        ex_type=ex_type,
        target_sets=TARGET_SETS,
        target_reps_per_set=TARGET_REPS_PER_SET,
        target_total_reps=TARGET_TOTAL_REPS,
        history_json=json.dumps(history_data),
        target_angle=target_angle_int,
        patient_email=patient_email  
    )

@app.route('/get-session-chart-data', methods=['GET'])
def get_session_chart_data():
    global python_collected_angles, python_collected_times, python_actual_reps
    global python_target_reps, python_calculated_compliance # Gọi biến global chuẩn
    
    return jsonify({
        "status": "success",
        "labels": python_collected_times,
        "angles": python_collected_angles,
        "actual_reps": python_actual_reps,
        "target_reps": python_target_reps,
        "compliance": python_calculated_compliance
    })                           
@app.route('/')
def index(): return send_from_directory('.', 'index.html')

@app.route('/info-page')
def info_page(): return send_from_directory('.', 'index3.html')

def load_users():
    if not os.path.exists(USER_DATA_FILE): return {}
    try:
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_users(users):
    try:
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            # Dùng indent=2 để dễ đọc file
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except IOError:
        return False

def get_patient_dir(email):
    """Lấy đường dẫn thư mục của bệnh nhân, tạo nếu chưa có"""
    if not email:
        return DATA_DIR
    safe_email = email.replace('@', '_at_').replace('.', '_dot_')
    patient_dir = os.path.join(DATA_DIR, safe_email)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
        print(f"Đã tạo thư mục: {patient_dir}")
    return patient_dir

def migrate_old_data():
    """Di chuyển dữ liệu cũ sang cấu trúc mới - CHỈ CHẠY 1 LẦN"""
    
    # Tạo thư mục data nếu chưa có
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Đọc danh sách bệnh nhân từ users.json
    users = load_users()  # Bây giờ hàm này đã được định nghĩa
    
    # Danh sách file JSON cũ
    old_files = [
        'smartarm_history_shoulder.json',
        'smartarm_history_elbow.json', 
        'smartarm_history_hand.json'
    ]
    
    # Với mỗi bệnh nhân, tạo thư mục riêng
    for email, user_info in users.items():
        if user_info.get('role') == 'Patient':
            patient_dir = get_patient_dir(email)
            print(f"Đã tạo thư mục cho {email}: {patient_dir}")
            
            # Di chuyển dữ liệu cũ nếu có
            for old_file in old_files:
                if os.path.exists(old_file):
                    try:
                        with open(old_file, 'r', encoding='utf-8') as f:
                            old_data = json.load(f)
                        
                        if old_data:
                            new_file_path = os.path.join(patient_dir, old_file)
                            with open(new_file_path, 'w', encoding='utf-8') as f:
                                json.dump(old_data, f, ensure_ascii=False, indent=4)
                            print(f"  - Đã copy dữ liệu từ {old_file}")
                    except Exception as e:
                        print(f"Lỗi copy {old_file}: {e}")
    
    print("Hoàn tất migration!")

# Chạy migration khi khởi động (chỉ 1 lần)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
# Kiểm tra nếu chưa migration thì chạy
if not os.path.exists(os.path.join(DATA_DIR, 'migrated.txt')):
    migrate_old_data()
    # Tạo file đánh dấu đã migration
    with open(os.path.join(DATA_DIR, 'migrated.txt'), 'w') as f:
        f.write('migrated')
    print("Đã hoàn thành migration dữ liệu cũ!")
@app.route('/start-exercise/<type>')
def start_exercise(type):
    reset_exercise_data()
    translations = {
        'shoulder': 'BÀI TẬP VAI', 
        'elbow': 'BÀI TẬP KHUỶU TAY', 
        'hand': 'BÀI TẬP BÀN TAY'
    }
    ten_bai_tap = translations.get(type, type.upper())
    ex_id = request.args.get('ex_id', '0')
    
    # Lấy đủ sets và reps
    sets = request.args.get('sets', '3')
    reps = request.args.get('reps', '10')
    target_angle = request.args.get('target_angle')
    
    # Lưu target_total_reps để dùng sau này
    target_total_reps = int(sets) * int(reps)
    
    return f'''
<html>
<head>
    <title>Luyện tập {ten_bai_tap}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            margin: 0; background: #000; color: white; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            display: flex; flex-direction: column; 
            align-items: center; justify-content: center; 
            height: 100vh; overflow: hidden; padding: 20px;
        }}
        .header-bar {{
            height: 55px; width: 100%; background: #111;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 15px; border-bottom: 1px solid #333; flex-shrink: 0; z-index: 100;
        }}
        h2 {{ margin: 0; font-size: 1.3rem; color: #ffffff !important; text-align: center; flex: 1; }}
        .video-wrapper {{
            flex-grow: 1; width: 100%; display: flex; justify-content: center;
            align-items: center; background: #000; overflow: hidden;
            border: none !important; box-shadow: none !important;
        }}
        img {{ width: 100%; height: 100%; object-fit: contain; background: #111; }}
        .btn {{ 
            top: 20px; padding: 10px 25px; color: white; border-radius: 30px; 
            cursor: pointer; font-weight: bold; text-decoration: none; 
            font-size: 1rem; transition: 0.3s; z-index: 100;
        }}
        .btn-back {{ background: #444; border: 2px solid #888; }}
        .btn-back:hover {{ background: #666; }}
        .btn-finish {{ background: #cc0000; border: 2px solid #ff4c4c; }}
        .btn-finish:hover {{ background: #ff0000; transform: scale(1.05); }}
        .nav-side {{ width: 180px; display: flex; }}
    </style>
</head>
<body>
    <div class="header-bar">
        <div class="nav-side">
            <button class="btn btn-back" onclick="cancelAndBack()">⬅ QUAY LẠI</button>
        </div>
        <h2>{ten_bai_tap} (Mục tiêu: {sets} Set x {reps} Rep)</h2>
        <div class="nav-side" style="justify-content: flex-end;">
            <button class="btn btn-finish" onclick="saveVideo()">HOÀN THÀNH & LƯU</button>
        </div>
    </div>
    
    <div class="video-wrapper">
        <img src="/video_feed/{type}?sets={sets}&reps={reps}">
    </div>

    <script>
        let currentAudio = null;
        let lastGuidance = "";

        const audioMap = {{
            "HAY TIEN LEN": "/static/audio/tien_len.mp3",
            "HAY LUI LAI, QUA GAN ROI": "/static/audio/lui_lai.mp3",
            "LUI LAI, CAN KHOANG TRONG DE DUOI TAY": "/static/audio/lui_lai_can_khoang_trong.mp3",
            "HAY DI CHUYEN SANG TRAI": "/static/audio/sang_trai.mp3",
            "HAY DI CHUYEN SANG PHAI": "/static/audio/sang_phai.mp3",
            "GIU TU THE THANG VOI CAMERA": "/static/audio/thang_camera.mp3",
            "VI TRI CHUAN! BAT DAU TAP": "/static/audio/thuc_hien-dong_tac.mp3",
            "HUONG LONG BAN TAY THANG VAO CAMERA": "/static/audio/ban_tay_thang_cam.mp3",
            "HAY DI CHUYEN TAY SANG TRAI": "/static/audio/sang_trai.mp3",
            "HAY DI CHUYEN TAY SANG PHAI": "/static/audio/sang_phai.mp3",
            "HAY DUA BAN TAY LAI GAN CAMERA MOT CHUT": "/static/audio/ban_tay_lai_gan.mp3",
            "HAY DUA BAN TAY RA XA CAMERA MOT CHUT": "/static/audio/ban_tay_ra_xa.mp3"
        }};

        function playGuidanceAudio(text) {{
            if (text === lastGuidance) return;
            lastGuidance = text;
            let audioUrl = audioMap[text];
            if (audioUrl) {{
                if (currentAudio) {{
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                }}
                currentAudio = new Audio(audioUrl);
                currentAudio.play().catch(e => console.log("Trình duyệt chặn:", e));
            }}
        }}

        setInterval(() => {{
            fetch('/get_current_guidance')
                .then(response => response.json())
                .then(data => {{
                    if (data.guidance) {{
                        playGuidanceAudio(data.guidance);
                    }}
                }})
                .catch((error) => console.error("Lỗi đồng bộ audio:", error));
        }}, 400);

         setInterval(() => {{
            fetch('/get-current-reps')
                .then(response => response.json())
                .then(data => {{
                    const actualReps = data.actual_reps;
                    const percent = Math.round((actualReps / {target_total_reps}) * 100);
                    const counterDiv = document.getElementById('reps-counter');
                    if (counterDiv) {{
                        counterDiv.innerHTML = `📊 Tiến độ: ${{actualReps}}/${{targetTotalReps}} reps (${{percent}}%)`;
                        if (actualReps >= targetTotalReps) {{
                            counterDiv.style.color = '#00ff00';
                            counterDiv.style.borderColor = '#00ff00';
                        }} else {{
                            counterDiv.style.color = '#4ce9ff';
                            counterDiv.style.borderColor = '#4ce9ff';
                        }}
                    }}
                }})
                .catch((error) => console.error("Lỗi lấy số reps:", error));
        }}, 1000);

        async function saveVideo() {{
            const exId = "{ex_id}";
            const exType = "{type}";
            const email = localStorage.getItem('patient_email');
            const targetSets = "{sets}";
            const targetRepsPerSet = "{reps}";
            
            console.log("Saving video - Email:", email);

            if (!email) {{
                alert("Không tìm thấy email! Vui lòng đăng nhập lại.");
                window.location.href = "/";
                return;
    }}
            let actualReps = 0;
            try {{
                const response = await fetch('/get-current-reps');
                const data = await response.json();
                actualReps = data.actual_reps;
                console.log("Actual reps từ API:", actualReps);
            }} catch(e) {{
                console.error("Lỗi lấy số reps:", e);
            }}
            
            console.log("Số reps thực tế: " + actualReps);

            if (email && exId && exId !== "0") {{
                try {{
                    await fetch('/complete-exercise', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            email: email,
                            ex_id: exId,
                            actual_reps: actualReps
                        }})
                    }});
                    localStorage.setItem('finished_ex_' + exId, 'true');
                }} catch (e) {{
                    console.error("Lỗi cập nhật trạng thái:", e);
                }}
            }}
 
            try {{ 
                await fetch('/stop_record', {{ method: 'POST' }});
            }} catch (e) {{ 
                console.error("stop_record loi:", e); 
            }}
 
            window.location.href = "/index5?ex_id=" + exId +
                                   "&ex_type=" + exType +
                                   "&target_sets=" + targetSets +
                                   "&target_reps_per_set=" + targetRepsPerSet +
                                   "&target_angle={target_angle}" +
                                   "&email=" + encodeURIComponent(email);
        }}
 
        function cancelAndBack() {{
            fetch('/cancel_record', {{ method: 'POST' }})
                .finally(() => {{ window.location.href = "/selection"; }});
        }}
    </script>
</body>
</html>
    '''

@app.route('/stop_record', methods=['POST'])
def stop_record():
    global video_writer
    if video_writer:
        video_writer.release() # Đóng file video
        video_writer = None    # Reset biến để lần tập sau tạo file mới
        return jsonify({"status": "success", "message": "Video bài tập đã được lưu thành công!"})
    return jsonify({"status": "error", "message": "Không tìm thấy file video đang quay."})


@app.route('/cancel_record', methods=['POST'])
def cancel_record():
    global video_writer
    if video_writer:
        # 1. Lấy tên file đang ghi trước khi đóng
        # Lưu ý: Nếu bạn dùng class VideoRecorder, bạn cần lưu filename vào self
        filename = "video_dang_quay.mp4" # Thay bằng tên biến file thực tế của bạn
        
        # 2. Dừng ghi và giải phóng file
        video_writer.release() 
        video_writer = None
        
        # 3.Xóa đúng file dựa trên tên đã lưu
        if current_video_filename and os.path.exists(current_video_filename):
            os.remove(current_video_filename)
            current_video_filename = ""
            
        return jsonify({"status": "success", "message": "Đã hủy và xóa video."})
    return jsonify({"status": "error", "message": "Không có video nào đang quay."})


@app.route('/assign-exercise', methods=['POST'])
def assign_exercise():
    data = request.json
    # Trong project của bạn, email là khóa chính trong users.json
    patient_email = data.get('email') 
    exercise_info = data.get('exercise')

    users = load_users() # Hàm này bạn đã có để đọc file

    if patient_email in users:
        # Nếu user chưa từng có bài tập nào, tạo một danh sách trống
        if 'exercises' not in users[patient_email]:
            users[patient_email]['exercises'] = []
        
        # Tự động thêm vào danh sách
        users[patient_email]['exercises'].append(exercise_info)
        
        # TỰ ĐỘNG GHI ĐÈ LẠI FILE users.json
        if save_users(users):
            return jsonify({"status": "success", "message": "Đã gán bài tập thành công!"})
        else:
            return jsonify({"status": "error", "message": "Lỗi khi lưu dữ liệu!"}), 500
    
    return jsonify({"status": "error", "message": "Không tìm thấy bệnh nhân!"}), 404

@app.route('/get-exercises', methods=['POST'])
def get_exercises():
    """Lấy danh sách bài tập của bệnh nhân dựa trên email"""
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({"status": "error", "message": "Thiếu email!"}), 400
        
    users = load_users()
    user = users.get(email)
    
    if not user:
        return jsonify({"status": "error", "message": "Không tìm thấy người dùng!"}), 404
        
    # Lấy danh sách bài tập, nếu không có thì trả về mảng rỗng
    exercises = user.get('exercises', [])
    completed_list = user.get('completed_exercises', [])

    # Gắn thêm cờ trạng thái hoàn thành cho từng bài tập trước khi trả về frontend
    for idx, ex in enumerate(exercises):
        ex_id_str = str(idx + 1)
        ex['is_completed'] = ex_id_str in completed_list
        
    return jsonify({"status": "success", "exercises": exercises})

# Thêm API mới trong app.py
@app.route('/complete-exercise', methods=['POST'])
def complete_exercise():
    data = request.json
    email = data.get('email')
    ex_id = data.get('ex_id')
    actual_reps = data.get('actual_reps', 0)
    
    print(f"=== complete-exercise: email={email}, ex_id={ex_id}, actual_reps={actual_reps} ===")
    
    if not email:
        return jsonify({"status": "error", "message": "Thiếu email!"}), 400
    
    # Cập nhật biến global python_actual_reps
    global python_actual_reps
    python_actual_reps = actual_reps
    
    # Cập nhật completed_exercises trong users.json
    users = load_users()
    if email in users:
        if 'completed_exercises' not in users[email]:
            users[email]['completed_exercises'] = []
        if ex_id not in users[email]['completed_exercises']:
            users[email]['completed_exercises'].append(ex_id)
        save_users(users)
        print(f"Đã cập nhật completed_exercises cho {email}")
    
    return jsonify({"status": "success"})

@app.route('/delete-exercise', methods=['POST'])
def delete_exercise():
    data = request.json
    patient_email = data.get('email')
    exercise_description = data.get('exerciseDescription')
    
    users = load_users()
    
    if patient_email in users and 'exercises' in users[patient_email]:
        # Tìm và xóa bài tập có description khớp
        original_length = len(users[patient_email]['exercises'])
        users[patient_email]['exercises'] = [
            ex for ex in users[patient_email]['exercises'] 
            if ex.get('description') != exercise_description
        ]
        
        if len(users[patient_email]['exercises']) < original_length:
            if save_users(users):
                return jsonify({"status": "success", "message": "Đã xóa bài tập!"})
    
    return jsonify({"status": "error", "message": "Không tìm thấy bài tập!"}), 404

@app.route('/save-exercise', methods=['POST'])
def save_exercise():
    data = request.json

    email = data.get("email")
    exercise = data.get("exercise")

    if not email or not exercise:
        return jsonify({
            "status": "error",
            "message": "Thiếu dữ liệu"
        })

    if not os.path.exists(USER_DATA_FILE):
        return jsonify({
            "status": "error",
            "message": "Không tìm thấy users.json"
        })

    with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)

    if email not in users:
        return jsonify({
            "status": "error",
            "message": "Không tìm thấy bệnh nhân"
        })

    if "exercises" not in users[email]:
        users[email]["exercises"] = []

    users[email]["exercises"].append(exercise)

    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

    return jsonify({
        "status": "success"
    })

@app.route('/get_current_guidance')
def get_current_guidance():
    global current_guidance_text
    return jsonify({"guidance": current_guidance_text})

if __name__ == '__main__':
       app.run(debug=True, port=5000, threaded=True)
