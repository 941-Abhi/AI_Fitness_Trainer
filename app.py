import streamlit as st
import cv2
import numpy as np
import time
from scipy.signal import find_peaks
from ultralytics import YOLO
import mediapipe as mp

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="💪 Fitness Tracker Dashboard", layout="wide")
st.title("💪 AI Fitness Tracker")

# -----------------------------
# Session State
# -----------------------------
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'exercise' not in st.session_state:
    st.session_state.exercise = "Biceps"
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0
if 'left_counter' not in st.session_state:
    st.session_state.left_counter = 0
if 'right_counter' not in st.session_state:
    st.session_state.right_counter = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# -----------------------------
# Sidebar Controls
# -----------------------------
st.session_state.exercise = st.sidebar.selectbox("Select Exercise", ["Biceps", "Pushups"])
if st.sidebar.button("Start Camera"):
    st.session_state.camera_on = True
    st.session_state.rep_count = 0
    st.session_state.left_counter = 0
    st.session_state.right_counter = 0
    st.session_state.frame_count = 0
if st.sidebar.button("Stop Camera"):
    st.session_state.camera_on = False

frame_placeholder = st.empty()

# -----------------------------
# Utility Functions
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def linear_regression(keypoint_name_list, keypoint_dict):
    x_data, y_data = [], []
    for keypoint in keypoint_name_list:
        if keypoint in keypoint_dict:
            x_data.append(keypoint_dict[keypoint]["x"])
            y_data.append(keypoint_dict[keypoint]["y"])
    if len(x_data) < 2:
        return np.nan, np.nan, np.nan
    N = len(x_data)
    sum_x, sum_y = sum(x_data), sum(y_data)
    sum_xy = sum(x*y for x, y in zip(x_data, y_data))
    sum_xx = sum(x*x for x in x_data)
    sum_yy = sum(y*y for y in y_data)
    denom = (N*sum_xx - sum_x**2)
    if denom == 0:
        return np.nan, np.nan, np.nan
    slope = (N*sum_xy - sum_x*sum_y)/denom
    intercept = (sum_y - slope*sum_x)/N
    numerator = (N*sum_xy - sum_x*sum_y)**2
    denominator = (N*sum_xx - sum_x*2)(N*sum_yy - sum_y**2)
    r_squared = numerator/denominator if denominator!=0 else np.nan
    return slope, intercept, r_squared

def ema_filter(dict_name):
    alpha = 0.3
    ema_value = None
    ema_values = {}
    for frame_number, value in sorted(dict_name.items()):
        if np.isnan(value):
            ema_values[frame_number] = np.nan
            continue
        if ema_value is None:
            ema_value = value
        else:
            ema_value = alpha*value + (1-alpha)*ema_value
        ema_values[frame_number] = ema_value
    return ema_values

# -----------------------------
# Model Setup
# -----------------------------
pose_model = YOLO("yolov8s-pose.pt")
keypoint_names = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# -----------------------------
# Camera Loop
# -----------------------------
if st.session_state.camera_on:
    cap = None
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                break
            else:
                cap.release()
                cap = None
        else:
            cap = None

    if cap is None or not cap.isOpened():
        st.error("⚠ Could not open any camera.")
    else:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        slope_dict = {}
        slope_emas_dict = {}
        old_num_peaks = 0
        last_time = time.time()
        fps_limit = 15
        frame_time = 1.0/fps_limit

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as curl_pose:
            while st.session_state.camera_on:
                current_time = time.time()
                if current_time - last_time < frame_time:
                    time.sleep(0.005)
                    continue
                last_time = current_time

                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                st.session_state.frame_count += 1

                # ----------------------------------------
                # BICEPS TRACKER
                # ----------------------------------------
                if st.session_state.exercise == "Biceps":
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = curl_pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    try:
                        landmarks = results.pose_landmarks.landmark
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        left_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                        right_angle = calculate_angle(right_shoulder,right_elbow,right_wrist)

                        if left_angle > 160: st.session_state.left_stage = "down"
                        if left_angle < 30 and st.session_state.left_stage == 'down':
                            st.session_state.left_stage = "up"
                            st.session_state.left_counter += 1

                        if right_angle > 160: st.session_state.right_stage = "down"
                        if right_angle < 30 and st.session_state.right_stage == 'down':
                            st.session_state.right_stage = "up"
                            st.session_state.right_counter += 1

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        cv2.putText(image,f"L: {st.session_state.left_counter}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                        cv2.putText(image,f"R: {st.session_state.right_counter}",(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                        cv2.putText(image,f"L_angle: {int(left_angle)}",(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                        cv2.putText(image,f"R_angle: {int(right_angle)}",(300,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

                    except:
                        pass
                    frame_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

                # ----------------------------------------
                # PUSHUP TRACKER
                # ----------------------------------------
                elif st.session_state.exercise == "Pushups":
                    pose_results = pose_model(frame, verbose=False, conf=0.5)
                    keypoint_dict = {}
                    person_detected = False
                    current_feedback = ""
                    feedback_color = (255, 255, 255)
                    m = b = r_sq = np.nan

                    if pose_results and len(pose_results) > 0:
                        person = pose_results[0]
                        if person.keypoints is not None and len(person.keypoints.data) > 0:
                            keypoints = person.keypoints.data[0]
                            person_detected = True
                            for kp, name in zip(keypoints, keypoint_names):
                                x, y, prob = kp
                                keypoint_dict[name] = {"x": x.item(), "y": y.item(), "probability": prob.item()}

                            prob_thresh = 0.3
                            required_kp = ["right_hip", "right_shoulder", "right_ankle"]
                            missing = [kp for kp in required_kp if kp not in keypoint_dict or keypoint_dict[kp]["probability"] < prob_thresh]

                            if missing:
                                current_feedback = f"Adjust position - missing: {', '.join(missing)}"
                                feedback_color = (0, 165, 255)
                                slope_dict[st.session_state.frame_count] = np.nan
                            else:
                                m, b, r_sq = linear_regression(required_kp, keypoint_dict)
                                if not np.isnan(m):
                                    slope_dict[st.session_state.frame_count] = abs(m)
                                    ema_values = ema_filter(slope_dict)
                                    if len(ema_values) > 10:
                                        arr = np.array([v for v in ema_values.values() if not np.isnan(v)])
                                        peaks, _ = find_peaks(arr, prominence=0.1, distance=10)
                                        num_peaks = len(peaks)
                                        if num_peaks > old_num_peaks:
                                            st.session_state.rep_count += 1
                                            current_feedback = f"Rep #{st.session_state.rep_count} completed!"
                                            feedback_color = (0,255,0)
                                        old_num_peaks = num_peaks

                    annotated_frame = pose_results[0].plot(img=frame) if person_detected else frame.copy()
                    overlay = annotated_frame.copy()
                    h, w = annotated_frame.shape[:2]
                    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, f"Reps: {st.session_state.rep_count}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                    if current_feedback:
                        cv2.putText(annotated_frame, current_feedback, (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
                    frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        st.session_state.camera_on = False
        st.success("Camera stopped ✅")