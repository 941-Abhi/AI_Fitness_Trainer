import cv2
import mediapipe as mp
import numpy as np

# Mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Curl counter variables
left_counter = 0 
left_stage = None
right_counter = 0 
right_stage = None

# Mediapipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        height, width, _ = frame.shape

        # Convert to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Left arm coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Right arm coordinates
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate elbow angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Update curl counters
            if left_angle > 160:
                left_stage = "down"
            if left_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1

            if right_angle > 160:
                right_stage = "down"
            if right_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1

        except:
            pass

        # Draw status boxes
        # Left arm box
        cv2.rectangle(image, (0,0), (400,230), (245,117,16), -1)
        cv2.putText(image, 'LEFT REP COUNT', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(left_counter), (20,90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 4)
        cv2.putText(image, 'STAGE:', (240,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(left_stage) if left_stage else 'None', (240,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.putText(image, 'ELBOW ANGLE:', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(int(left_angle)) if 'left_angle' in locals() else '0', (20,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

        # Right arm box
        cv2.rectangle(image, (0,240), (400,470), (245,117,16), -1)
        cv2.putText(image, 'RIGHT REP COUNT', (20,270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(right_counter), (20,330), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 4)
        cv2.putText(image, 'STAGE:', (240,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(right_stage) if right_stage else 'None', (240,330), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.putText(image, 'ELBOW ANGLE:', (20,390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(image, str(int(right_angle)) if 'right_angle' in locals() else '0', (20,440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing_styles.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing_styles.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
