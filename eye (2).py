import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import speech_recognition as sr
import threading
import sys

pyautogui.FAILSAFE = False

print("Program started")

# ---------------- GLOBALS ----------------
running = True
control_active = True

smooth_x = 0
smooth_y = 0

last_click = 0
blink_start = None

# ---------------- SETTINGS ----------------
SENSITIVITY = 0.25
DEAD_ZONE_X = 5
DEAD_ZONE_Y = 3
SMOOTHING = 0.7

BLINK_THRESHOLD = 10
BLINK_TIME = 0.2
CLICK_COOLDOWN = 1

SCROLL_THRESHOLD = 15

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

cam = cv2.VideoCapture(0)

def get_landmark_coords(landmarks,index,w,h):
    return np.array([
        int(landmarks[index].x * w),
        int(landmarks[index].y * h)
    ])

# ---------------- VOICE CONTROL ----------------
def voice_control():
    global running, control_active
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Voice control started...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while running:
            try:
                audio = recognizer.listen(source, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()

                print("You said:", command)

                if "pause" in command:
                    control_active = False

                elif "start" in command:
                    control_active = True

                elif "double click" in command:
                    pyautogui.doubleClick()

                elif "click" in command:
                    pyautogui.click()

                elif "scroll down" in command:
                    pyautogui.scroll(-100)

                elif "scroll up" in command:
                    pyautogui.scroll(100)

                elif "stop" in command:
                    running = False
                    break

            except:
                pass

threading.Thread(target=voice_control, daemon=True).start()

# ---------------- CALIBRATION ----------------
print("Look straight at screen")

center_samples = []

for _ in range(30):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0].landmark

        left = np.mean([get_landmark_coords(mesh,i,frame.shape[1],frame.shape[0]) for i in LEFT_IRIS], axis=0)
        right = np.mean([get_landmark_coords(mesh,i,frame.shape[1],frame.shape[0]) for i in RIGHT_IRIS], axis=0)

        center_samples.append((left + right)/2)

    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

center_point = np.mean(center_samples, axis=0)

print("Calibration done")

# ---------------- MAIN LOOP ----------------
while running:

    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        mesh = result.multi_face_landmarks[0].landmark

        left = np.mean([get_landmark_coords(mesh,i,w,h) for i in LEFT_IRIS], axis=0)
        right = np.mean([get_landmark_coords(mesh,i,w,h) for i in RIGHT_IRIS], axis=0)

        iris_center = (left + right)/2
        cv2.circle(frame, tuple(left.astype(int)), 3, (0,255,0), -1)
        cv2.circle(frame, tuple(right.astype(int)), 3, (0,255,0), -1)

        dx = iris_center[0] - center_point[0]
        dy = iris_center[1] - center_point[1]

        if control_active:

            # -------- MOVEMENT --------
            if abs(dx) < DEAD_ZONE_X:
                dx = 0
            if abs(dy) < DEAD_ZONE_Y:
                dy = 0

            move_x = dx * SENSITIVITY
            move_y = dy * SENSITIVITY * 1.5   # ✅ correct direction

            # reduce tiny noise
            if abs(move_y) < 1:
                move_y = 0

            smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * move_x
            smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * move_y

            pyautogui.moveRel(int(smooth_x), int(smooth_y))

            # -------- SCROLL --------
            if dy > SCROLL_THRESHOLD:
                pyautogui.scroll(-40)
            elif dy < -SCROLL_THRESHOLD:
                pyautogui.scroll(40)

            # -------- BLINK CLICK + DOUBLE CLICK --------
            top = get_landmark_coords(mesh,LEFT_EYE_TOP,w,h)
            bottom = get_landmark_coords(mesh,LEFT_EYE_BOTTOM,w,h)

            blink_distance = np.linalg.norm(top-bottom)
            current_time = time.time()

            if blink_distance < BLINK_THRESHOLD:

                if blink_start is None:
                    blink_start = current_time

                elif current_time - blink_start > BLINK_TIME:

                    if current_time - last_click < 0.5:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.click()

                    last_click = current_time
                    blink_start = None

            else:
                blink_start = None

        else:
            smooth_x = 0
            smooth_y = 0
            blink_start = None

    # -------- UI --------
    status = "ACTIVE" if control_active else "PAUSED"
    cv2.putText(frame,f"Mode: {status}",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Eye Mouse", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        control_active = not control_active

    elif key == ord('q'):
        running = False
        break

# ---------------- EXIT ----------------
cam.release()
cv2.destroyAllWindows()

print("Program exited")
sys.exit()