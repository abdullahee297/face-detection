import cv2
import time
import pyautogui
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe setup
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

prev_nose_y = None
last_action_time = 0
COOLDOWN = 1.2        # seconds
THRESHOLD = 25        # movement sensitivity

print("Move face UP → Next Reel | DOWN → Previous")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        h, w, _ = frame.shape
        face = result.face_landmarks[0]

        # Nose landmark index = 1
        nose = face[1]
        nose_y = int(nose.y * h)

        cv2.circle(frame, (int(nose.x * w), nose_y), 6, (0, 0, 255), -1)

        if prev_nose_y is not None:
            diff = prev_nose_y - nose_y
            current_time = time.time()

            if abs(diff) > THRESHOLD and (current_time - last_action_time) > COOLDOWN:
                if diff > 0:
                    pyautogui.scroll(-600)   # NEXT reel
                    print("⬆ NEXT")
                else:
                    pyautogui.scroll(600)    # PREVIOUS reel
                    print("⬇ PREVIOUS")

                last_action_time = current_time

        prev_nose_y = nose_y

    cv2.imshow("Face Scroll Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
