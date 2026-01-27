import cv2
import time
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


while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        for face in result.face_landmarks:
            h, w, _ = frame.shape
            
            lm_list = []
            
            for lm in face:
                lm_list.append((int(lm.x*w), int(lm.y*h)))
                
            for x, y in lm_list:
                cv2.circle(frame, (x, y), 1, (255, 255, 0), cv2.FILLED)

    cv2.imshow("Face  Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()