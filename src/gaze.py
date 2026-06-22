import mediapipe as mp
import cv2

class Vision:
    """This class is responsible for capturing frames from the camera live feed."""

    # Responsible for initialization and setting up the camera for capture
    def __init__(self):
        self.videocapture = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.iris_position = None
        self.ear_value = None
        self.head_tilt = None

    # Responsible for capturing the frame correctly from the video live feed
    def frame_capture(self):
        ret, frame = self.videocapture.read()  # ret stores True/False value telling if the frame was captured, frame is the actual image
        if not ret:
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting it into RGB color from BGR

        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return
        
        face_landmarks = results.multi_face_landmarks[0]