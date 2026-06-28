import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class Vision:
    """This class is responsible for capturing frames from the camera live feed."""

    # Responsible for initialization and setting up the camera for capture
    def __init__(self):
        self.videocapture = cv2.VideoCapture(0)
        self.iris_position = None
        self.iris_position_vertical = None
        self.ear_value = None
        self.head_tilt = None

        model_path = "model/face_landmarker.task"
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)
        
        self.landmarker = FaceLandmarker.create_from_options(options)

    # Responsible for capturing the frame correctly from the video live feed
    def frame_capture(self):
        ret, frame = self.videocapture.read()  # ret stores True/False value telling if the frame was captured, frame is the actual image
        if not ret:
            return
        
        frame = cv2.flip(frame, 1) # Flip the frame horizontally so left/right in the image matches the user's actual left/right, since raw camera frames are mirrored
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting it into RGB color from BGR

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # Wraps the RGB frame into MediaPipe's required Image format for the new Tasks API

        timestamp_ms = int(time.time() * 1000) # Get the current time in milliseconds

        face_landmarker_result = self.landmarker.detect_for_video(mp_image, timestamp_ms) # Runs face detection on this frame using the model, passing in the image and its timestamp

        # # Skip this frame if no face was detected, to avoid errors trying to process empty data
        if not face_landmarker_result.face_landmarks:
            return

        face_landmarks = face_landmarker_result.face_landmarks[0] # # Grabs the first (and only) detected face's landmarks from this frame, for use in the calculations below


        # Get the coordinates for the right, left iris etc... using get_landmark_coordinates
        right_iris = self.get_landmark_coordinates(face_landmarks, [468, 469, 470, 471, 472])
        left_iris = self.get_landmark_coordinates(face_landmarks, [473, 474, 475, 476, 477])
        right_eye_corners = self.get_landmark_coordinates(face_landmarks, [33, 133])
        left_eye_corners = self.get_landmark_coordinates(face_landmarks, [362, 263])
        right_eye_lids = self.get_landmark_coordinates(face_landmarks, [159, 145])
        left_eye_lids = self.get_landmark_coordinates(face_landmarks, [386, 374])

        #getting the ratio for the eyes using iris_eye_ratio on the x axis
        right_ratio = self.iris_eye_ratio(right_eye_corners, right_iris)
        left_ratio = self.iris_eye_ratio(left_eye_corners, left_iris)
        avg_ratio = (right_ratio + left_ratio) / 2
        self.iris_position = avg_ratio

        # Getting the ratio for the eye on the y axis 
        right_ratio_lids = self.iris_lid_ratio(right_eye_lids, right_iris)
        left_ratio_lids = self.iris_lid_ratio(left_eye_lids, left_iris)
        avg_ratio_lids = (right_ratio_lids + left_ratio_lids) / 2
        self.iris_position_vertical = avg_ratio_lids

        return frame


    # Returns a list of (x, y) coordinates for the specific landmark indices requested
    def get_landmark_coordinates(self, face_landmarks, indices):
        
        coordinates  = []
        for curr in indices:
            point = face_landmarks[curr]
            coordinates.append((point.x, point.y))
        return coordinates
    

    # calculates the ratio for how far the iris is moving rletive to the eye size (x axis)
    def iris_eye_ratio(self, corner, iris):
        iris_x = iris[0][0]
        corner1_x = corner[0][0]
        corner2_x = corner[1][0]

        ratio = (iris_x - corner1_x ) / (corner2_x - corner1_x) 

        return ratio
    
    

    # calculates the ratio for how far the iris is moving rletive to the lids (y axis)
    def iris_lid_ratio(self, lids, iris):
        iris_y = iris[0][1]
        lids1_y = lids[0][1]
        lids2_y = lids[1][1]

        ratio = (iris_y - lids1_y ) / (lids2_y - lids1_y) 

        return ratio


