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
        
        frame = cv2.flip(frame, 1) # Flip the frame horizontally so left/right in the image matches the user's actual left/right, since raw camera frames are mirrored
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting it into RGB color from BGR

        results = self.face_mesh.process(rgb_frame) # this line runs the model

        if not results.multi_face_landmarks:
            return
        
        # Holds all 478 facial landmarks detected in this frame; passed into get_landmark_coordinates so it can extract specific coordinates from it
        face_landmarks = results.multi_face_landmarks[0]

        # Get the coordinates for the right, left iris etc... using get_landmark_coordinates
        right_iris = self.get_landmark_coordinates(face_landmarks, [468, 469, 470, 471, 472])
        left_iris = self.get_landmark_coordinates(face_landmarks, [473, 474, 475, 476, 477])
        right_eye_corners = self.get_landmark_coordinates(face_landmarks, [33, 133])
        left_eye_corners = self.get_landmark_coordinates(face_landmarks, [362, 263])

        #getting the ratio for the eyes using iris_eye_ratio
        right_ratio = self.iris_eye_ratio(right_eye_corners, right_iris)
        left_ratio = self.iris_eye_ratio(left_eye_corners, left_iris)
        avg_ratio = (right_ratio + left_ratio) / 2
    


    # Returns a list of (x, y) coordinates for the specific landmark indices requested
    def get_landmark_coordinates(self, face_landmarks, indices):
        
        coordinates  = []
        for curr in indices:
            point = face_landmarks.landmark[curr]
            coordinates.append((point.x, point.y))
        return coordinates
    
    # calculates the ratio for how far the iris is moving rletive to the eye size 
    def iris_eye_ratio(self, corner, iris):
        iris_x = iris[0][0]
        corner1_x = corner[0][0]
        corner2_x = corner[1][0]

        ratio = (iris_x - corner1_x ) / (corner2_x - corner1_x) 

        return ratio




