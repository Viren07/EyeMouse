import cv2

class Calibration: 
    """This class is responsible for calibrating the screen space."""
    
    # Stores the 5 known screen positions used for calibration and an empty list to collect the user's gaze ratios at each point
    def __init__(self):
        self.screen_points = [(0, 0), (1920, 0), (0, 1080), (1920, 1080), (960, 540)] #5 calibration key points on the screen (top and bottom, left and right, and center)
        self.gaze_data = [] #list for storing the eye gaze values of the 5 calibratoin points 
    
    # Displays a dot at a given screen position and waits for the user to press spacebar before capturing and storing their current gaze ratios
    def show_point(self, point, frame, gaze_ratio_x, gaze_ratio_y):

        captured = False #boolean for noting if the point has been captures 

        #looping until the point has been captured 
        while not captured:
            cv2.circle(frame, point, 10, (0, 0, 255), -1) #createing the circle to display at the 5 points
            cv2.imshow("EyeMouse", frame) #displaying the plint 

            # when space is pressed the eye gaze value is captured, if any other key is pressed a message asking to press the space bar is shown
            key = cv2.waitKey(0) 
            if key == ord(' '):
                self.gaze_data.append((gaze_ratio_x, gaze_ratio_y))
                captured = True
            else:
                cv2.putText(frame, "Please use the space bar", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)