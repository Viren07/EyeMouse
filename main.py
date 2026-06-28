from src.gaze import Vision
import cv2

# Calling the class from gaze.py file 
vision = Vision()

# Loop to make sure the function is called and running constatnly 
while True:

    frame = vision.frame_capture() #storing the frame in a varible 

    # Making sure the program does not send an error when a person is not detected and skips that frame instead 
    if frame is None:
        continue

    gaze_ratio_x = vision.iris_position # Getting the eye iris ratio from gaze.py
    cv2.putText(frame, str(gaze_ratio_x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #displaying the coordinates of where I am looking on the screen (x-axis)

    gaze_ratio_y = vision.iris_position_vertical # Getting the eye lid ratio from gaze.py
    cv2.putText(frame, str(gaze_ratio_y), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #displaying the coordinates of where I am looking on the screen (y-axis)
    
    frame = cv2.resize(frame, (960, 540)) # This is for setting the resolution of the app that launches 

    cv2.imshow("EyeMouse", frame) # Displaying the page with the camera 

    #when q is pressed we end the loop and the program stops 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

#Clearning all the frames and closing the webcam connection
vision.videocapture.release()
cv2.destroyAllWindows()

