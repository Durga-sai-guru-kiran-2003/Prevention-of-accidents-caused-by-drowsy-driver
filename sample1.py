import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

# Constants
CONSECUTIVE_FRAMES = 30  # Number of frames for drowsiness to trigger an alert
HEAD_POSE_THRESHOLD = 15  # A threshold angle for head pose factor adjustment
YAWN_THRESHOLD = 0.95  # Threshold for MAR to detect yawning

# Initialize dlib's face detector and facial landmark predictor
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alarm.wav")  # Replace 'alarm.wav' with your own alert sound file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper function to convert dlib shape object to numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Helper function to calculate mEAR (modified EAR)
def modified_eye_aspect_ratio(eye, head_pose_matrix):
    """Calculate modified Eye Aspect Ratio (mEAR) based on extra eyelid landmarks and head pose adjustments."""
    # Define the specific eye landmarks using the eye array
    p1, p2, p3, p4, p5, p6 = eye  # Left or right eye landmarks

    # Calculate distances
    vertical_dist1 = dist.euclidean(p2,p6)
    vertical_dist2 = dist.euclidean(p3,p5)
    horizontal_dist = dist.euclidean(p1,p4)

    additional_eyelid_points = 0  # Modify based on additional eyelid landmarks
    mear = (vertical_dist1 + vertical_dist2 + additional_eyelid_points) / (2 * horizontal_dist)

     # Head pose adjustment
    head_pose_factor = np.abs(head_pose_matrix[0])  # Assuming head_pose_matrix is a 3x3 rotation matrix (adjust accordingly)

    # If head_pose_factor is an array, use np.any() or np.all() to extract scalar value for comparison
    if np.any(head_pose_factor > HEAD_POSE_THRESHOLD):  # This will check if any element of the array exceeds the threshold
        mear = mear * (1 - np.max(head_pose_factor) / 100)  # Use the max value for adjustment

    return mear

# Helper function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) to detect yawning."""
    # Calculate the vertical distances between points (top and bottom parts of the mouth)
    vert_dist1 = dist.euclidean(mouth[0],mouth[6])  # Outer left and right corners
    vert_dist2 = dist.euclidean(mouth[1],mouth[5])  # Upper left and lower left
    vert_dist3 = dist.euclidean(mouth[2],mouth[4])  # Upper right and lower right

    # Calculate the horizontal distance (width of the mouth)
    hor_dist = dist.euclidean(mouth[3],mouth[9])  # Left and right corners of the mouth

    # MAR calculation
    mar = (vert_dist1 + vert_dist2 + vert_dist3) / (3 * hor_dist)

    return mar

# Function to calibrate the EAR
def calibrate_ear(ear_values):
    """Compute the personalized EAR threshold based on calibration data."""
    if len(ear_values) == 0:
        print("Warning: No EAR values collected during calibration.")
        return 0.25  # Default threshold if no data was collected

    # Compute the baseline EAR, ensuring the list is not empty
    baseline_ear = np.mean(ear_values)
    
    # Check for NaN values in the mean (in case of invalid data)
    if np.isnan(baseline_ear):
        print("Warning: Invalid EAR values detected during calibration.")
        return 0.25  # Default threshold if invalid data is found

    return baseline_ear * 0.75  # Set threshold at 75% of baseline EAR

# Collect EAR values for 5 seconds to establish a baseline
def collect_calibration_data():
    """Collect EAR data to establish baseline threshold."""
    calibration_data = []
    start_time = time.time()

    while time.time() - start_time < 5:
        # Get the current frame and landmarks
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = shape_to_np(landmarks)  # Convert to NumPy array

            # Extract eye landmarks (36-41 are for the left eye and 42-47 are for the right eye)
            left_eye = landmarks[36:42]  # Left eye (6 points)
            right_eye = landmarks[42:48]  # Right eye (6 points)

            # Get the head pose matrix (just a placeholder, need implementation)
            head_pose_matrix = get_head_pose(landmarks,frame)

            # Use the mEAR for eye aspect ratio calculation
            mear_left = modified_eye_aspect_ratio(left_eye, head_pose_matrix)
            mear_right = modified_eye_aspect_ratio(right_eye, head_pose_matrix)
            
            # Optionally, you can average both eyes' mEAR values
            mear = (mear_left + mear_right) / 2.0

            # Append the EAR to the calibration data
            calibration_data.append(mear)

    # Compute personalized threshold
    personalized_threshold = calibrate_ear(calibration_data)
    return personalized_threshold

# Head pose estimation using PnP (Perspective-n-Point)
def get_head_pose(landmarks,frame):
    """Estimate the head pose matrix based on facial landmarks."""
    # 3D model points (based on a simple 3D face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype="double")

    # Corresponding 2D image points
    image_points = np.array([
        (landmarks[30][0], landmarks[30][1]), # Nose tip
        (landmarks[8][0], landmarks[8][1]),   # Chin
        (landmarks[36][0], landmarks[36][1]), # Left eye left corner
        (landmarks[45][0], landmarks[45][1]), # Right eye right corner
        (landmarks[48][0], landmarks[48][1]), # Left mouth corner
        (landmarks[54][0], landmarks[54][1])  # Right mouth corner
    ], dtype="double")

    # Camera matrix (assuming no zoom, etc.)
    size = (640, 480)  # Adjust this based on your camera resolution
    focal_length = size[0]
    center = (size[0] / 2, size[1] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    # Distortion coefficients (assuming no lens distortion)
    dist_coeffs = np.zeros((4, 1))

    # Solve for head pose using PnP
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Project 3D points to 2D to visualize the pose
    #projected_points, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, distCoeffs=None)

    # If successful, Return the rotation matrix if success
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return rotation_matrix
    else:
        print("Head pose estimation failed.")
        return None

# Get current EAR value
def get_current_ear(left_eye, right_eye, head_pose_matrix):
    mear_left = modified_eye_aspect_ratio(left_eye, head_pose_matrix)
    mear_right = modified_eye_aspect_ratio(right_eye, head_pose_matrix)
    return (mear_left + mear_right) / 2.0

# Initialize video capture and counters
cap = cv2.VideoCapture(0)
COUNTER = 0
ALERT_FLAG = False

# Calibrate EAR (collect data for 5 seconds)
PERSONALIZED_EAR_THRESHOLD = collect_calibration_data()

print(f"Personalized EAR Threshold: {PERSONALIZED_EAR_THRESHOLD}")

frame_counter = 0
frame_skip = 5  # Process every 5th frame
key = -1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Skip frames for performance optimization
    if frame_counter % frame_skip != 0:
        frame_counter += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = shape_to_np(landmarks)  # Convert dlib shape to numpy array

        # Extract eye landmarks (36-41 are for the left eye and 42-47 are for the right eye)
        left_eye = landmarks[36:42]  # Left eye (6 points)
        right_eye = landmarks[42:48]  # Right eye (6 points)
        
        # Extract mouth landmarks (48-54 for the mouth)
        mouth = landmarks[48:60]  # Mouth (12 points from 48 to 59)
        
        # Get the head pose matrix
        head_pose_matrix = get_head_pose(landmarks,frame)

        # Calculate the mEAR for both eyes
        mear = get_current_ear(left_eye, right_eye, head_pose_matrix)

        # Draw contours for the eyes (mEAR)
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 0, 255), 2)  # Left eye contour in red
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 0, 255), 2)  # Right eye contour in red
        
        # Draw contours for the mouth (MAR)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 2)  # Mouth contour in green

        # Check if the mEAR is below the personalized threshold for consecutive frames
        if mear < PERSONALIZED_EAR_THRESHOLD:
            COUNTER += 1
            if ALERT_FLAG:
                if not (pygame.mixer.get_busy()):
                    alert_sound.play()
            if COUNTER >= CONSECUTIVE_FRAMES:
                if not ALERT_FLAG:
                    ALERT_FLAG = True
                else:
                    cv2.putText(frame, "Drowsiness Alert!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALERT_FLAG = False
            pygame.mixer.Sound.stop(alert_sound)

        # Calculate MAR for yawning detection
        mar = mouth_aspect_ratio(mouth)

        # Check if the MAR exceeds the yawning threshold
        if mar < YAWN_THRESHOLD:
            cv2.putText(frame, "Yawning Alert!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the mEAR and MAR values on the frame
        cv2.putText(frame, "mEAR: {:.2f}".format(mear), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Press 'v' draw specific lines for eyes and mouth
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            cv2.line(frame, tuple(left_eye[1]), tuple(left_eye[5]), (0,0,0), 2)
            cv2.line(frame, tuple(left_eye[2]), tuple(left_eye[4]), (0,0,0), 2)
            cv2.line(frame, tuple(left_eye[0]), tuple(left_eye[3]), (0,0,0), 2)  

            cv2.line(frame, tuple(right_eye[1]), tuple(right_eye[5]), (0,0,0), 2)  
            cv2.line(frame, tuple(right_eye[2]), tuple(right_eye[4]), (0,0,0), 2)  
            cv2.line(frame, tuple(right_eye[0]), tuple(right_eye[3]), (0,0,0), 2)  

            cv2.line(frame, tuple(mouth[3]), tuple(mouth[9]), (0,0,0), 2)  
            cv2.line(frame, tuple(mouth[2]), tuple(mouth[10]), (0,0,0), 2)  
            cv2.line(frame, tuple(mouth[4]), tuple(mouth[8]), (0,0,0), 2)  
            cv2.line(frame, tuple(mouth[1]), tuple(mouth[5]), (0,0,0), 2)  
            cv2.line(frame, tuple(mouth[0]), tuple(mouth[6]), (0,0,0), 2)  
            cv2.line(frame, tuple(mouth[7]), tuple(mouth[11]), (0,0,0), 2)
        elif key == ord('q'):
            break
    cv2.imshow("Driver Drowsiness Detection", frame)

    if key == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
