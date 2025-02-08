# Prevention-of-accidents-caused-by-drowsy-driver

---

# Driver Drowsiness Detection

This project is an implementation of a driver drowsiness detection system using computer vision and machine learning techniques. It tracks the eye aspect ratio (EAR) and mouth aspect ratio (MAR) to detect drowsiness and yawning, triggering an alert when the driver exhibits signs of drowsiness or is yawning. It also includes head pose estimation to improve accuracy.

## Requirements

1. Python 3.x
2. Install required libraries:
   - OpenCV
   - dlib
   - NumPy
   - pygame
   - scipy

You can install the required libraries using `pip`:

```bash
pip install opencv-python dlib numpy pygame scipy
```

You will also need:
- A webcam connected to your computer.
- A pre-trained shape predictor file `shape_predictor_68_face_landmarks.dat`. You can download it from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

## Project Overview

The system uses:
- **dlib's facial landmark detector** to identify facial landmarks (eyes, mouth, nose).
- **Eye Aspect Ratio (EAR)** to detect drowsiness (eyes closing).
- **Mouth Aspect Ratio (MAR)** to detect yawning.
- **Head Pose Estimation** using the Perspective-n-Point (PnP) algorithm to account for the driver's head position.

The system triggers an **alert** if:
- The **EAR** falls below a certain threshold indicating drowsiness.
- The **MAR** exceeds a threshold indicating yawning.

### Sound Alert
The system plays an alert sound (`alarm.wav`) when drowsiness is detected.

### Key Features:
- **Real-time Drowsiness Detection**: Monitors the eyes and mouth in real-time using a webcam.
- **Personalized Calibration**: Automatically calibrates the EAR threshold for each user based on the baseline EAR during the initial 5 seconds of operation.
- **Yawning Detection**: Detects when the user is yawning by calculating the Mouth Aspect Ratio (MAR).
- **Head Pose Adjustment**: Adjusts the eye aspect ratio based on the head pose to improve accuracy in detecting drowsiness.

## Usage

1. Run the Python script.

   ```bash
   python drowsiness_detection.py
   ```

2. The system will ask you to calibrate the EAR by tracking the eye movements for 5 seconds. This will set a personalized threshold for drowsiness detection.

3. The system will continuously monitor the eyes and mouth for signs of drowsiness or yawning. If detected, it will display a warning message on the screen and play an alert sound.

4. Press 'v' to draw lines over the eyes and mouth to visualize the landmarks used for the detection.

5. Press 'q' to quit the application.

## Code Explanation

### Key Functions:

- **`collect_calibration_data()`**: Collects EAR data for 5 seconds to calibrate the system for the user.
- **`modified_eye_aspect_ratio()`**: Calculates the modified EAR, taking into account head pose adjustments.
- **`mouth_aspect_ratio()`**: Calculates MAR to detect yawning.
- **`get_head_pose()`**: Estimates the head pose based on facial landmarks using the PnP algorithm.
- **`get_current_ear()`**: Computes the current EAR value based on detected eye landmarks.

### Constants:
- **`CONSECUTIVE_FRAMES`**: Number of frames the EAR must remain below the threshold to trigger a drowsiness alert.
- **`HEAD_POSE_THRESHOLD`**: Threshold for head pose adjustments to improve EAR accuracy.
- **`YAWN_THRESHOLD`**: Threshold for detecting yawning based on MAR.

## Troubleshooting

- **Low FPS or lag**: This might happen if your computer is not powerful enough to handle real-time video processing. Try reducing the video resolution or optimizing the code.
- **Sound not playing**: Ensure you have a valid sound file (`alarm.wav`) in the same directory as the script or update the file path.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [dlib](http://dlib.net/): For the face landmark detection.
- [OpenCV](https://opencv.org/): For real-time computer vision tasks.
- [pygame](https://www.pygame.org/): For sound and multimedia functionality.

---
