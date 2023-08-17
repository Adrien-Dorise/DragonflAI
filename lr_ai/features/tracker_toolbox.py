"""
 Toolbox to process data using the eyetracker, for the faceMask app
 Author: Julia Cohen (jcohen@lrtechnologies.fr) - LR Technologies
 Created: March 2023
 Last updated: Julia Cohen - March 2023
"""

import cv2
import numpy as np

from tracker.camerasensor import MeshFaceDetector

# -----------------------------------------------------
# Sets of points of interest
LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 
            388, 390, 398, 466] 

RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 
             161, 163, 173, 246]  

PUPILS = list(range(468, 478))

MOUTH = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 
         146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 
         314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

MOUTH_CONTOUR = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 
                 291, 314, 321, 375, 405, 409]

FACE_CONTOURS = [9, 152, 127, 356]   # pt 10 is too high, migh be cut from the frame
LARGE_FACE_CONTOURS = [10, 152, 127, 356]

FULL_EYES = LEFT_EYE+RIGHT_EYE+PUPILS  # total of 42 pts
# eye corners: 362, 263, 33, 133  # mouth corners: 61, 291
V1 = [362, 263, 33, 133, 6, 5, 61, 291, 0] + FACE_CONTOURS  # total of 13pts
V2 = FULL_EYES + [5, 6, 8] + MOUTH_CONTOUR + FACE_CONTOURS  # total of 69pts

ALL_POINTS = list(range(478))


POINTS_SELECTOR = {0: ALL_POINTS,
                   1: V1,
                   2: V2}
# -----------------------------------------------------

def crop_face(frame, landmarks, offset=10):
    """
    Crop a portion of the frame, focused on the face.

    Parameters
    ----------
    frame: np.ndarray/frame loaded by opencv
        The frame to crop.
    landmarks: mediapipe FaceMesh SolutionOutput
        Output of the eyetracker applied to the frame
    offset: int
        Number of pixels to keep around the face. Defaults to 10.
    Returns
    -------
    the cropped face from input frame
    """
    return _crop(frame, landmarks, LARGE_FACE_CONTOURS, offset)

def crop_eyes(frame, landmarks, offset=5):
    """
    Crop a portion of the frame, focused on the eyes.

    Parameters
    ----------
    frame: np.ndarray/frame loaded by opencv
        The frame to crop.
    landmarks: mediapipe FaceMesh SolutionOutput
        Output of the eyetracker applied to the frame
    offset: int
        Number of pixels to keep around the eyes. Defaults to 5.
    Returns
    -------
    the cropped eyes from input frame
    """
    return _crop(frame, landmarks, LEFT_EYE+RIGHT_EYE, offset)

def _crop(frame, landmarks, select_set, offset):
    """
    Utility function to crop a frame using a set of points

    Parameters
    ----------
    frame: np.ndarray/frame loaded by opencv
        The frame to crop.
    landmarks: mediapipe FaceMesh SolutionOutput
        Output of the eyetracker applied to the frame
    select_set: list
        Subset of points.
    offset: int
        Number of pixels to keep around the eyes. 
    Returns
    -------
    the cropped eyes from input frame
    """
    min_x = 1
    max_x = 0
    min_y = 1
    max_y = 0
    for i in select_set:
        lmk = landmarks.multi_face_landmarks[0].landmark[i]
        min_x = lmk.x if lmk.x < min_x else min_x
        max_x = lmk.x if lmk.x > max_x else max_x
        min_y = lmk.y if lmk.y < min_y else min_y
        max_y = lmk.y if lmk.y > max_y else max_y
    h, w, _ = frame.shape
    min_x, max_x = int(min_x*w), int(max_x*w)
    min_y, max_y = int(min_y*h), int(max_y*h)
    
    begin_x = max(0, min_x-offset)
    end_x = min(w, max_x+offset)
    begin_y = max(0, min_y-offset)
    end_y = min(h, max_y+offset)
    cropped = frame[begin_y:end_y, begin_x:end_x, :]
    return cropped

def select_points(points, coords="xy", version=1):
    """
    Select some of the tracker keypoints coordinates from the numpy array stored 
    in the csv file.

    Parameters
    ----------
    points: np.ndarray
        datapoints loaded from the csv file (without index column).
    coords: str, optional
        specify if we keep the 3D coordinates ('xyz') or only 2D 
        coordinates ('xy'). Defaults to "xy".
    version: int, optional
        identifier of the set of keypoints to keep. Defaults to 1. 
        Options are:
            - 0: keep all 478 points.
            - 1: V1, keep 13 points (2pts for the sides of the face, 1pt for 
                    the chin, 4pts for the corners of the eyes, 4pts for the 
                    face vertical axis, 2pts for the corners of the mouth)
            - 2: V2, keep 69 points (2pts for the sides of the face, 1pt for 
                    the chin, 10pts for the pupils, 32pts for the eyes, 4pts 
                    for the face vertical axis, 20 pts for the mouth contour)
    Returns
    -------
    a numpy array with only the selected columns
    """
    assert isinstance(points, np.ndarray), f"Argument 'points' should be a numpy array, got {type(points)}."
    assert points.shape[1] == (478*3), f"Argument 'points' should have {478*3} columns, got a shape of {points.shape}"

    indexes = list()
    for i in POINTS_SELECTOR[version]:
        if coords == 'xy':
            indexes.append(i*3)
            indexes.append(i*3+1)
        elif coords == 'xyz':
            indexes.append(i*3)
            indexes.append(i*3+1)
            indexes.append(i*3+2)
    selected = points[:, indexes]
    return selected

def get_detector():
    """
    Instantiate a detector to predict the face points.
    Parameters
    ----------
    Returns
    -------
    a MeshFaceDetector object from camerasensor package.
    """
    return MeshFaceDetector(refine_landmarks=True, static_image_mode=True)

def get_landmarks(frame, detector=None):
    """
    Predicts the keypoints detected by the detector on the given frame.

    Parameters
    ----------
    frame: np.ndarray/opencv frame IN RGB FORMAT
        input RGB frame.
    detector: MeshFaceDetector
        detector already instantiated, or None to produce a new detector.
    Returns
    -------
    
    """
    detector = get_detector() if detector is None else detector
    landmarks = detector.detect(frame, roi=None)
    landmarks = detector.postprocess_detections(landmarks)
    return landmarks



if __name__ == "__main__":
    image = r"data/face2.jpg"

    frame = cv2.imread(image)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = get_detector()
    landmarks = get_landmarks(rgb_frame, detector)

    data = list()
    h, w, _ = frame.shape
    print("Shape before resize=", frame.shape)

    reshaped = cv2.resize(frame, (w*3, h*3))
    print("Shape after resize=", reshaped.shape)
    h, w, _ = reshaped.shape
    try:
        for i in range(len(landmarks.multi_face_landmarks[0].landmark)):
            lmk = landmarks.multi_face_landmarks[0].landmark[i]
            data.append(lmk.x)
            data.append(lmk.y)
            data.append(lmk.z)

        # Change the set from which selecting the index to display them
        for IDX in V2:
            # Use .copy() to display points one by one
            show = reshaped  #.copy()
            x, y = int(data[IDX*3]*w), int(data[IDX*3+1]*h)
            if IDX in MOUTH:
                color = (255, 0, 0)
            elif IDX in V1:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.circle(show, (x, y), 2, color)
            cv2.putText(show, str(IDX), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0))
            cv2.imshow("Frame", show)
            # use waitKey(0) to pause between each point (useful when using .copy())
            # use waitKey(1) to draw all points and see the result on the same 
            # frame (and remove .copy())
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
        cv2.imshow("Frame", show)
        cv2.waitKey(0)

        cropped_face = crop_face(frame, landmarks)
        cv2.imshow("Cropped face", cropped_face)
        cv2.waitKey(0)
        cropped_eyes = crop_eyes(frame, landmarks)
        cv2.imshow("Cropped eyes", cropped_eyes)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()

