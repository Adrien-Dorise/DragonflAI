import os

import sys
sys.path.append("libergo-eyetracker")
from camerasensor.tracker import VideoEyeTracker

import cv2
import numpy as np
import torch.nn as nn
import torch

from LR_AI.features.preprocessing import load_metadata, getTrackerSet
from LR_AI.visualisation.display_mouse_prediction import draw_point_on_frame

def tracker_no_AI(folder_path, config_file='LR_AI/config/trackerConfig.json'):
    """This method applies an eyetracker (without AI) to an input video.
    The predictions are evaluated with the same score as our AI models.

    Algo:
    - Init tracker
    - Read videos
    - Load metadata for the videos
    - Loop over the videos
        - Loop over the frames
            - Apply the tracker to obtain the X-Y position of the mosue
            - Add predictions and targets to list
    - Compute loss
    - Print loss

    Note
    - Calibration?
    
    Args:
        folder_path (string): path to the folder with test videos
        config_file (str, optional): Path to the tracker config file. Default to 'LR_AI/config/trackerConfig.json'
    """
    video_paths = [os.path.join(folder_path, vid) for vid in os.listdir(folder_path) 
                   if vid.endswith(".avi")]
    targets_paths = [p.replace("user", "mouse").replace(".avi", ".csv") for p in video_paths]
    
    test_loss = list()
    loss_fct = nn.L1Loss()
    for i, (video_path, target_path) in enumerate(zip(video_paths, targets_paths)):
        # New tracker every video, so the mouse controller is reinit
        tracker = VideoEyeTracker(config_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error while loading video file {video_path}.")
            exit(1)
        print(f"Loading video {video_path}")
        metadata_path = target_path.replace('mouse', 'metadata')
        frame_w, frame_h = load_metadata(metadata_path)

        target = getTrackerSet(None, target_path)[1]
        target[:, 0] /= frame_w
        target[:, 1] /= frame_h

        count_frame = -1
        screen_w, screen_h = tracker.mouse_controller.get_screen_size()
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                count_frame += 1

                if not ret:
                    break

                rel_landmarks, rgb_frame = tracker.process_frame(frame)
                if rel_landmarks is None:
                    print("No detection in frame.")
                    continue
                geometry = tracker.process_geometry(rel_landmarks)

                tracker.mouse_controller.compute_mouse_target(**geometry)
                prediction = np.array([tracker.mouse_controller.X, tracker.mouse_controller.Y], dtype=np.float32)
                prediction[0] = prediction[0]/screen_w
                prediction[1] = prediction[1]/screen_h

                pred = torch.from_numpy(prediction)
                targ = torch.from_numpy(target[count_frame, :])

                # draw_point_on_frame(frame, targ, pred)
                # cv2.imshow("Frame", frame)
                # if (cv2.waitKey(10) & 0xFF) == ord('q'):
                #     break
            
                error = loss_fct(pred, targ)
                test_loss.append(error.item())
        
        cap.release()
        del tracker
    
    cv2.destroyAllWindows()
    
    mean_loss = np.mean(test_loss)
    print(f"Mean test loss={mean_loss}")



if __name__ == "__main__":
    config_path = 'LR_AI/config/trackerConfig.json'
    test_path = r"data/debug"
    tracker_no_AI(test_path, config_path)
