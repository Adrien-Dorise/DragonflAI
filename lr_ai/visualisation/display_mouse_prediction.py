import numpy as np
import cv2
import torch
import os
from enum import Enum

from lr_ai.features.preprocessing import load_metadata, getTrackerSet
from lr_ai.visualisation.draw_tracker import process_one_frame
from lr_ai.features.tracker_toolbox import get_detector, crop_face, crop_eyes, get_landmarks

class Background(Enum):
    ORIGINAL = 0
    BLACK = 1
    TRACKER_ORIGINAL = 2
    TRACKER_BLACK = 3
    

def get_writer_from_cap(cap, input_path, special_name=None):
    """
    Use properties of VideoCapture to produce a VideoWriter object.

    Args:
        cap (cv2.VideoCapture): object containing the input video
        input_path (string): video path

    Returns:
        cv2.VideoWriter: object to write frames.
    """
    ext = os.path.splitext(input_path)[-1]
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name_output = f"output/output_video{ext}"
    if special_name is not None:
        name_output = special_name
    writer_output = cv2.VideoWriter(name_output,
                                    cv2.VideoWriter_fourcc('M','J','P','G'),
                                    vid_fps,
                                    (frame_width, frame_height))
    print(f"Saving video to {name_output}")
    return writer_output


def draw_point_on_frame(frame, target, output=None):
    """
    Draw a target point (and optionnally an output point) on a frame.

    Args:
        frame (torch.Tensor or np.ndarray): 
            frame returned by a DataLoader or simple 3-channel array
        target (np.ndarray, tuple, list): 
            iterable of length 2 containing point coordinates in floats [0., 1.]
        output (np.ndarray, tuple, list): 
            (optional) iterable of length 2 containing point coordinates. 
            Default: None.
    """
    if isinstance(frame, torch.Tensor):
        assert len(frame.shape) == 3, f"Got a batch of size {frame.shape}, you "\
            "should remove the batch dimension and give only one image."
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    # Draw target
    if isinstance(target, torch.Tensor):
        target = target.detach()
    
    target[0] = 0. if target[0]<0. else target[0]
    target[1] = 0. if target[1]<0. else target[1]
    target[0] = 1. if target[0]>1. else target[0]
    target[1] = 1. if target[1]>1. else target[1]
    x = int((1.-target[0])*w)  # Flip left right for display
    y = int(target[1]*h)
    
    cv2.circle(frame, center=(x, y), radius=6, color=(0, 255, 0), thickness=-1)
    
    if output is None:
        return
    # Draw output
    if isinstance(output, torch.Tensor):
        output = output.detach()
    if len(output) == 1:
        output = output[0]
        
    output[0] = 0. if output[0]<0. else output[0]
    output[1] = 0. if output[1]<0. else output[1]
    output[0] = 1. if output[0]>1. else output[0]
    output[1] = 1. if output[1]>1. else output[1]
    x = int((1.-output[0])*w)  # Flip left right for display
    y = int(output[1]*h)
    
    cv2.circle(frame, center=(x, y), radius=6, color=(0, 0, 255), thickness=-1)
    
    
def visu_target(video_path):
    """
    Visualization of ground truth mouse position on the video, to check eye/mouse correlation.
    Left/Right camera flipping is integrated at the display level (when drawing 
    the point on the frame).
    Args:
        video_path (string): Path to the video to visualise
    """
    mouse_path = video_path.replace('user', 'mouse').replace('.avi', '.csv')
    metadata_path = mouse_path.replace('mouse', 'metadata')
    _, target = getTrackerSet(None, mouse_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error while loading video file {video_path}.")
        exit(1)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Got {num_frames} frames.")
    assert num_frames == target.shape[0], \
        f"Got {num_frames} frames, but target has shape {target.shape}"
    screen_w, screen_h = load_metadata(metadata_path)
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("No frame!")
            continue
        
        coord_x = target[i, 0]/screen_w
        coord_y = target[i, 1]/screen_h
        draw_point_on_frame(frame, [coord_x, coord_y])
        cv2.imshow("Frame + Target", frame)
        if (cv2.waitKey(int(1000/25)) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()



def compare_target_prediction(folder_path, target, prediction, save=False, frame_background = Background.ORIGINAL, tracker_version = 0, crop = None):
    """Visualization of mouse position targets vs predictions.
    Args:
        folder_path (string): Path of the folder containing the video. Note that only the first video is considered for visualisation
        target (array): Ground truth
        prediction (array): Prediction of the model
        save (bool, opt): Set to True to save the output folder. Default to False
        frame_background (Background enum, opt): Select the background. The choices are defined in the Background enumeration. Defaults to Background.ORIGINAL.
            - ORIGINAL: Display the original video
            - BLACK: Display default black background
            - TRACKER_ORIGINAL: Display the tracker mask points over the original video. Version 0 of the tracker is used for displays, and four versions are used when saving.
            - TRACKER_BLACK: Display the tracker mask points over a black background. Version 0 of the tracker is used for displays, and four versions are used when saving.
        tracker_version (int, optional): Tracker version to used for visualisation. Only used when tracker is called with the background parameter. Default to 0.
        crop (str, optional): Crop options used for with the model. Only usefull when original video is used for background. Options are {None, "face", "eyes"}. Default is None
    """


    videos = os.listdir(folder_path)
    video = [folder_path+'/'+vid for vid in videos if vid.endswith(".avi")][0]
    metadata_path = video.replace('user', 'metadata').replace('.avi', '.csv')
    screen_w, screen_h = load_metadata(metadata_path)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Error while loading video file {video}.")
        exit(1)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"\nVideo has {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames of resolution {height}x{width}")
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #wait_time = int(1000/fps)

    detector = get_detector()
    if crop == "face":
        crop_func = crop_face
    elif crop == "eyes":
        crop_func == crop_eyes

    if save:
        writer = get_writer_from_cap(cap, video, "output/output_vid.avi")
    
    for i in range(target.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print("No frame!")
            break
    
        #Set background option
        if frame_background == Background.ORIGINAL or frame_background == Background.TRACKER_ORIGINAL: #Original background
            background = frame.copy()
            #Crop option
            if crop is not None:
                landmarks = get_landmarks(frame, detector)
                if landmarks is not None:
                    background = crop_func(frame, landmarks)
                    background = cv2.resize(background,(int(width),int(height)))
        elif frame_background == Background.BLACK or frame_background == Background.TRACKER_BLACK: #Black background
            background = np.zeros((int(height), int(width), 3), dtype=np.uint8) 
        
        #Tracker option
        if frame_background == Background.TRACKER_ORIGINAL or frame_background == Background.TRACKER_BLACK:
            out_frames = process_one_frame(frame, detector, background) #output_frames is [mesh,V0,V1,V2]
            background = out_frames[tracker_version+1]
        
        draw_point_on_frame(background, target[i, :], prediction[i, :])

        cv2.imshow("Target (green) + Prediction (red) / <q> to quit", background)
        if save:
            writer.write(background)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                break
    
    cap.release()
    if save:
        writer.release()
    cv2.destroyAllWindows()

    


        
        

if __name__ == "__main__":
    import numpy as np
    test = "OUTPUT"
    """
    test == "TARGET" 
        Visualization of ground truth mouse position on the video, to check eye/mouse correlation.
        Left/Right camera flipping is integrated at the display level (when drawing 
        the point on the frame).
    test == "OUTPUT
        Visualization of mouse position targets vs predictions.
    """


    if test == "TARGET":
        path = r"data/val/user 2023-03-22 100754.avi"
        visu_target(path)


    elif test == "OUTPUT":
        import lr_ai.features.preprocessing as pr
        import lr_ai.features.image_preprocessing as imgpr
        from lr_ai.model.machineLearning import Regressor
        import lr_ai.model.neural_network_architectures.FCNN as NN
        from sklearn.preprocessing import MinMaxScaler


        #Parameters
        tracker_input = True
        data_path = "data/debug" 
        crop = 'face'
        coords="xyz"
        input_size = 69*3
        output_size = 2
        tracker_version = 2
        
        scaler = None
        model = NN.fullyConnectedNN(input_size, output_size)
        seq_length = 0
        
        #Init
        temporal = (seq_length != 0)

        if(tracker_input):
            data_set, scaler = pr.loader(data_path, shuffle = True, scaler=scaler, coords=coords, tracker_version=tracker_version, temporal=temporal, sequence_length=seq_length)
        else:
            data_set = imgpr.img_loader(data_path,True,crop=crop,shuffle=False,temporal=temporal,sequence_length=seq_length)

        
        model.loadModel("models/full_dataset/FCNN_tracker_v2/model1/epoch500_1.json")
        
        loss, (target, outputs) = model.predict(data_set)
        
        compare_target_prediction(data_path,target,outputs)
