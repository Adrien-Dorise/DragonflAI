import argparse
import os

import numpy as np
import cv2

from lr_ai.features.tracker_toolbox import get_detector, V1, V2


def load_video(path):
    """
    Load the video and return a cv2.VideoCapture object to iterate over to 
    get the frames.

    Args:
        path (str): path to the video to load.

    Returns:
        cv2.VideoCapture: object containing the video.
    """
    assert os.path.isfile(path), f"Path {path} is incorrect"
    return cv2.VideoCapture(path)

def load_image(path):
    """
    Load and return an image.

    Args:
        path (str): _description_
    Returns:
        np.ndarray:array containing the frame as BGR
    """
    assert os.path.isfile(path), f"Path {path} is incorrect"
    return cv2.imread(path)

def preprocess_frame(frame):
    """
    Convert a BGR to RGB frame.

    Args:
        frame (np.ndarray): imagein BGR format

    Returns:
        np.ndarray: RGB frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def postprocess_frame(frame):
    """
    Convert a RGB to BGR frame.

    Args:
        frame (np.ndarray): image in RGB format.

    Returns:
        np.ndarray: BGR frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_one_frame(frame, detector, background_frame=None):
    """
    Compute landmarks and draw them on 4 frames.
    Frame 0: face mesh (mediapipe style).
    Frame 1: all points.
    Frame 2: points from tracker_toolbox.V1 subset.
    Frame 3: points from tracker_toolbox.V2 subset.

    Args:
        frame (np.ndarray): BGR frame
        detector (lr_facetracker.detection.MeshFaceDetector): landmarks detector. 
                Can be obtained using tracker_toolbox.get_detector().
        background_frame (np.ndarray): Frame used for background when drawing the new frame. If None, the main frame is used. Default to None

    Returns:
        list[np.ndarray*4]: 4 BGR frames.
    """
    detection_data = detector.process(frame)
    landmarks = detection_data.get_landmarks()
    rgb_frame = preprocess_frame(frame)
    if landmarks is None:
        print("No detections for this frame")
        return
    
    # draw mesh
    frame_mesh = rgb_frame.copy()
    detector.draw_mesh(frame_mesh, detection_data)
    frame_mesh = postprocess_frame(frame_mesh)

    # draw points
    if background_frame is None:
        background_frame = frame.copy()
    frame_v0 = background_frame.copy()
    frame_v1 = background_frame.copy()
    frame_v2 = background_frame.copy()
    
    for idx, lmk in enumerate(landmarks):
        draw_point(frame_v0, lmk)
        if idx in V1:
            draw_point(frame_v1, lmk)
        if idx in V2:
            draw_point(frame_v2, lmk)

    return [frame_mesh, frame_v0, frame_v1, frame_v2]

def draw_point(frame, landmark):
    """
    Draw a green dot on the given frame.

    Args:
        frame (np.ndarray): BGR image.
        landmark (list of floats): landmarks as (x, y, z) points.
    Returns:
        None: `frame` argument is modified.
    """
    x = int(landmark[0]*frame.shape[1])
    y = int(landmark[1]*frame.shape[0])
    cv2.circle(frame, (x, y), 2, color=(0, 225, 0), thickness=-1)

def debug_show(frames_list, wait=True):
    concat_frames = np.hstack(frames_list)
    cv2.imshow("Frames", concat_frames)
    return cv2.waitKey(0 if wait else 10)

def get_writers_from_cap(cap, input_path):
    """
    Use properties of VideoCapture to produce 4 VideoWriter objects.

    Args:
        cap (cv2.VideoCapture): object containing the input video
        input_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    ext = os.path.splitext(input_path)[-1]
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name_mesh = input_path.replace(f"{ext}", f"_mesh{ext}")
    writer_mesh = cv2.VideoWriter(name_mesh,
                                 cv2.VideoWriter_fourcc('M','J','P','G'),
                                 vid_fps,
                                 (frame_width, frame_height))
    print(f"Saving video to {name_mesh}")
    
    name_v0 = input_path.replace(f"{ext}", f"_V0{ext}")
    writer_v0 = cv2.VideoWriter(name_v0,
                                 cv2.VideoWriter_fourcc('M','J','P','G'),
                                 vid_fps,
                                 (frame_width, frame_height))
    print(f"Saving video to {name_v0}")
    
    name_v1 = input_path.replace(f"{ext}", f"_V1{ext}")
    writer_v1 = cv2.VideoWriter(name_v1,
                                cv2.VideoWriter_fourcc('M','J','P','G'),
                                vid_fps,
                                (frame_width, frame_height))
    print(f"Saving video to {name_v1}")
    
    name_v2 = input_path.replace(f"{ext}", f"_V2{ext}")
    writer_v2 = cv2.VideoWriter(name_v2,
                                cv2.VideoWriter_fourcc('M','J','P','G'),
                                vid_fps,
                                (frame_width, frame_height))
    print(f"Saving video to {name_v2}")
    
    return [writer_mesh, writer_v0, writer_v1, writer_v2]


def parse_arguments():
    args = argparse.ArgumentParser("Draw tracker points on an image or video.")
    args.add_argument("--image", type=str, help="Path to the input image.")
    args.add_argument("--video", type=str, help="Path to the input video.")
    args.add_argument("--debug", action="store_true", 
                      help="Flag to display image(s) instead of saving.")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Load image or video
    if args.image is not None:
        frame = load_image(args.image)
        cap = None
    elif args.video is not None:
        cap = load_video(args.video)
        frame = None
        if not cap.isOpened():
            print(f"Error while loading video file {args.video}.")
            exit(1)
    else:
        print("Either --image or --video must have a value. Exiting now...")
        exit(0)

    detector = get_detector()

    if frame is not None:
        out_frames = process_one_frame(frame, detector)

        if args.debug:
            debug_show(out_frames, wait=True)
        else:
            ext = os.path.splitext(args.image)[-1]
            # Save mesh image
            out_path = args.image.replace(f"{ext}", f"_mesh{ext}")
            cv2.imwrite(out_path, out_frames[0])
            print(f"Saved {out_path}.")
            # Save all points image (V0)
            out_path = args.image.replace(f"{ext}", f"_V0{ext}")
            cv2.imwrite(out_path, out_frames[1])
            print(f"Saved {out_path}.")
            # Save V1 image
            out_path = args.image.replace(f"{ext}", f"_V1{ext}")
            cv2.imwrite(out_path, out_frames[2])
            print(f"Saved {out_path}.")
            # Save V2 image
            out_path = args.image.replace(f"{ext}", f"_V2{ext}")
            cv2.imwrite(out_path, out_frames[3])
            print(f"Saved {out_path}.")
    else:
        writers = get_writers_from_cap(cap, args.video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i}")
                continue
            out_frames = process_one_frame(frame, detector)

            if args.debug:
                k = debug_show(out_frames, wait=False)
                if (k & 0xFF) == ord('q'):
                    break
            else:
                for index, out_frame in enumerate(out_frames):
                    writers[index].write(out_frame)

        cap.release()
        for writer in writers:
            writer.release()
    
    cv2.destroyAllWindows()

