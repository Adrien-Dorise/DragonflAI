"""
 Data related (paths, preprocessing...) parameters
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

from dragonflai.experiment import InputType, ParamType
from sklearn.preprocessing import MinMaxScaler



param = ParamType.MOVE_CURSOR


if(param == ParamType.MOVE_CURSOR):
    train_path = r"data/split1/train"
    val_path = r"data/split1/val"
    test_path = r"data/split1/test"
    visu_path = r"data/visu"
    save_path = r"models/tmp/dummy_experiment"

    train_path = val_path = test_path = visu_path = r"data/debug_move_cursor"
    
    input_type = InputType.TRACKER
    seq_length = 0
    crop = "face"
    coords= "xyz"
    tracker_version = 1
    scaler = MinMaxScaler()
    nb_workers = 0


elif(param == ParamType.LOOK_AT_SCREEN): #Set InputType to IMAGE, and scaler to None
    train_path = r"data/look_at_screen/train"
    val_path = r"data/look_at_screen/valid"
    test_path = r"data/look_at_screen/test"
    visu_path = r"data/debug_lookAtScreen/lookAtScreen"
    save_path = r"models/tmp/dummy_experiment"

    train_path = val_path = test_path = r"data/debug_lookAtScreen"
    
    input_type = InputType.IMAGE
    scaler = None

    seq_length = 0
    crop = "face"
    coords= "xyz"
    tracker_version = 1
    nb_workers = 0
