from setuptools import setup, find_packages

setup(name="dragonflai",
      version="0.8.1",
      description="LR Technologies template for articial intelligence related projects",
      author="Adrien Dorise, Edouard Villain, Julia Cohen, Boris Lenseigne",
      author_email="{adorise, evillain, jcohen, blenseigne}@lrtechnologies.fr",
      url="https://gitlab.com/lr-technologies2/dragonflai",
      py_modules=["dragonflai.experiment"],
      packages=["dragonflai.config",
                "dragonflai.dataset", 
                "dragonflai.features", 
                "dragonflai.model", 
                "dragonflai.visualisation", 
                "dragonflai.model.neural_network_architectures",
                ],
      install_requires=[
            "cnnvis @ git+https://github.com/Adrien-Dorise/cnnvis-pytorch", 
            "lr_facetracker @ git+https://gitlab.com/lr-technologies2/lr_facetracker.git@v1.0.0",
            "numpy >= 1.24.2",
            "torch >= 1.13.1",
            "torchvision >= 0.14.1",
            "torchinfo >= 1.8.0",
            "torchview >= 0.2.6",
            "matplotlib >= 3.8.1",
            "scikit-learn >= 1.3.2",
            "pandas >= 2.1.2",
            "h5py >= 3.8.0",
            "Pillow >= 9.4.0",
            "opencv-python >= 4.7.0.72",
            "graphviz >= 0.20.1",
            "mediapipe >=0.10.7",
            "imgaug >= 0.4.0",
            "tqdm >= 4.66.1"
      ], 
      python_requires=">=3.10.1",
     )