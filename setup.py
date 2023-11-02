from setuptools import setup, find_packages

setup(name="dragonflai",
      version="0.8.0",
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
            "h5py >= 3.8.0",
            "Pillow >= 9.4.0",
            "opencv-python >=4.7.0.72"
      ], 
      python_requires=">=3.10.1",
     )