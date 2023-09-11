from setuptools import setup, find_packages

setup(name='lr_ai',
      version='0.1.0',
      description='LR Technologies template for articial intelligence related projects',
      author='LR Technologies - Boris Lenseigne, Julia Cohen, Adrien Dorise, Edouard Villain',
      author_email='{blenseigne, jcohen, adorise, evillain}@lrtechnologies.fr',
      url='https://gitlab.com/lr-technologies2/LR_AI',
      py_modules=['lr_ai.experiment'],
      packages=['lr_ai.dataset', 
                'lr_ai.features', 
                'lr_ai.model', 
                'lr_ai.visualisation', 
                'lr_ai.model.neural_network_architectures',
                ]
     )