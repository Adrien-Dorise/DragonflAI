# DragonflAI

DragonflAI is a novel way to implement, train and test artificial intelligence models. Similar to Keras for Tensorflow, it is thought to be an additional layer to the PyTorch API. 

DragonflAI API is based on a main experiment class that encapsulates all tools needed to create an AI model. To use more complex models, the user can override key functions during the training, the test, or the visualisation of an AI model.

---

## Features

- **Experiment-centric API** — Encapsulate data, model, training, testing, and visualization in one class you control.  
- **Extensible hooks** — Override key methods for training/testing/visualization to fit simple to complex workflows.  
- **Examples included** — See the `examples/` folder to get started quickly.

---

## Installation

> DragonflAI is a Python package (MIT licensed).

**Prereqs**
- Python 3.8+ (recommend 3.10+)
- PyTorch (install per your CUDA/CPU setup from pytorch.org)

**From source (editable)**
```bash
git clone https://github.com/Adrien-Dorise/DragonflAI.git
cd DragonflAI
pip install -e .
```

**From GitHub**
```bash
pip install "git+https://github.com/Adrien-Dorise/DragonflAI.git"
```

---


## Quickstart

Create your own experiment by subclassing the core Experiment and overriding the pieces you need.  

```bash
# minimal_skeleton.py
from dragonflai import Experiment  # the main abstraction

class MyExperiment(Experiment):
    def build_datasets(self):
        # return train_loader, val_loader, test_loader
        ...

    def build_model(self):
        # return a torch.nn.Module
        ...

    def configure_optim(self, model):
        # return optimizer, (optional) scheduler
        ...

    def train_step(self, batch, model, optim):
        # compute loss, backprop, step; return metrics dict
        ...

    def eval_step(self, batch, model):
        # return metrics dict (e.g., loss/accuracy)
        ...

    def visualize(self, model):
        # (optional) custom plots or sanity checks
        ...

if __name__ == "__main__":
    exp = MyExperiment(config={"seed": 42, "epochs": 10})
    exp.fit()     # train
    exp.test()    # evaluate
    exp.visualize()
```
The framework centers around a main experiment class that encapsulates all tools needed to create an AI model, with override points during training, test, and visualization. See examples/ for concrete patterns.


**Examples**  
Check the examples/ directory for ready-to-run scripts that demonstrate:
 - dataset/model construction
 - custom train_step/eval_step
 - logging/visualization patterns

---

## Projet structure

```graphql
DragonflAI/
├─ dragonflai/           # package source (core Experiment API and helpers)
├─ examples/             # runnable examples & templates
├─ main.py               # entry point / script helpers
├─ setup.py              # package metadata
├─ LICENCE               # MIT
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
└─ README.md
```
---

## Credits

DragonflAI is an open-source project, supported by the Research team of LR Technologies.  
Main Contributors: Adrien Dorise, Edouard Villain, Julia Cohen, Boris Lenseigne
