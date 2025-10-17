# Re-Resformer

**Hybridizing ResNet-50 and Vision Transformer for laser-speckle based milling surface roughness measurement**  
Repository for the paper **“Online measurement of milling surface roughness via laser speckle image using a hybrid ResNet-Transformer network”**  
Author / GitHub: `libo-yueling` — `https://github.com/libo-yueling/Re-Resformer`

---

## Overview
This repository contains the implementation, configs and demos for **Re-Resformer**, implemented following MMClassification / mm-style conventions. It includes model code, dataset utilities, training & evaluation scripts and `mmpretrain`-style helpers bundled in `mmpretrain/`.

Primary goals of this README: tell other researchers how to reproduce experiments (environment, commands, configs), how to run training/evaluation/inference, and how to release model weights.

---

# Environment (exact training configuration)
The training procedures were conducted on a workstation configured with **PyTorch 2.0.0** and **Python 3.8**, operating under **Ubuntu 20.04**.  
The computational environment comprised a single **NVIDIA RTX 3090 (24 GB VRAM)**, **14 virtual CPUs** (Intel Xeon Platinum 8362 @ 2.80 GHz), and **45 GB RAM**. Storage resources included a **30 GB system disk** and a **50 GB data disk**.

**Framework:** MMClassification-style code and utilities (some tools adapted under `mmpretrain/`).

---

# Repo layout (important paths)
```bash
Re-Resformer/
├── configs/
│ │── re-resformer/
│ │   └── Re-Resformer.py # model & training configs used in experiments
├── dataset/ # dataset loaders / preprocessing / split scripts
├── demo/ # inference demo scripts
├── mmpretrain/ # mm-style modules (apis, models, datasets, engine, etc.)
├── requirements/ # optional per-platform pinned reqs
├── tools/ # training/testing entrypoints (mm-style)
├── tests
├── docker
├── docs
├── projects
├── resources
├── README.md
└── CITATION.cff, LICENSE, .gitignore
```
# Dataset description
The test set data is confidential, so this project only uploads the training set and validation set, 
and the data in the test set file in the project is data of the validation set.

# Dataset structure
```bash
Re-Resformer/dataset/
├── 2Q235/
│   │── train/
│   │   │──0.648
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ...
│   │   │── 1.258
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ...
│   │── val/
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ...
│   │   │── 1.258
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ...
│   │── test/
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ......
│   │   │── 1.258
│   │   │  └──image0.bmp
│   │   │  └──image1.bmp
│   │   │  ...
│   │── train.txt
│   │── val.txt
│   │── test.txt
├── label_make.py
├── meta/
│   └── train.txt
│   └── val.txt
│   └── test.txt
```
---
# Dataset label
We have a python file (label_make.py) for label generation in the dataset folder, and after the dataset is set up according to the data structure you can directly run label_make.py to get the text file of the data label:
```bash
cd path to /Re-Resformer/dataset
python label_make.py
```
you will get three txt (train.txt,val.txt,test.txt)

# Quick start (reproducible setup)

> All commands assume you run them from the repository root (`Re-Resformer`).

1. Create & activate conda env
```bash
conda create -n reresformer python=3.8 -y
conda activate reresformer
```
---

2. Install Python dependencies
open the document named 'requirements' and install all dependencies
install example (requirements/docs.txt):
```bash
pip install docutils==0.18.1
...
```
3. requirements.txt (If an upgrade is needed, please do so as required.):
```bash
docutils==0.18.1
modelindex
myst-parser
git+https://github.com/mzr1996/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
sphinx==6.1.3
sphinx-copybutton
sphinx-notfound-page
sphinx-tabs
sphinxcontrib-jquery
tabulate
mmcv>=2.0.0,<2.4.0
mmengine>=0.8.3,<1.0.0
pycocotools
transformers>=4.28.0
albumentations>=0.3.2 --no-binary qudida,albumentations    # For Albumentations data transform
grad-cam >= 1.3.7,<1.5.0   # For CAM visualization
requests            # For torchserve
scikit-learn        # For t-SNE visualization and unit tests.
--extra-index-url https://download.pytorch.org/whl/cpu
mmcv-lite>=2.0.0rc4
pycocotools
torch
torchvision
einops
importlib-metadata
mat4py
matplotlib
numpy
rich
coverage
interrogate
pytest
```
   
---

3. Install `mmcv-full` matching your CUDA and PyTorch:
- Important: `mmcv-full` normally needs a prebuilt wheel for your CUDA / torch combination. Follow the official mmcv docs to find the correct wheel URL and install with pip install <wheel-url>.
Example placeholder (replace with correct wheel):
```bash
pip install mmcv-full==<version> -f https://download.openmmlab.com/mmcv/dist/<...>
```

---
4. Create & activate conda envAdd project root to PYTHONPATH (so mmpretrain imports work)
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
# add the above line to ~/.bashrc or to scripts/train.sh for persistence
```
# Data

Place your dataset where configs/_re-resformer/Re-Resformer expects it (see data section in the config). Example dataset root:
```bash
Re-Resformer/dataset/speckle_image/
```
Dataset loader: dataset/ contains loaders and preprocessing utilities. Make sure the train/val/test paths in the config point to your prepared splits.

# Configs

Experiment configs live in:
```bash
/root/autodl-tmp/Re-Resformer/configs/re-resformer/Re-Resformer.py
```
Typical files:
- train_config.py or config.yaml — training hyperparameters, optimizer, lr schedule, seeds
- dataset_config.py — dataset & pipeline
- model_config.py — model architecture / initialization
When reporting experiments, include the exact config file name and work_dir used.

# Run: training / evaluation / inference Single-GPU training (example)

open tools/train.py and set parameterparameters:
```bash
#!/usr/bin/env bash
CONFIG=$1
WORK_DIR=${2:-work_dirs/re-resformer}
export PYTHONPATH=$(pwd):$PYTHONPATH

#script name
/root/autodl-tmp/Re-Resformer/tools/train.py

#script parameter
/root/autodl-tmp/Re-Resformer/configs/re-resformer/Re-Resformer.py

#working directory
/root/autodl-tmp/Re-Resformer/tools
```
Run:
```bash
python train.py
```
# Evaluation / test

Create scripts/eval.sh:
```bash
#!/usr/bin/env bash
CONFIG=$1
CKPT=$2
export PYTHONPATH=$(pwd):$PYTHONPATH

#script name
/root/autodl-tmp/Re-Resformer/tools/test.py

#script parameter
/root/autodl-tmp/Re-Resformer/configs/re-resformer/Re-Resformer.py

#working directory
/root/autodl-tmp/Re-Resformer/tools
```
Run:
```bash
python test.py  /root/autodl-tmp/Re-Resformerconfigs/re-resformer/Re-Resformer/config.py work_dirs/re-resformer/latest.pth
```
# Evaluation / test

Example:
```bash
python demo/inference_demo.py \
  --config configs/_re-resformer/Re-Resformer/config.py \
  --checkpoint work_dirs/re-resformer/latest.pth \
  --input examples/image1.bmp \
  --out results/out.png
```
# Reproducibility checklist (include this in experiments)
To enable reproducibility, do all of the following and record them in the experiment log:

1. Environment: Python, PyTorch, mmcv-full, mmcls, CUDA driver versions (exact).
Example: Python 3.8, PyTorch 2.0.0, mmcv-full x.y.z (CUDA 11.7), mmcls vX.Y.

2. Hardware: list GPU, CPU, RAM, disk (we used RTX 3090 (24GB) + 14 vCPU Intel Xeon Platinum 8362 + 45GB RAM).

3. Exact command line used to start training (copy & paste).

4. Config file: attach the exact config file used (commit it to repo or include in work_dir).

5. Random seed: set and record (example: seed=42).

6. Checkpoint & logs: upload final checkpoint(s) and TensorBoard logs (or provide link).

7. Notes on determinism:
```bash
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
# Model weights & release (recommended)

- Do not push large *.pth files to git.

- Recommended options:

- GitHub Releases — upload model files to Releases (ok for moderate sizes).

- Zenodo — link Release to Zenodo to mint DOI for reproducible artifact citation.

- Google Drive / Dropbox — provide direct download links + MD5 checksums in MODEL_ZOO.md.

- Example MODEL_ZOO.md entry:
```bash
| Model | Checkpoint | Download | MD5 | Notes |
|-------|-----------:|---------:|-----:|------|
| Re-Resformer (best) | re_resformer_best.pth | https://link.to/weights | abcdef123456 | trained on dataset X, config: configs/_re-resformer/Re-Resformer/config.py |
```

# CITATION
If you use this code, please cite the paper:
```bash
@article{your2025reresformer,
  title={Online measurement of milling surface roughness via laser speckle image using a hybrid ResNet-Transformer network},
  author={...}, # replace with full author list in final citation
  year={2025},
  note={Code: https://github.com/libo-yueling/Re-Resformer}
}
```
Add CITATION.cff in the repo root (we provide one in the repo).

# Recommended repo files (if not present)
- README.md (this file)

- LICENSE (MIT recommended)

- CITATION.cff

- requirements.txt and optional requirements/<cuda-version>.txt

- tools/train.py, tools/test.py

- MODEL_ZOO.md, REPRODUCIBILITY.md

- .github/workflows/ci.yml (basic install & lint)

# Troubleshooting & tips
- mmcv/mmcls import errors: ensure mmcv-full was installed with the correct prebuilt wheel and PYTHONPATH contains project root.

- OOM on RTX 3090: reduce batch_size or apply gradient accumulation.

- Inconsistent config keys: ensure your installed mmcls / mmpretrain version matches the config syntax used in configs/.

- Large files: git-ignore checkpoints/, data/ in .gitignore. Use Releases/Zenodo for distribution.

# Example quick commands (copy-paste)
```bash
# prepare env
conda create -n reresformer python=3.8 -y
conda activate reresformer
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH

# train (single GPU)
bash tools/train.py ../configs/re-resformer/Re-Resformer.py

# if train failed ,use absolute address：
bash tools/train.py /root/autodl-tmp/Re-Resformer/configs/re-resformer/Re-Resformer.py
# test
bash tools/test.py ../configs/re-resformer/Re-Resformer.py ../tools/work_dirs/Re-Resformer/epoch_100.pth

# demo
python demo/inference_demo.py --config configs/re-resformer/Re-Resformer/config.py --checkpoint work_dirs/re-resformer/latest.pth --input examples/img1.jpg

# demo example:

/root/autodl-tmp/Re-Resformer/dataset/2Q235/test/6.3/2024-07-25_08_46_53_892.bmp ../configs/re-resformer/Re-Resformer.py --checkpoint ../tools/work_dirs/Re-Resformer/epoch_80.pth
# run
python demo/image_demo.py /root/autodl-tmp/Re-Resformer/demo/2.989/2025-03-17_11_30_59_694.bmp ../configs/re-resformer/Re-Resformer.py --checkpoint ../tools/work_dirs/Re-Resformer/epoch_80.pth
```


# Contact & contribution
- Contributions via issues and pull requests are welcome.

- Author / contact: libo-yueling (add 2023010091@ybu.edu.cn).
