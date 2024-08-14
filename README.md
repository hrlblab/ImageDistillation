# ImageDistillation

This is the code repository for "Dataset Distillation in Medical Imaging: A Feasibility Study".

## Dataset
The dataset used here is from MedMNIST, which provides with medical images of 64x64, 128x128, and 224x224 for 2D, and 64x64x64 for 3D. Source: https://medmnist.com/.

## Getting Started
First, download our repo:
```
  git clone https://github.com/hrlblab/ImageDistillation.git
```
Then create conda environment with our yaml file.
```
conda env create -f environment.yaml
conda activate distillation
```

## Distillation on mtt
This is modified from: https://github.com/GeorgeCazenavette/mtt-distillation
