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

## Distillation with MTT
This is modified from: https://github.com/GeorgeCazenavette/mtt-distillation

Before doing any distillation, you'll need to generate some expert trajectories using buffer.py

The following command will train 100 ConvNet models on PathMNIST(28x28) with ZCA whitening for 50 epochs each:
```
cd MTT
python buffer.py --dataset=PathMNIST --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

Note that experts need only be trained once and can be re-used for multiple distillation experiments.

The following command will then use the buffers we just generated to distill PathMNIST(28x28) down to just 1 image per class:
```
python distill.py --dataset=PathMNIST --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-06 --lr_teacher=0.01 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

Noticed that the lr_lr might need to be adjusted to make sure the loss will not be negative.

## Distillation with DC
This is modified from: https://github.com/VICO-UoE/DatasetCondensation

```
cd DC
python main.py  --dataset PathMNIST  --model ConvNet  --ipc 10
```
