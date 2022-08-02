# Automatic Placenta Segmentation in BOLD MRI
A neural network model for automatic segmentation of the placenta embedded in whole-uterus Blood Oxygen Level Dependent (BOLD) MRI. The model was trained on a diverse dataset containing subjects with singleton and twin pregnancies, a broad range of gestational ages, and pregnancy conditions including healthy controls, fetal growth restriction, and high BMI. This repo contains our trained model and scripts for inference, training, and evaluation. The trained model can predict placental segmentations on individual MRI and time series data. Training and evaluation scripts can be used to train a new model from scratch, or one initialized with our trained weights. This repo is based on the paper **TODO**

![alt text](https://github.com/mabulnaga/placenta-segmentation-release/blob/master/teaser_github.png)
*Predicted segmentations in our test set (red) and ground truth segmentations (yellow).*

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python
- Python Packages:
    - [PyTorch 1.9.1](https://pytorch.org/)
    - [MedPy 0.3.0](https://pypi.org/project/MedPy/)
    - [MONAI 0.9.0](https://monai.io/)
    - [TorchIO 0.18.83](https://torchio.readthedocs.io/)
    - [NiBabel 4.0.1](https://nipy.org/nibabel/)
- (Optional) GPU with CUDA 11.3 compute capability

## Installation
All required libraries are supplied in the [environment_cpu.yml](https://github.com/mabulnaga/placenta-segmentation-release/blob/master/environment_cpu.yml) or [environment_gpu.yml](https://github.com/mabulnaga/placenta-segmentation-release/blob/master/environment_gpu.yml) files. 
First install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and select the version appropriate for your system. After installation, create a conda environment that contains the required packages:
``` conda env create -f environment_gpu.yml ```. This will create the environment ```placenta-segmentation-gpu```. If you do not have a CUDA capable GPU, then use the ```environment_cpu.yml``` file.

## Data Organization ##
The scripts assume the data is organized in a specific way. There is one parent directory with **N** subdirectories, each containing the images for one subject.
Within a subject's directory, the MRI images must be in a subdirectory named ``image``, and the groundtruth segmentation labels in a subdirectory named ``segmentation``:
```
data_path/
    ├── Subject_1
        ├── image/
            ├── img_1_name.nii
            ├── img_2_name.nii
        ├── segmentation/
            ├── segmentation_1_name.nii
            ├── segmentation_2_name.nii
    ├── Subject_2
        ├── image/
            ├── img_1_name.nii
        ├── segmentation/
            ├── segmentation_1_name.nii
    ├── ...
 ```   
 Each subject can have a different number of image and label pairs. There is no restriction on the naming of each individual MRI image, and the script can use ``.nii`` and ``.nii.gz`` extensions. 
 
  For evaluating on BOLD time series without ground truth segmentations, you do not need to create the ``segmentation`` subdirectory.
 
 The data is padded to be divisible by 16 to pass through the UNet. If your dataset has images with different sizes, you can use the function ``preprocess.preprocess`` to make them all the same shape. Otherwise, you can only use a batch size of 1.
 
 :warning: The script uses numerical ordering 
 of the file names to associate image and segmentation pairs. Ensure the image and segmentation names have the same ordering, or are the same. :warning:
  
 :warning: **Currently, we can only process 3D nifti files.** :warning:

## Usage
There are three main scripts:
- ```train_placenta.py```: used to train the model
- ```run_model.py```: used to run inference on the trained model and compute statistics with ground truth segmentations.
- ```run_model_timeseries.py```: used to run inference on time series data.

First, activate the conda environment

``` conda activate placenta-segmentation-cpu``` or ``` conda activate placenta-segmentation-gpu```

## Time Series Segmentation Prediction

To run inference on BOLD MRI time series, run:

``` python run_model_timeseries.py --data_dir 'PATH_TO_DATA' --save_dir 'PATH_TO_OUTPUT' ```

You specify PATH_TO_DATA as the location where your data is stored, and PATH_TO_OUTPUT to be where you want the output predicted segmentations to be saved to. **Note that you do not need ground truth segmentations**

## Training
The script ```train_placenta.py``` is used to train a model from scratch, or from one initialized with our model's weights. The script will create a dataset split into train, validation, and test.

#### Data splits:
The script will create train, validation, and test set splits. The script assumes that each folder within the data directory is a separate subject, and will randomly split 65%, 15%, and 20% of the folders into train, validation, and test, respectively. Ideally, the user will create their own dataset folds, taking into account subject demographic information including gestational age, number of fetuses, and pregnancy conditions. The script will save these folds into ```model-folds.npy```.

#### Code parameters and flags
- --data_path: full path where the data is located
- --save_path: full path where to save the trained model and output files
- --epochs: number of epochs to run for (default: 4000)
- --lr: learning rate (default: 1e-3)
- --load_model: flag to initialize using our trained model
- --transform: string of comma separated values indicating what data augmentations to run. For example, ```--transform affine,flip,intensity,noise```
- --use_weighted_bce: flag to use a multiplicative weight on the positive examples in the loss function.
- --batch_size: batch size to use
- --boundary_kernel: average pooling kernel size used in estimating the placenta boundary
- --inner_boundary_weight: multiplicative weight for the voxels inside the placenta boundary (Eq. 1 in paper)
- --outer_boundary_weight: multiplicative weight for the voxels outside the placenta boundary (Eq. 1 in paper)
- --randomize_image_loader: flag to randomly subsample one of N_l images per subject during training. Use if dataset contains more than one labeled example per subject.
- --aug_severity: augmentation severity 
- --use_Dice_loss: flag to use additive Dice loss. See [here](https://docs.monai.io/en/stable/losses.html#diceloss) for details.
- --dice_loss_weight: multiplicative scaling parameter for the Dice loss
- --use_Focal_loss: flag to use Focal loss instead of Cross Entropy. See [here](https://docs.monai.io/en/stable/losses.html#focalloss) for details.
- --focal_loss_weight: multicative scaling parameter for the Focal loss
- --focal_loss_gamma: $\gamma$ exponential parameter in Focal loss
- --focal_loss_alpha: $\alpha$ parameter weighting the classes in Focal loss

#### Running 
To train as in the paper

``` python train_placenta.py --data_path 'PATH_TO_DATA' --save_path 'PATH_TO_OUTPUT' --epochs 4000 --lr 1e-4 --transform 'affine,flip,intensity,noise,gamma,elastic' --use_weighted_bce --batch_size 4 --boundary_kernel 11 --inner_boundary_weight 1 --outer_boundary_weight 40 --randomize_image_loader --aug_severity 1```

## Evaluating the Trained Model
The script ``run_model.py`` is used to evaluate model performance by computing the Dice score, the Hausdorff distance (HD), HD95, and the mean BOLD difference between predicted segmentations and ground truth. The script requires ground truth segmentation labels for each subject, and can be used on any set of model weights. This script can be run post-training to compute stats for the train, validation, and test sets, or to compute stats on your entire dataset using our trained model.

To run after training a new model,

``` python run_model.py --data_path 'PATH_TO_DATA' --save_path 'PATH_TO_OUTPUT' --eval_existing_folds --model_name 'model_best' ```


To run on our trained model and evaluate on all data in your dataset,

``` python run_model.py --data_path 'PATH_TO_DATA --save_path 'PATH_TO_OUTPUT' --model_name 'model_PIPPI' ```



## Development
Please contact Mazdak Abulnaga, abulnaga@mit.edu.

## Citing and Paper
If you use this method or some parts of the code, please consider citing our paper: 
