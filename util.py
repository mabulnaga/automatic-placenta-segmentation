import numpy as np
from numpy.core.fromnumeric import shape
from scipy.ndimage import zoom
import nibabel as nib
import os
import torch
from torch.nn.functional import avg_pool3d
import torchvision.transforms as transforms
import shutil
import sys
import pandas as pd 
import zipfile
import math
from torchio_transforms import *
from data_loader import DataLoader

def split_train_val(data_dir, img_dir, label_dir, transforms_string, data_split, batch_size=8, pad_factor=1, randomize_img_dataloader=False, aug_severity=0, store_images=True,test_only=False):
    """
    Gets a dataloader with the train, test, and validation sets. 

    Params:
    data_dir: the directory where the data is stored. 
    img_dir: subdirectory where the images are stored
    label_dir: subdirectory where the labels are stored. TODO: get rid of data_dir, and just use these 2
    transforms_string: a comma-delimited string of the transforms to apply. 
    data_split: dictionary to a list of files to load as train, test, validation data.
                previous just passed in global variable SUBJS.
    pad_factor: factor to make images divisible by. Default is 1, meaning no padding.
    use_raw_img_dataloader: Flag for using the data loader on the raw dataset.
    randomize_img_dataloader: True/False: whether to randomly sample one image per subject in the train data loder. (for train only)
    use_torchio: True/False: whether to use torchio format for data loader.
    aug_severity: integer (0 to 4) to determine the augmentation severity to use.
    store_images: bool. If True, stores data in RAM, else reads from disk
    test_only: bool. If True, creates only a test set.
    Return: a dataloader for the train, test, and validation sets. 
    """
    transform = build_transforms(transforms_string, aug_severity=aug_severity)
    return DataLoader(data_dir, data_split, img_dir, label_dir, \
             batch_size=batch_size, transform=transform, pad_factor = pad_factor, randomize_img_per_subj=randomize_img_dataloader, store_images=store_images,test_only=test_only)


def build_transforms(tr,aug_severity=0,use_teacher_learning=False):
    """
    Builds a list of transformations to use for training. 
    
    Params: 
    tr: a comma-delimited string of the transforms to use. 

    Return: a list of transforms. 
    """
    if tr == "none":
        return None
    transform = []
    aug_params = get_augmentation_params(severity=aug_severity)
    for s in tr.split(','):
        if s == 'intensity':
            transform.append(TorchioIntensity(aug_params['intensity'][0]))
        if s == 'affine':
            transform.append(TorchioAffine(degrees=aug_params['affine'][0], translation=aug_params['affine'][1]))
        if s == 'flip':
            transform.append(TorchioFlip(axes=aug_params['flip'][0], flip_probability=aug_params['flip'][1]))
        if s == 'noise':
            transform.append(TorchioNoise(mean=aug_params['noise'][0], std=aug_params['noise'][1]))
        if s == 'elastic':
            transform.append(ElasticTransform(max_disp=aug_params['elastic'][0], num_control_points=aug_params['elastic'][1])) # max_disp=20, num_control_points=(8,8,6) upper limit on these values
        if s == 'motion':
            transform.append(TorchioMotion(degrees=aug_params['motion'][0], translation=aug_params['motion'][1], num_transforms=aug_params['motion'][2])) #20, 20, 5
        if s == 'spike':
            transform.append(TorchioSpike(num_spikes=aug_params['spike'][0], intensity=aug_params['spike'][1])) #.25, 1.5
        if s == 'bias':
            transform.append(TorchioBiasField(coefficients=aug_params['bias'][0], order=aug_params['bias'][1])) #1, 3
        if s == 'blur':
            transform.append(TorchioBlur(std=aug_params['blur'][0])) #.75
        if s == 'gamma':
            transform.append(TorchioGamma(log_gamma=aug_params['gamma'][0])) #-.75,.75
        if s == 'brightness':
            transform.append(TorchioBrightness(scale=aug_params['brightness'][0],full_image=False)) 
    
    return transforms.Compose(transform)


def append_img_4d_dict(img_dictionary,subj_key,img):
    '''
    Appends an image to a dictionary. Builds a 4D tensor for time series. Assumes images are ordered.
    inputs:
            img_dictionary: dictionary
            subj_key: key, usually subject name (string)
            img: 3D tensor
    '''
    if subj_key in img_dictionary:
        # append the image
        imgs = img_dictionary[subj_key]
        if len(np.shape(imgs)) == 3:
            imgs = np.expand_dims(imgs,0)
        img = np.expand_dims(img,0)
        imgs = np.concatenate((imgs, img),axis=0)
        img_dictionary[subj_key] = imgs
    else:
        img_dictionary[subj_key] = img

    return img_dictionary


def read_subjs( fn):
    """
    reads a list of subjects out of a file. 

    Params:
    fn is the filename. 

    Return: a list of subjects. 
    """
    subjs = set()
    with open(fn, 'r') as f:
        for line in f:
            subjs.add(line.rstrip())
    return subjs

def extract_subj_names(data_path, files):
    '''
        Reads the list of subjects in a file.

        Inputs:
            data_path: file path for the files
            files: name of the files
        Return: a list of subjects.
    '''
    subjs = []
    for f in files:
        subjs.extend(read_subjs(os.path.join(data_path, f)))
    return subjs


def listdir_nohidden(path):
    '''
    Args:
        path: directory path

    Returns: list of files in numerical order. No directories or hidden files.
    Files without numbers are placed at the front of the list
    '''
    onlyFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyFiles = [f for f in onlyFiles if not f.startswith('.')]
    onlyFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
    return onlyFiles


def listdir_nohidden_sort_numerical(path, list_dir=False, sort_digit=True):
    '''
    Args:
        path: directory path (string)
        list_dir: true if wanting to return directories, otherwise files
        sort_digit: True if wanting to sort numerically, otherwise uses default sorting.

    Returns: list of files in numerical order. No directories or hidden files
    '''
    if list_dir:
        onlyFiles = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        onlyFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyFiles = [f for f in onlyFiles if not f.startswith('.')]
    if sort_digit:
        onlyFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    else:
        onlyFiles = sorted(onlyFiles)
    return onlyFiles


def save_code_dir(code_dir, save_dir):
    '''
        Function to save the directory with all the code during experiment run time.

    '''
    root_path = os.path.dirname(code_dir)
    zipf = zipfile.ZipFile(os.path.join(save_dir,'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.join('code/',os.path.relpath(file_path, root_path))
            zipf.write(file_path, arcname=relative_path) 
            #zipf.write(os.path.join(root, file))
    zipf.close()


def save_checkpt(state, checkpoint_dir, best_model_name='model_best', is_best=False):
    '''
    Save a model. Saves either a checkpoint, or the best model, or both. For the best
    model, simply copies over.
    Inputs:
            state: dictionary with the model states to save.
            checkpoint_dir: directory where to save the models
            teacher: boolean, whether we are saving the teacher model or not
            is_best: Boolean, whether to save the best model. If True, only saves the best. otherwise, saves a checkpoint.
            best_model_name: name for the best model
        
    '''
    save_name = 'model_checkpoint.pt'
    f_path = os.path.join(checkpoint_dir , save_name)
    if not is_best:
        torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir, best_model_name + '.pt')
        shutil.copyfile(f_path, best_fpath)


def load_checkpt(checkpoint_fpath, model, optimizer=None):
    '''
    Loads a model checkpoint. Uses this to load the state.
    Inputs:
            checkpoint_fpath: full path of the checkpoint
            model: torch model to load state to
            optimizer: torch optimizer to load state to.
            dices: running list of dice scores from previous experiments.
    Returns:
            model, optimizer, epoch, dices
    Code adopted from https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_fpath)
    else:
        checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['dices']


def rand_bool(p_true):
    """
    generates a boolean that is true with given probability. 

    Params:
    p_true: probability of the boolean being true. 

    Return: a boolean. 
    """
    if np.random.random() < p_true:
        return True
    return False


def make_dir_if_not_exist(dir_path): 
    """
    makes a directory at the given path if it doesn't exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_augmentation_params(severity=0):
    '''
    Gets a set of augmentation parameters to use torchio. 
    Input: severity. string: options are 0 to 4, increasing in severity. 2 is likely the most usable one.
    Returns: aug_params: dictionary. Each field refers to the transformation function.
        Fields:
            intensity: [(min,max)]
            affine: [(degree),(translation)]
            flip: [(axes),(flip_probability)]
            noise: [(mean), (std)]
            elastic: [(max_disp),(num_control_points)]
            motion: [(degrees),(translation),(num_transforms)]
            spike: [(num_spikes),(intensity)]
            bias: [(coefficients),(order)]
            blur: [(std)]
            gamma: [(log_gamma)]
    '''

    aug_params = {}

    if severity == 0:
        aug_params = {
            'intensity': [[0.9,1.1]],
            'affine': [(11,11,11), (5,5,2)],
            'flip': [(0,1,2),(0.1)],
            'noise': [(0.0,0.0),(0.05,0.05)],
            'elastic': [(5),(5)],
            'motion': [(5),(5),(1)],
            'spike': [(1),(0.1,0.25)],
            'bias': [(0.25),(1)],
            'blur': [(0.25)],
            'gamma': [(-0.1,0.1)],
            'brightness':[(-0.05,0.05)]
        }
    elif severity == 1:
        aug_params = {
            'intensity': [[0.8,1.25]],
            'affine': [(22,22,22), (10,10,5)],
            'flip': [(0,1,2),(0.25)],
            'noise': [(0.0,0.0),(0.25)],
            'elastic': [(10),(5)],
            'motion': [(5),(5),(2)],
            'spike': [(1),(0.1,0.5)],
            'bias': [(0.5),(1)],
            'blur': [(0.25)],
            'gamma': [(-0.25,0.25)],
            'brightness':[(-0.15,0.15)]
        }
    
    elif severity == 2:
        aug_params = {
            'intensity': [[0.8,1.25]],
            'affine': [(45,45,45), (20,20,10)],
            'flip': [(0,1,2),(0.5)],
            'noise': [(0.0,0.0),(0.5)],
            'elastic': [(10),(5)],
            'motion': [(10),(10),(2)],
            'spike': [(1),(0.1,0.75)],
            'bias': [(0.5),(2)],
            'blur': [(0.25)],
            'gamma': [(-0.25,0.25)],
            'brightness':[(-0.20,0.20)]
        }
    elif severity == 3:
        aug_params = {
            'intensity': [[0.75,1.33]],
            'affine': [(60,60,60), (25,25,12)],
            'flip': [(0,1,2),(0.5)],
            'noise': [(0.0,0.0),(0.5)],
            'elastic': [(12),(6)],
            'motion': [(15),(15),(3)],
            'spike': [(1),(0.1,1)],
            'bias': [(0.5),(3)],
            'blur': [(0.25)],
            'gamma': [(-0.5,0.5)],
            'brightness':[(-0.25,0.25)]
        }
    elif severity >= 4:
        aug_params = {
            'intensity': [[0.5,1.5]],
            'affine': [(90,90,90), (30,30,15)],
            'flip': [(0,1,2),(0.5)],
            'noise': [(0.0,0.0),(0.5)],
            'elastic': [(16),(6)],
            'motion': [(20),(20),(4)],
            'spike': [(1),(0.1,1.25)],
            'bias': [(0.75),(3)],
            'blur': [(0.5)],
            'gamma': [(-0.75,0.75)],
            'brightness':[(-0.35,0.35)]
        }

    return aug_params


def get_per_patient_stats(img_names, metric_scores):
    '''
    Returns a dictionary with per-patient stats. Assumes the naming convention MAP-C303, etc...
    Inputs:
        img_names: list of image names
        metric_scores: list of corresponding metric scores
    Returns:
        dictionary with stats per patient
    '''
    name_subjs = [n[0:8] for n in img_names]
    setn = set(name_subjs)
    name_subjs = list(setn)
    metric_patient = dict()
    for name in name_subjs:
        inds = [ name in name_subjs for name_subjs in img_names]
        inds = [i for i, x in enumerate(inds) if x]
        metric_ct = 0.
        ct = 0
        for i in inds:
            metric_ct = metric_ct + metric_scores[i]
            ct = ct + 1
        metric_patient[name] = metric_ct/ct
    
    return metric_patient


def crop_range(img, shape, dim):
    """
    Computes the range of indices in the resized images for a given dimension. 

    Params:
    img: the original image
    shape: the resized shape
    dim: the dimension to compute

    Return: a list containing the range of indices in the resized image for a given dimension
    where the original image will be placed. 
    """
    offset = -(img.shape[dim] - shape[dim]) // 2
    if offset >= 0:
        return list(range(offset, offset + img.shape[dim]))
    else:
        return list(range(0, shape[dim]))


def get_crop_indices(img, shape):
    """
    Computes the range of indices in the resized images where the old image will be placed. 

    Params:
    img: the original image
    shape: the resized shape

    Return: a list containing the range of indices in the resized image
    where the original image will be placed. 
    """
    return crop_range(img, shape, 0), crop_range(img, shape, 1), crop_range(img, shape, 2)


def img_range(img, shape, dim):
    """
    Computes the range of indices in the original image that will be placed in the new image,
    for the given dimension. 

    Params: 
    img: the original image
    shape: the shape of the new image
    dim: the dimension to compute 

    Return: a list of indices. 
    """
    if img.shape[dim] <= shape[dim]:
        return list(range(0, img.shape[dim]))
    else:
        offset = (img.shape[dim] - shape[dim]) // 2
        return list(range(offset, offset + shape[dim]))


def get_img_indices(img, shape):
    """
    Computes the range of indices in the original image that will be placed in the new image.

    Params: 
    img: the original image
    shape: the shape of the new image

    Return: a list of indices. 
    """
    return img_range(img, shape, 0), img_range(img, shape, 1), img_range(img, shape, 2)

def unpad_img(img,pad_amnt):
    '''
    unpads an image.
    inputs: img: HxWxD numpy array
            pad_amnt: 1x6 numpy array with pad amounts
    returns:
            unpadded image
    '''
    sz = np.shape(img)
    return img[pad_amnt[0]:sz[0]-pad_amnt[1],pad_amnt[2]:sz[1]-pad_amnt[3],pad_amnt[4]:sz[2]-pad_amnt[5]]

def crop_or_pad(img, shape, distr, labels=False):
    """
    crops or pads an image to the new shape, filling padding with noise. 

    Params:
    img: the original image
    shape: the desired size
    distr: the distribution for the padding noise
    labels: bool, True if the image is a label

    Return: the resized image
    """
    x, y, z = get_crop_indices(img, shape)
    new_img = np.zeros(shape)
    if not labels:
        new_img = np.random.normal(loc=distr[0], scale=distr[1], size=shape)
        new_img[new_img < 0] = 0
    x_crop, y_crop, z_crop = get_crop_indices(img, shape)
    x_img, y_img, z_img = get_img_indices(img, shape)
    new_img[np.ix_(x_crop, y_crop, z_crop)] = img[np.ix_(x_img, y_img, z_img)]
    return new_img


def get_pos_weight(data):
    """
    Gets the weight that should be used for positive labels in weighted cross entropy loss. 

    Params:
    data: the training dataset

    Return: the weight that should be used for positive labels
    """
    zeros = 0
    ones = 0
    for batch in data:
        label = batch['label']['data']
        size = 1
        for dim in label.shape:
            size *= dim
        ones += torch.sum(label)
        zeros += size - (torch.sum(label))
    # print(zeros/ones)
    return zeros / ones


def use_boundary_kernel(segmentation, kernel_size=7):
    """
    runs kernel on 3d segmentation and then returns 0's for nonboundary, 
    1's for boundary 

    segmentation: 3d label/prediction of placenta/nonplacenta 
    kernel_size: used to determine boundary width, must be 3d odd tuple
    """
    epsilon = sys.float_info.epsilon   
    segmentation = torch.from_numpy(segmentation)
    #reshaping segmentation 
    new_shape = (1,) + segmentation.shape
    reshaped_seg = torch.reshape(segmentation, new_shape)
    #calculate appropriate padding based on kernel size
    padding = int((kernel_size-1)/2)
    
    #use avg pooling to determine boundaries
    seg_avg_pool =  avg_pool3d(reshaped_seg, 
                                kernel_size= kernel_size, 
                                stride =(1,1,1), 
                                padding = padding, 
                                count_include_pad = False) 
    reshaped_boundaries = (seg_avg_pool > epsilon) & (seg_avg_pool < 1-epsilon)
    reshaped_boundaries_int = reshaped_boundaries.int() 
    final_boundaries = torch.reshape(reshaped_boundaries_int, segmentation.shape)
    return final_boundaries.numpy().astype(float)


def load_img(path):
    """
    loads an image. 

    Params:
    path: the path to the file that should be loaded
    """
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img.astype(float)


def save_img(data, path, fn, affine=np.eye(4)):
    """
    saves an image as a nifti. 

    Params:
    data: the image to be saved. 
    path: the path to the file where the image should be saved. 
    fn: the filename that the image should be saved at. 
    affine: the affine header of the image
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, os.path.join(path, fn))


def unnormalize_img(img, low, high):
    '''
    Unnormalize an image
    '''
    img =  img *(high-low) + low
    return img


def best_metrics(csvfile, output_dir):
    """
    get best metrics from an existing csv file 
    """
    MAX_METRICS = ['val_dice','train_dice']
    MIN_METRICS = ['val_loss', 'train_loss']
    #get the rows with the best of each metric, save to csv
    df = pd.read_csv(csvfile)

    best_metrics = []
    for metric in MAX_METRICS:
        ind = df.index[df[metric] == df[metric].max()]
        s = df.iloc[ind,:]
        if len(s) > 1:
            s = s.iloc[0,:]
        best_metrics.append(s)
    for metric in MIN_METRICS:
        ind = df.index[df[metric] == df[metric].min()]
        s = df.iloc[ind,:]        
        if len(s) > 1:
            s = s.iloc[0,:]
        best_metrics.append(s)
    best_of_each_df = pd.concat(best_metrics)
    best_of_each_df["optimized_metric"] = MAX_METRICS+MIN_METRICS
    best_of_each_df.to_csv(os.path.join(output_dir,'best_metrics.csv'), index=False)


def normalize_img(img, pct=99): 
    """
    normalizes an image to [0,1] by clipping values above the percentile
    and dividing by the percentile 
    """
    scaled = img/(np.percentile(img, pct)-np.min(img))*255
    return np.clip(scaled, a_min=0, a_max=255)


def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]

def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]
