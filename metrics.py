import numpy as np
import torch
from medpy.metric.binary import assd as ASSD
from medpy.metric.binary import hd as Hausdorff_Distance
from medpy.metric.binary import hd95 as Hausdorff_Distance_95

def metric_time_series(img_4D,metric="None",voxel_spacing=1):
    '''
    img_4D: 4D time series
    metric: "dice", "hausdorff", "hausdorff_95","assd"
    
    '''
    if len(np.shape(img_4D))!=4:
        return np.zeros(1)
    N = np.shape(img_4D)[0]
    stats = np.zeros((N-1))
    for i in range(1,N):
        img_1 = img_4D[i-1,:,:,:]
        img_2 = img_4D[i,:,:,:]
        if metric == "dice":
            stats[i-1] = dice(img_1,img_2)
        elif metric == "hausdorff":
            stats[i-1] = Hausdorff_Distance(img_1,img_2,voxelspacing=voxel_spacing)
        elif metric == "hausdorff_95":
            stats[i-1] = Hausdorff_Distance_95(img_1,img_2,voxelspacing=voxel_spacing)
        elif metric == "assd":
            stats[i-1] = ASSD(img_1,img_2,voxelspacing=voxel_spacing)

    return stats

def mean_BOLD_difference(img_ref,label_ref,label_pred, img_2=None):
    '''
    Computes the normalized mean BOLD difference between two images and their corresponding masks.
    Input:
            img_ref: MRI image: HxWxD
            label_ref: boolean matrix indicating location of placenta: HxWxD
            label_pred: boolean matrix of prediction: HxWxD
            img_2: second MRI image. By default assumes it is the same as img_ref
    Returns:
            |mean(BOLD_1)-mean(BOLD_2)|/mean(BOLD_1)
    '''
    assert np.array_equal(label_ref, label_ref.astype(bool)) 
    assert np.array_equal(label_pred, label_pred.astype(bool))
    
    BOLD1 = np.mean(img_ref,where=label_ref)
    BOLD2 =np.mean(img_ref,where=label_pred) if img_2 is None else np.mean(img_2,where=label_pred)
    
    return np.abs(BOLD1-BOLD2)/np.abs(BOLD1)

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched. From https://gist.github.com/JDWarner/6730747
    """
    im1 = im1 > 0.5
    im2 = im2 > 0.5
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def dice_tensor(im1, im2, empty_score=0.0, thresh=0.5):
    '''
    Computes the Dice coefficient between two tensors, im1 and im2.
    Inputs:
            im1, im2: HxWxD torch tensors
            empty_score: scalar, what to return if no binary image given as input
            thresh: threshold to convert image to binary. Default is 0.5
    Returns:
            dice_score: 1x1 torch scalar
    '''
    im1 = im1 > thresh
    im2 = im2 > thresh
    intersection = torch.flatten(im1) * torch.flatten(im2)
    im_sum = torch.sum(im1) + torch.sum(im2)
    if im_sum == 0:
        return empty_score
    return 2. * intersection.sum().item() / im_sum.item()