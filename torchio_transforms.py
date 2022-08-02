
from torchio import RandomElasticDeformation, RandomAffine, RandomFlip, RandomNoise, RandomMotion, RandomSpike, RandomBiasField, RandomBlur, RandomGamma
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import torch
import sys 
import torchio as tio
from torchio.transforms.augmentation.intensity.random_bias_field import RandomBiasField


class ElasticTransform(object):
    """
    A transformation to add random elastic deformation
    the distance between nibabel grid points is 3mm 

    Params: 
        replace: the value in the image to replace
        distr: the distribution of the gaussian to use for replacing values
    """
    def __init__(self, max_disp= 20, num_control_points=(8, 8, 6), locked_borders=2):
        self.max_disp = max_disp
        self.num_control_points = num_control_points
        self.transform = RandomElasticDeformation( 
            max_displacement=max_disp,
            num_control_points=num_control_points,
            locked_borders=locked_borders,
            )
        # interpolate 0 with mean values
    
    def __call__(self, sample):
        """
        img, label, pct = sample['img'], sample['label'], sample['90_pct']
        trans_img = self.transform(img)
        reproduce_transform = trans_img.get_composed_history()
        trans_label = reproduce_transform(label)

        return {'img': trans_img, 'label': trans_label, 'fn': sample['fn'], '90_pct': sample['90_pct']}
        """
        trans_img = self.transform(sample)
        
        return trans_img

class TorchioAffine(object):
    def __init__(self, scales=0, degrees=(11, 11, 11), translation=(10, 10, 5), default_pad_value='mean', isotropic=False, center='image', image_interpolation='linear'):
        '''
        Random affine transformations
        scales -- Tuple (a1,b1,a2,b2,a3,b3) defining the scaling ranges. The scaling values along each dimension are (s1,s2,s3), where si∼U(ai,bi). 
                If two values (a,b) are provided, then si∼U(a,b). If only one value x is provided, then si∼U(1−x,1+x). If three values (x1,x2,x3) are provided, then si∼U(1−xi,1+xi). 
                For example, using scales=(0.5, 0.5) will zoom out the image, making the objects inside look twice as small while preserving the physical size and position of the image bounds.
        degrees -- tuple (a1,b1,a2,b2,a3,b3) defining rotation in degrees. Rotation sampled as Theta_i ~ U(ai,bi), or Theta_i ~ U(-ai, ai) if only one value per axis provided.
        translation -- tuple (a1,b1,a2,b2,a3,b3) defining translation ranges in mm. translation t_i ~ U(ai,bi) or t_i ~ U(-a_i,a_i) if no bi provided.
        isotropic -- if True, scaling factor along all dimensions the same.
        center -- if 'image', rotation and scaling will be done about image center. If 'origin', will be performed about origin in real world coordinates.
        default_pad_value -- How to pad images near border after rotation
        image_interpolation -- linear, nearest, bspline
        '''
        
        self.transform = RandomAffine( 
            scales=scales,
            degrees=degrees,
            translation=translation,
            default_pad_value=default_pad_value,
            isotropic=isotropic,
            center=center,
            image_interpolation=image_interpolation,
            )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioFlip(object):
    def __init__(self, axes =(0,1,2), flip_probability=0.5):
        self.transform = RandomFlip(
            axes=axes,
            flip_probability=flip_probability,
        )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioNoise(object):
    '''
        mean – Mean μ of the Gaussian distribution from which the noise is sampled. If two values (a,b) are provided, then μ∼U(a,b). If only one value d is provided, μ∼U(−d,d)
        std – Standard deviation of the Gaussian distribution from which the noise is sampled. If two values (a,b) are provided, then σ∼U(a,b). If only one value d is provided, σ∼U(0,d).
    '''
    def __init__(self, mean=(0,0), std=(0.25,0.5)):
        self.transform = RandomNoise(
            mean=mean,
            std=std,
        )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioBlur(object):
    """
    Blur an image using a random Gaussian filter

    Param:
            std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.

    """
    def __init__(self, std =(0.25)):
        self.transform = RandomBlur(
            std=std
        )
    
    def __call__(self,sample):
        trans_img = self.transform(sample)
        
        return trans_img

class TorchioGamma(object):
    """
    Randomly change contrast of an image by raising its values to the power γ
    Params:
        log_gamma – Tuple (a,b) to compute the exponent γ=e^β, where β∼U(a,b). 
            If a single value d is provided, then β∼U(−d,d). Negative and positive values for this argument perform gamma compression and expansion, respectively. 
    """
    def __init__(self, log_gamma=(-0.3,0.3)):
        self.transform = RandomGamma(
            log_gamma=log_gamma
        )
    def __call__(self,sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioMotion(object):
    """
    Add a random motion artifact to MRI images.

    Params:
        degrees: Tuple :math:`(a, b)` defining the rotation range in degrees of
            the simulated movements. The rotation angles around each axis are
            (\theta_1, \theta_2, \theta_3),
            where `\theta_i ~ {U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\theta_i \sim \mathcal{U}(-d, d)`.
            Larger values generate more distorted images.
        translation: Tuple :math:`(a, b)` defining the translation in mm of
            the simulated movements. The translations along each axis are
            :math:`(t_1, t_2, t_3)`,
            where :math:`t_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`t` is provided,
            :math:`t_i \sim \mathcal{U}(-t, t)`.
            Larger values generate more distorted images.
        num_transforms: Number of simulated movements.
            Larger values generate more distorted images.
        image_interpolation: Interpolation
    """
    def __init__(self,degrees=10,translation=10,num_transforms=2,image_interpolation='linear'):
        self.transform = RandomMotion( 
            degrees=degrees,
            translation=translation,
            num_transforms=num_transforms,
            image_interpolation=image_interpolation,
            )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioSpike(object):
    """
    Add random spike artifacts
    
    Params:
        num_spikes: Number of spikes :n present in k-space.
            If a tuple :math:`(a, b)` is provided, then
            n ~\mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :d is provided,
            n ~ \mathcal{U}(0, d) \cap \mathbb{N}`.
            Larger values generate more distorted images.
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
            If a tuple :math:`(a, b)` is provided, then
            :math:`r \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`r \sim \mathcal{U}(-d, d)`.
            Larger values generate more distorted images.
    """
    def __init__(self, num_spikes=1, intensity=(1,3)):
        self.transform = RandomSpike(
            num_spikes=num_spikes,
            intensity=intensity
        )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img

class TorchioBiasField(object):
    """
    Adds bias field to the image

    Params:
        coefficients – Maximum magnitude n of polynomial coefficients. If a tuple (a,b) is specified, then n∼U(a,b)
        order – Order of the basis polynomial functions.
    """
    def __init__(self, coefficients=0.5,order=3):
        self.transform = RandomBiasField(
            coefficients=coefficients,
            order=order
        )
    
    def __call__(self, sample):
        trans_img = self.transform(sample)

        return trans_img


class TorchioIntensity(object):
    """
    A transformation to scale the intensity of an image

    Params:
        scale: the constant factor to multiply the image by
    """
    def __init__(self, scale,):
        self.scale = scale
    
    def __call__(self, sample):
        scale = random(self.scale[0], self.scale[1])
        img, label  = sample['img']['data'], sample['label']['data']
        img *= scale
        
        return tio.Subject({
                'img': tio.ScalarImage(tensor=img),
                'label': tio.LabelMap(tensor=label),
                'fn': sample['fn'],
                'fn_img_path': sample['fn_img_path'],
                'fn_label_path': sample['fn_label_path'],
                'fn_label': sample['fn_label'],
                '90_pct': sample['90_pct'],
                'low' : sample['low'],
                'affine' : sample['affine'],
                'label_affine': sample['label_affine'],
                'pad_amnt': sample['pad_amnt']
            })
            
class TorchioBrightness(object):
    """
    A transformation to scale the brightness of an image. This can scale either only the
    labeled area, or the entire image. 

    Params:
        scale: the constant factor to add to the image
        full_image: True if the entire image should be scaled, False otherwise
    """
    def __init__(self, scale, full_image=False):
        self.scale = scale
        self.full_image = full_image

    def __call__(self, sample):
        scale = random(self.scale[0], self.scale[1])
        img, label  = sample['img']['data'], sample['label']['data']
        if self.full_image:
            img += scale
        else:
            img[label==1] += scale
        
        return tio.Subject({
                'img': tio.ScalarImage(tensor=img),
                'label': tio.LabelMap(tensor=label),
                'fn': sample['fn'],
                'fn_img_path': sample['fn_img_path'],
                'fn_label_path': sample['fn_label_path'],
                'fn_label': sample['fn_label'],
                '90_pct': sample['90_pct'],
                'low' : sample['low'],
                'affine' : sample['affine'],
                'label_affine': sample['label_affine'],
                'pad_amnt': sample['pad_amnt']
            })

def random(lower, upper):
    return np.random.random() * (upper - lower) + lower


