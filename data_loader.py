import numpy as np
import nibabel as nib
import torch.utils.data as data
import torch
import os
import os.path
import util
from torchvision import transforms
import torchio as tio
import multiprocessing
from os.path import exists

#data loader
num_workers = 8
print('NUM WORKERS: '+str(num_workers))
SEGMENTATION_KEY = "" #optional keys used to identify specific files within a folder
IMAGE_KEY = ""


class DataLoader():
    def __init__(self, data_dir, subjs, img_dir, label_dir, batch_size=1, randomize_img_per_subj=False, transform=None, pad_factor=1, test_only=False, store_images=True):
        """
        A Dataloader for the train, test, and val sets, on the raw images.
        Each subject has two directories within, segmentation and volume.
        Ex: MAP-B304/segmentation/, MAP-B304/volume. Within these subdirectories, the images live. The data loader will randomly pick from a (segmentation, volume)
        pair within each subdirectory.

        Params:
        data_dir: full path to dataset (str). 
        subjs: dictionary containing lists for train/test/val data (dict)
        img_dir: subdirectory name that contains source images (str)
        label_dir: subdirectory name that contains ground truth segmentations (str)
        batch_size: batch size used (int)
        randomize_img_per_subj: Flag for the data loader, whether to treat each subject as an input data point, and randomly sample an image per subject. (bool)
        transform: string of data augmentation transforms to apply (str)
        pad_factor: a factor to make images divisible by. 1 by default, which leads to no padding. (int)
        test_only: create only a test set (bool)
        store_images: whether to store the images to memory or load from disk on each call (bool)
        """
        self.data_dir = data_dir
        self.subjs = subjs
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs_map = {}
        self.labels_map = {}

        self.imgs_map, self.labels_map = self.get_map_directories(self.data_dir,self.img_dir,self.label_dir,img_key=IMAGE_KEY, segmentation_key=SEGMENTATION_KEY)
        if not test_only:
            self.train = self.get_loader(subjs['train'], transform=transform, batch_size=batch_size, shuffle=True, is_train=randomize_img_per_subj, pad_factor=pad_factor, store_images=store_images)
            self.val = self.get_loader(subjs['val'], batch_size=batch_size, shuffle=False, is_train=False,pad_factor=pad_factor,store_images=store_images)
        else:
            self.train, self.val = None, None
        self.test = self.get_loader(subjs['test'], batch_size=batch_size, shuffle=False, is_train=False,pad_factor=pad_factor,store_images=store_images)
        

    def get_map_directories( self, data_dir, img_dir, label_dir, img_key=None, segmentation_key=None):
        '''
        Method to get image and label map directories
        Inputs:
        data_dir: full path to directory containing data, with each subfolder being a subject. ex: '~/data/' (str)
        img_dir: subdirectory name, per subject, that contains the image data. ex: 'image' (str)
        label_dir: subdirectory name, per subject, that contains the segmentation data. ex: 'segmentation' (str)
        img_key: ex: 'BOLD' searches files containing 'BOLD' within subfolder
        label_key: ex: 'placenta' searches files containing 'placenta'
        assumes structures data_dir/SUBJECT/img_dir/*.nii
        '''
        imgs_map = {}
        labels_map = {}
        for fn in os.listdir(data_dir):
            imgs = []
            if os.path.isdir(os.path.join(data_dir,fn)):
                for subname in util.listdir_nohidden(os.path.join(data_dir,fn,img_dir)): #lists the directory in ascending order
                    # check if we have the proper key
                    if subname.find(img_key) >= 0:
                        imgs.append(subname)
                imgs_map[fn] = imgs
        
        for fn in os.listdir(data_dir):
            labels = []
            if os.path.isdir(os.path.join(data_dir,fn)):
                if label_dir is not None:
                    for subname in util.listdir_nohidden(os.path.join(data_dir,fn,label_dir)): #lists the directory in ascending order
                        # check if we have the proper key
                        if subname.find(segmentation_key) >= 0:
                            labels.append(subname)
                    labels_map[fn] = labels
            
        return imgs_map, labels_map

    def get_loader(self, files, transform=None, batch_size=1, shuffle=True, store_images=True, is_train=True, pad_factor=1):
        """
        gets a single data loader. 

        Params: 
        files: list of filenames from which to load subject names. (list)
            we assume that the files containing the subjects are located in self.data_dir. 
        transform: string containing data augmentation transformations to apply (str)
        batch_size: batch size (int)
        store_images: whether to store images in RAM or load (bool)
        is_train: parameter whether this is from a training set or not. For training set, randomly samples images from each subject to 
            account for class imbalance. If false, just takes all the images. (bool)
        pad_factor: factor to make images divisible by (int)

        Return: a dataloader for the specified set. 
        """
        self.imgs, self.labels, self.subj_names = self.get_all_filenames(files, self.img_dir, self.label_dir, self.imgs_map, self.labels_map, is_train=is_train)
        return data.DataLoader(MriDataset(self.data_dir, self.img_dir, self.label_dir, self.subj_names, self.imgs, self.labels, pad_factor=pad_factor, is_train = is_train, transforms=transform, store_images=store_images), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)


    def get_all_filenames(self, files, img_dir, label_dir, imgs_map, labels_map, is_train=False):
        '''
        files: list of filenames from which to load subject names. (list)
            we assume that the files containing the subjects are located in self.data_dir. 
        img_dir: subdirectory name, per subject, that contains the image data. ex: 'image' (str)
        label_dir: subdirectory name, per subject, that contains the segmentation data. ex: 'segmentation' (str)
        imgs_map: mapping to grab images (list)
        labels_map: mapping to grab labels (list)
        is_train: whether creating a train dataset or not. If true, (bool)

        Returns:
            imgs, labels, subj_names (lists)
        '''
        subjs = files
        imgs = {}
        labels = {}
        keys_to_extract = []
        for fn in subjs:
            if fn in imgs_map:
                keys_to_extract.append(fn)
        imgs = { key: imgs_map[key] for key in keys_to_extract}
        if labels_map is not None:
            labels = {key: labels_map[key] for key in keys_to_extract}

        #if not training, instead we will pass a list of all the image and label paths
        subj_names = list(imgs.keys())
        subj_names.sort()

        if not is_train:
            img_names = list(imgs.items())
            label_names = list(labels.items())

            imgs = []
            labels = []
            for i in range(0,len(img_names)):
                name = img_names[i][1]
                subjName = img_names[i][0]
                for count, img_id in enumerate(name):
                    imgs.append((subjName + '/' + img_dir + '/' + img_id))
            if len(label_names) > 0:
                for i in range(0,len(label_names)):
                    name = label_names[i][1]
                    subjName = label_names[i][0]
                    for count, label_id in enumerate(name):
                        labels.append((subjName + '/' + label_dir + '/' + label_id))

        return imgs, labels, subj_names      

    def read_subjs(self, fn):
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

class MriDataset(data.Dataset):
    """
    creates a dataset from the given images/labels.

    Params: 
    subj_names: list of subject names
    data_path: parent directory where the data lives. Structure: data_dir/SUBJNAME/img_dir/
    img_dir: image subdirectory. (str)
    label_dir: label subdirectory. (str)
    img: list of the images' filenames. (list)
    label: list of labels' filenames. (list)
    transform: transformations to apply (str)
    is_train: boolean to indicate whether this is from training set and we want to use randomized loader
    pad_factor: scalar: factor to make images divisible by
    store_images: boolean, whether to store images in memory or use I/O when accesing data.
    """
    def __init__(self, data_path, img_path, label_path, subj_names, img, label, is_train, pad_factor=1, transforms=None, store_images=True):
        self.subj_names = subj_names
        self.data_path = data_path
        self.img_dir = img_path
        self.label_dir = label_path
        self.imgs = img
        self.labels = label
        self.transform = transforms
        self.is_train = is_train
        self.pad_factor = pad_factor
        self.store_images = store_images


        # load all the images, and all the labels as a tensor (or in torch.io, load them sequentially)
        self.img_tensor, self.label_tensor, self.img_affine, self.label_affine, self.highs, self.filenames, \
            self.filenames_label, self.filename_img_paths, self.filename_label_paths, self.idx_map, self.subj_names_nontrain, self.pad_amnts = self.get_img_tensor( self.data_path, self.img_dir, self.label_dir, self.imgs, self.labels, self.is_train, store_images=self.store_images)
     

    def get_img_tensor( self, data_path, img_dir, label_dir, imgs_dict, labels_dict, is_train=True, store_images=True):
        '''
        Get a tensor of all the images (or labels)
        Params:
            is_train: If true, creates indexing per subject rather than per image
            store_images: boolean. If true, stores images and labels in RAM
        Returns:
            imgs : NxXxYxZ numpy array of all N images in the dataset
            labels: NxXxYxZ numpy array of all N labels in the dataset
            img_affines: Nx4x4: affine matrix of images, used for saving
            highs: 2XN matrix, where highs[0,i] = 95th percentile of image i, highs[1,i] is the min of image i
            filenames: names of the image files
            filenames_label: names of the label files
            filename_img_paths: paths of the images
            filename_label_paths: paths of the labels
            index_map: list with subject names, and pointer to where to grab the index labels. Only used for when needing random data loader.
        '''
        imgs = np.array([])
        labels = np.array([])
        img_affines = np.array([])
        label_affines = np.array([])
        filenames = list()
        filename_img_paths = list()
        filenames_label = list()
        filename_label_paths = list()
        highs = np.array([])
        pad_amnts = np.array([])
        index_map = dict()
        subj_names_non_train = list()
        if not is_train:
            for (img_filename, label_filename) in zip(imgs_dict, labels_dict):
                #grab the names
                img_name = img_filename.split('/')[-1]
                img_path = img_filename
                label_name = label_filename.split('/')[-1]
                label_path = label_filename
                subj_name = img_filename.split('/')[0]
                filenames.append(img_name)
                filename_img_paths.append(img_path)
                filenames_label.append(label_name)
                filename_label_paths.append(label_path)
                subj_names_non_train.append(subj_name)
                #get the image, label
                if store_images:
                    img, pct, low, affine, pad_amnt = self.get(os.path.join(data_path,img_filename), is_img=True)
                    label, _, _, affine_label, _ = self.get(os.path.join(data_path, label_filename), is_img=False)
                    label = np.around(np.clip(label,0,1)).astype(np.int8)
                    imgs, labels, img_affines, label_affines, highs, pad_amnts = self.store_data_RAM(imgs, labels, highs, img_affines, label_affines, pad_amnts, img, label, pct, low, affine, affine_label, pad_amnt)
        else:
            ctr = int(0)
            for (subj_img, subj_label) in zip(imgs_dict.keys(), labels_dict.keys()):
                label_map = np.array([])
                assert subj_img == subj_label
                for (img_filename, label_filename) in zip(imgs_dict[subj_img], labels_dict[subj_label]):
                    #grab the names
                    img_name = img_filename
                    label_name = label_filename
                    img_path = os.path.join( subj_img,img_dir, img_filename)
                    label_path = os.path.join(subj_label,label_dir, label_filename)
                    filenames.append(img_name)
                    filename_img_paths.append(img_path)
                    filenames_label.append(label_name)
                    filename_label_paths.append(label_path)
                    # append the counter
                    label_map = np.append(label_map, int(ctr))
                    index_map[subj_img] = label_map
                    ctr = ctr+1
                    #get the image, label
                    if store_images:
                        img, pct, low, affine, pad_amnt = self.get(os.path.join(data_path, subj_img, img_dir, img_filename), is_img=True)
                        label, _, _, affine_label,_ = self.get(os.path.join(data_path, subj_label, label_dir, label_filename), is_img=False)
                        label = np.around(np.clip(label,0,1)).astype(np.int8)
                        imgs, labels, img_affines, label_affines, highs, pad_amnts = self.store_data_RAM(imgs, labels, highs, img_affines, label_affines, pad_amnts, img, label, pct, low, affine, affine_label, pad_amnt)

        return imgs, labels, img_affines, label_affines, highs, filenames, filenames_label, filename_img_paths, filename_label_paths, index_map, subj_names_non_train, pad_amnts

    
    def store_data_RAM(self, imgs, labels, highs, img_affines, label_affines, pad_amnts, img, label, pct, low, affine, affine_label, pad_amnt):
        '''
        Stores images, labels, image intensity parameters, image NIFTI header, and pad amounts in RAM
        '''
        if highs.size == 0:
            highs = np.vstack((pct, low))
        else:
            highs = np.hstack((highs, np.vstack((pct, low)) ))
        #stack them
        img = np.expand_dims(img, 0)
        label = np.expand_dims(label, 0)
        affine = np.expand_dims(affine,0)
        affine_label = np.expand_dims(affine_label,0)
        pad_amnt = np.expand_dims(pad_amnt, 0)
        if imgs.size == 0:
            imgs = img
            labels = label
            img_affines = affine
            pad_amnts = pad_amnt
            label_affines = affine_label
        else:
            imgs = np.concatenate((imgs, img), axis=0)
            labels = np.concatenate((labels,label), axis=0)
            img_affines = np.concatenate((img_affines,affine),axis = 0)
            label_affines = np.concatenate((label_affines,affine_label),axis=0)
            pad_amnts = np.concatenate((pad_amnts,pad_amnt),axis=0)
        
        return imgs, labels, img_affines, label_affines, highs, pad_amnts

    
    def normalize(self, img, pct=95):
        """
        linearly scales the percentile (pct) of the image to 1. 
        """
        img[img < 0] = 0
        high = np.percentile(img, pct)
        low = np.min(img)
        #img = (img - low) / high
        img = (img - low) / (high-low)
        return img, high, low

    def pad_img(self, img, factor=16, mode='minimum'):
        """
        pads an image to be divisible by factor
        """
        if factor <= 1:
            return img
        pad_amnt = np.remainder(np.shape(img),factor)
        pad_amnt = factor - pad_amnt
        pad_amnt[np.where(pad_amnt==factor)] = 0
        rem = np.remainder(pad_amnt,2)
        pad_amnt = np.floor(np.divide(pad_amnt,2))
        pad_amnt = np.repeat(pad_amnt,2)
        pad_amnt[::2] = pad_amnt[::2] + rem
        pad_amnt = pad_amnt.reshape(-1,2)
        pad_amnt = pad_amnt.astype(int)
        # for 4D, don't pad the end
        if len(np.shape(img))>3:
            pad_amnt[-1,:] = 0
        img = np.pad(img,pad_amnt,mode=mode) 
        return img, pad_amnt

    def get(self, path, is_img=False):
        """
        gets an image/label given its path. 

        Params: the path to the file to read. 

        Return: (img, 95_pct)
            - img: the image
            - 95_pct: the value of the 95th percentile if path leads to an image
        """
        pad_factor = self.pad_factor
        img = nib.load(path)
        affine = img.affine
        img = np.array(img.dataobj)
        #condition for 4D cases.
        if len(img.shape) == 4:
            if img.shape[3] == 1:
                img = img[:,:,:,0]
        img = img.astype(float)
        if is_img:
            img, high, low = self.normalize(img)
            img, pad_amnt = self.pad_img(img,factor=pad_factor,mode='minimum')
        else:
            img, pad_amnt = self.pad_img(img,factor=pad_factor,mode='constant')
            high = -1
            low = -1
        return img, high, low, affine, pad_amnt
    
    def get_img_idx(self,idx):
        '''
        Gets the appropriate index from the images
        '''
        if self.is_train:
            subj_names = self.subj_names
            subj_name = subj_names[idx]
            #now, grab the list of images corresponding to this subject.
            inds = self.idx_map[subj_name]
            #randomly sample an image to grab
            idx_img = np.random.randint(inds[0],high=inds[-1]+1,size=1)[0]
        else:
            idx_img = idx
            subj_name = self.subj_names_nontrain[idx]
        
        return idx_img, subj_name

    def get_item_stored(self,idx):
        '''
        Used to grab data from memory
        '''
        idx_img, subj_name = self.get_img_idx(idx)
        img = self.img_tensor[idx_img,:,:,:]
        label = self.label_tensor[idx_img,:,:,:]
        img_affine = self.img_affine[idx_img,:,:]
        label_affine = self.label_affine[idx_img,:,:]
        pct = self.highs[0,idx_img]
        low = self.highs[1,idx_img]
        img_name = self.filenames[idx_img]
        label_name = self.filenames_label[idx_img]
        img_path = self.filename_img_paths[idx_img]
        label_path = self.filename_label_paths[idx_img]
        pad_amnt = self.pad_amnts[idx_img]
        return img, label, img_affine, label_affine, pct, low, subj_name, img_name, label_name, img_path, label_path, pad_amnt

    def get_item_io(self, idx):
        '''
        grabs data from disk
        '''
        idx_img, subj_name = self.get_img_idx(idx)
        img, pct, low, img_affine, pad_amnt = self.get(os.path.join(self.data_path,self.imgs[idx_img]), is_img=True)
        label, _, _,label_affine, _ = self.get(os.path.join(self.data_path, self.labels[idx_img]), is_img=False)
        label = np.around(np.clip(label,0,1)).astype(np.int8)
        label_name = self.filenames_label[idx_img]
        img_path = self.filename_img_paths[idx_img]
        label_path = self.filename_label_paths[idx_img]
        img_name = self.filenames[idx_img]
        return img, label, img_affine, label_affine, pct, low, subj_name, img_name, label_name, img_path, label_path, pad_amnt 
            

    def __getitem__(self, idx):
        """
        gets a sample with given index. 

        Params:
        idx: the index of the sample. The idx refers to the subject. Then, a random
        (img, label) pair is pulled from the available list of images.

        Return: a dict with keys: 
            - 'img': the image
            - 'label': the label
            - 'fn': the filename of the sample
            - 'fn_img_path': full path of the image
            - 'fn_label_path': full path of the segmentation label
            - 'fn_label': filename of the segmentation label
            - '90_pct': the Nth percentile of the image, used to rescale the image
                back to normal intensity for saving
            - 'low': minimum of the image value used in normalization
            - 'affine': image NIFTI affine header (numpy array)
            - 'subj_name': name of the subject
            - 'pad_amnt': 3x2 numpy array with pad amounts used on the image
        """
        if self.store_images:
            img, label, img_affine, label_affine, pct, low, subj_name, img_name, label_name, img_path, label_path, pad_amnt = self.get_item_stored(idx)
        else:
            img, label, img_affine, label_affine, pct, low, subj_name, img_name, label_name, img_path, label_path, pad_amnt = self.get_item_io(idx)
        img, label = np.expand_dims(img, 0), np.expand_dims(label, 0)
        sample = tio.Subject({
            'img': tio.ScalarImage(tensor=img),
            'label': tio.LabelMap(tensor=label),
            'fn': img_name,
            'fn_img_path': img_path,
            'fn_label_path': label_path,
            'fn_label': label_name,
            '90_pct': pct,
            'low' : low,
            'affine': img_affine,
            'label_affine': label_affine,
            'subj_name': subj_name,
            'pad_amnt': pad_amnt
        })

        if self.transform != None:
            sample = self.transform(sample) # make sure to use include and exclude keywords
        return sample

    # need to specify this to iterate through the dataset. 
    def __len__(self):
        if self.is_train:
            return len(self.subj_names)
        else:
            return len(self.imgs)
