import util
from util import split_train_val
from losses import boundary_weighted_loss
from metrics import dice_tensor
from unet_3d import UNet
from monai.losses.dice import DiceLoss, FocalLoss
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import argparse
import pandas as pd
import math

IMG_DIR_NAME = 'image' # subdirectory name where images are stored (per subject) 
LABEL_DIR_NAME = 'segmentation' # subdirectory name where labels are stored (per subject)
DATA_SPLITS = [0.65,0.15,0.2] #train, validation, test split percentages
PAD_FACTOR = 16 #factor to make images divisible by.
STORE_IMAGES = True #stores images in RAM. If dataset has images of different sizes, set to False.

def test_model(args, model, test, device, loss_function, inner_boundary_weight=None, outer_boundary_weight=None,
    boundary_kernel=None, test_set=False):
    """
    Evaluates the model on the given test set. 

    Params:
    model: the model to evaluate
    test: the test dataset
    device: the device to evaluate on 
    loss_function: the loss function
    save_output: a bool. True if output should be saved, False otherwise
    pred_path: the path to the directory where predictions should be saved
    data_path: the path to the directory with the data
    test_set: bool: if True, output will be on test set, else validation set. Does not affect any computation.

    Return: average test loss, average test dice
    """
    model.eval()
    dice_score = 0
    total_loss = 0
    total_dice = 0
    val_loss = 0
    val_boundary_loss = 0
    sig = nn.Sigmoid()
    dice_loss = DiceLoss(include_background=True,sigmoid=True) if args.use_Dice_loss else None

    with torch.no_grad():
        for batch in test:
            images, labels, fn, factor = batch['img']['data'], batch['label']['data'], batch['fn'], batch['90_pct']
            images, labels = images.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)
            predicted = model(images)
            
            batch_loss, boundary_batch_loss = loss(loss_function, predicted, labels, dice_loss, boundary_kernel=boundary_kernel, inner_boundary_weight=inner_boundary_weight,
                outer_boundary_weight=outer_boundary_weight, use_Dice_loss=args.use_Dice_loss, dice_loss_weight=args.dice_loss_weight)
            
            val_loss+=batch_loss.item()
            val_boundary_loss+=boundary_batch_loss.item()
            total_loss += 1

            predicted = sig(predicted)
            if len(np.shape(predicted)) == 5:
                predicted = torch.squeeze(predicted,1)
                labels = torch.squeeze(labels,1)
                images = torch.squeeze(images,1)
            predicted = (predicted > 0.5).float()
            for i in range(predicted.shape[0]):
                d = dice_tensor(predicted[i], labels[i])
                dice_score += d
                total_dice += 1
    if test_set:
        print_name = 'test'
    else:
        print_name = 'validation'
    print('Average dice of the network on the {} images: {}'.format(print_name, dice_score / total_dice),flush=True)
    print('Average loss of the network on the {} images: {}'.format(print_name,
        val_loss / total_loss),flush=True)
    print('Average boundary loss of the network on the {} images: {}'.format(print_name,
        val_boundary_loss / total_loss),flush=True)
    model.train()
    return val_loss/total_loss, dice_score/total_dice, val_boundary_loss / total_loss


def train_model(args, data_dir, img_dir, label_dir, model_path, transform_str, weighted_bce, loss_output,
                epochs=500, lr=0.001, data_split=[], aug_severity=0, randomize_img_dataloader=False,
               inner_boundary_weight=None, outer_boundary_weight=None, boundary_kernel=(7,7,7), load_model_checkpt=False):
    """
    trains the model. 

    Params:
    args: argparse arguments
    data_dir: the directory where the data is saved. (str)
    img_dir: subdirectory where the images are (str)
    label_dir: subdirectory where the labels are (str)
    model_path: the path where the model should be saved. (str)
    transform_str: a comma delimited string with the transforms to apply during training. (str)
    weighted_bce: a bool that is True if weighted bce should be used and false otherwise (bool)
    loss_output: a string with the file where the loss/dice output should be written. (str)
    epochs: the number of epochs to train (int)
    lr: the learning rate to use (float)
    data_split: folds to use in training, testing, and validation. (dict)
    aug_severity: integer determining the augmentation severity to use (0 to 4) (int)
    randomize_img_dataloader: True/False: whether to randomly sample one image per subject in the train data loder. (for train only) (bool)
    inner_boundary_weight: additive proportional weight (to CE weight) to the inner boundary (inside the placenta) ex: 0.5, add 0.5*CE (float)
    outer_boundary_weight: additive proportional weight (to CE weight) to the outer boundary (outside the plancenta) (float)
    kernel_size: (k,k,k) tuple for boundary kernel size. (tuple)
    load_model_checkpt: Boolean, whether to load the model from the checkpoint, or from scratch. (bool)
    """
    # set up the CSV columns
    CSV_COLUMNS = ['epoch', 'val_dice', 'val_boundary_loss', 'val_loss', 'train_dice', 'train_boundary_loss', 'train_loss', 'teacher_loss']
    MAX_METRICS = ['val_dice','train_dice']
    MIN_METRICS = ['val_loss', 'train_loss']
    if inner_boundary_weight or outer_boundary_weight: 
        MIN_METRICS = ['val_boundary_loss', 'val_loss', 'train_boundary_loss', 'train_loss']
    output = []
    dices = []
    gamma=args.focal_loss_gamma
    start_epoch = 0
    sig = nn.Sigmoid()
    pad_factor = PAD_FACTOR
    batch_size = args.batch_size
    # load the data set and data loader
    data = split_train_val(data_dir, img_dir, label_dir, transform_str, data_split, \
        batch_size = batch_size, pad_factor = pad_factor, \
         randomize_img_dataloader=randomize_img_dataloader, aug_severity=aug_severity, store_images=STORE_IMAGES)
    # split into train/test/val
    train, test, val = data.train, data.test, data.val
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    # set up the loss functions
    dice_loss = DiceLoss(include_background=True,sigmoid=True,squared_pred=True)
    dice_loss = dice_loss.to(device) 
    weight = None
    if weighted_bce:
        weight = torch.ones([1], dtype=torch.float) * util.get_pos_weight(train)
        focal_weight = float(1-1/weight)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=weight,reduction="none") if not args.use_Focal_loss else FocalLoss(weight=focal_weight, gamma=gamma,reduction="none",include_background=True)
    else:
        focal_weight = args.focal_loss_alpha
        loss_function = nn.BCEWithLogitsLoss(reduction="none") if not args.use_Focal_loss else FocalLoss(weight=focal_weight, gamma=gamma,reduction="none",include_background=True)
    loss_function = loss_function.to(device)

    # initialize model
    model = UNet(1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # load model if necessary
    if load_model_checkpt:
        model, optimizer, start_epoch, dices = util.load_checkpt(args.preloaded_model_path, model, optimizer)
        dices = []
        start_epoch = start_epoch - 1

    global_step = len(train)*start_epoch
    # optimizer
    # total_steps = len(train) * epochs 
    # lambda1 = lambda global_step: 1.0 - 1.00 * (global_step/ total_steps)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # START TRAINING!
    for epoch in range(start_epoch, epochs):
        boundary_running_loss = 0 
        total_running_loss = 0
        total_running_loss_2 = 0
        running_dice = 0
        total_loss = 0
        total_dice = 0
        
        # save the output CSV
        if epoch % 25 == 0 or len(output) == 0:
            df = pd.DataFrame(output, columns=CSV_COLUMNS)
            df.to_csv(loss_output, index=False)

        print('Starting epoch {}/{}'.format(epoch + 1, epochs))
        model.train()

        for batch in train:
            global_step += 1
            total_loss += 1
            # get the data
            inputs, labels, fn = batch['img']['data'], batch['label']['data'], batch['fn']
            inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)

            optimizer.zero_grad()
            # pass through model and compute predictions
            outputs = model(inputs)
            # compute the loss
            loss1, loss1_boundary = loss(loss_function, outputs, labels, dice_loss, boundary_kernel=boundary_kernel, inner_boundary_weight=inner_boundary_weight,
                outer_boundary_weight=outer_boundary_weight, use_Dice_loss=args.use_Dice_loss, dice_loss_weight=args.dice_loss_weight)
            boundary_running_loss += loss1_boundary.item() 
            total_weighted_loss = loss1
            total_running_loss+= loss1.item()            
            total_weighted_loss.backward()
            optimizer.step()
           
            # calculate dice            
            predicted = sig(outputs)
            if len(np.shape(predicted)) == 5:
                predicted = torch.squeeze(predicted,1)
                labels = torch.squeeze(labels,1)
            for i in range(predicted.shape[0]):
                dice_tensor_score = dice_tensor(predicted[i], labels[i])
                running_dice += dice_tensor_score
                total_dice += 1
            
        
        val_loss, dice_score, val_boundary_loss = test_model(
            args,
            model,
            val,
            device,
            loss_function,
            inner_boundary_weight=inner_boundary_weight,
            outer_boundary_weight=outer_boundary_weight,
            boundary_kernel=boundary_kernel,
            test_set=False
        )
        #scheduler.step()
        print('Average dice of the network on the train images: {}'.format(running_dice / total_dice),flush=True)
        print('Average loss of the network on the train images: {}'.format(total_running_loss / total_loss),flush=True)
        print('Average boundary loss of the network on the train images: {}'.format(boundary_running_loss / total_loss),flush=True)

        output.append([epoch + 1, dice_score, val_boundary_loss, val_loss, running_dice/total_dice, boundary_running_loss/total_loss, total_running_loss/total_loss, total_running_loss_2/total_loss])

        # save model with best dice scores so far
        if len(dices) == 0 or dice_score > max(dices) or epoch % 25==0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dices'    : dices
                        }
            if len(dices) == 0 or dice_score > max(dices):
                is_best=True
            else:
                is_best=False
            util.save_checkpt(checkpoint, model_path, best_model_name='model_best', is_best=False)
            if is_best:
                util.save_checkpt(checkpoint, model_path, best_model_name='model_best', is_best=True)
            best_dice = epoch + 1
            dices.append(dice_score)
        
        # every 25 epochs, also save the best metrics
        if epoch % 25 == 0 or is_best:
            df = pd.DataFrame(output, columns=CSV_COLUMNS)
            max_metrics, min_metrics = [], []
            for metric in MAX_METRICS:
                ind = df.index[df[metric] == df[metric].max()]
                s = df.iloc[[ind[0]]]
                max_metrics.append(s)
            for metric in MIN_METRICS:
                ind = df.index[df[metric] == df[metric].min()]
                s = df.iloc[[ind[0]]]       
                min_metrics.append(s)
            best_of_each_df = pd.concat(max_metrics+min_metrics)
            directory = os.path.dirname(loss_output)
            best_of_each_df.to_csv(os.path.join(directory,'best_metrics.csv'),index=False)  


    ##### TRAINING FINISHED ######
    #save last round, in case not caught by epoch % n 
    df = pd.DataFrame(output, columns=CSV_COLUMNS)
    df.to_csv(loss_output, index=False)
    model, optimizer, start_epoch, dices = util.load_checkpt(os.path.join(model_path,'model_best.pt'), model, optimizer)
    test_model(args, model, test, device, loss_function, inner_boundary_weight=inner_boundary_weight, outer_boundary_weight=outer_boundary_weight, boundary_kernel=boundary_kernel, test_set=True)


def loss(loss_function,outputs, labels, dice_loss_handle=None, boundary_kernel = (3,3,3), inner_boundary_weight = 0, outer_boundary_weight = 0, dice_loss_weight=1.0, use_Dice_loss=False):
    '''
    Computes the loss function
    Inputs:
            loss_function: handle to loss function (Cross-entropy or Focal)
            Outputs: BxCxHxWxD tensor
            Labels: BxCxHxWxD tensor of ground truth values
            dice_loss_handle: class to Dice loss
            weight: positive weight to use in cross-entropy
            boundary_kernel: kernel size used to identify boundary
            inner_boundary_weight: additive weight for inner placenta boundary. Set to 0 if not wanting to use
            outer_boundary_weight: additive weight for outer placenta boundary
            dice_loss_weight: weighting of Dice loss
            use_Dice_loss: Boolean (default false)
    Returns:
            total_loss
            boundary_loss
    '''
    boundary_loss = boundary_weighted_loss(
        loss_function,
        outputs, 
        labels, 
        patch_size=boundary_kernel, 
        boundaries_add_factor=inner_boundary_weight, 
        out_boundary_factor=outer_boundary_weight,
        just_boundary=True
    )
    total_weighted_loss = boundary_weighted_loss(
        loss_function,
        outputs, 
        labels, 
        patch_size=boundary_kernel, 
        boundaries_add_factor=inner_boundary_weight, 
        out_boundary_factor=outer_boundary_weight,
    )
                
    if use_Dice_loss:           
        print('using Dice loss')
        dice_l = dice_loss_weight*dice_loss_handle(outputs,labels)
        total_weighted_loss = total_weighted_loss + dice_l
    else:
        dice_l = 0

    return total_weighted_loss, boundary_loss


def main(args, data_path, model_path, output_path, epochs, lr, transform_str, weighted_bce, loss_output, img_dir, label_dir, data_split, 
 aug_severity, randomize_img_dataloader,  inner_boundary_weight, outer_boundary_weight, boundary_kernel, load_model_checkpt):
    """
    Trains a model on the given data. 

    Params:
    args: all arguments
    data_path: the directory where data is stored
    model_path: the directory where model should be saved
    output_path: the directory where predictions should be saved
    epochs: the number of epochs to train for
    lr: the learning rate
    transform_str: a comma-delimited string of the transforms to be applied, none for no transforms
    weighted_bce: a bool, True if weighted bce should be used, False for unweighted
    loss_output: a string with the file where the loss/dice output should be written
    aug_severity: integer determining the severity of the augmentation to use
    randomize_img_dataloader: True/False: whether to randomly sample one image per subject in the train data loder. (for train only)
    inner_boundary_weight: additive proportional weight (to CE weight) to the inner boundary (inside the placenta) ex: 0.5, add 0.5*CE
    outer_boundary_weight: additive proportional weight (to CE weight) to the outer boundary (outside the plance)
    kernel_size: (k,k,k) tuple for boundary kernel size.
    load_model_checkpt: Boolean, whether to load an existing model or train fmor scratch.
    volSARatio_weight: weight on the volume to surface area loss.
    """
    print('Starting training')
    train_model(args, data_path, img_dir, label_dir, model_path, transform_str, weighted_bce, loss_output, epochs=epochs, lr=lr, data_split=data_split, 
                aug_severity=aug_severity, randomize_img_dataloader=randomize_img_dataloader,
               inner_boundary_weight=inner_boundary_weight, outer_boundary_weight=outer_boundary_weight, boundary_kernel=boundary_kernel, load_model_checkpt=load_model_checkpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train unet model for placenta mri scans')
    parser.add_argument('--data_path', dest='data_path', 
        default = '~/Documents/data/')
    parser.add_argument('--save_path', dest='save_path', default='~/Documents/model-output/test001/', help='full path to location where experiment outputs will go')
    parser.add_argument('--epochs', dest='epochs', default=4000)
    parser.add_argument('--lr', dest='lr', default=0.001)
    parser.add_argument('--load_model', action='store_true', help='load a pretrained model')
    parser.add_argument('--transform', dest='transform', default='affine,flip,intensity,noise,gamma,elastic,brightness')
    parser.add_argument('--use_weighted_bce', dest='weighted_bce', action='store_true') 
    parser.add_argument('--inner_boundary_weight', default=1,
                    type=float, help="additive weight on all boundary to scale loss. Will only be placenta boundary if out_boundary_factor is specified")   
    parser.add_argument('--boundary_kernel', default=11,
                    type=int, help="3d kernel size (should be odd int) to determine boundaries")
    parser.add_argument('--outer_boundary_weight', default=40,
                    type=float, help="additive weight on non-placenta boundary to scale loss")          
    parser.add_argument('--randomize_image_dataloader', action="store_true")#to make the dataloader randomize which image it picks from each subject. A batch is considered # subjects in this case.
    parser.add_argument('--aug_severity', dest='aug_severity', default=0, type=int, help='integer for augmentation severity, 0 to 4.')
    parser.add_argument('--use_Dice_loss', action='store_true')
    parser.add_argument('--use_Focal_loss', action='store_true') #if true, will replace cross-entropy
    parser.add_argument('--focal_loss_weight', default=1.0, type=float)
    parser.add_argument('--focal_loss_gamma', default=2.0, type=float)
    parser.add_argument('--dice_loss_weight', default=1.0, type=float)
    parser.add_argument('--focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    # set up some directories to save
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    loss_output = os.path.join(args.save_path , 'output.csv')
    model_path = os.path.join(args.save_path, 'model')
    output_path = os.path.join(args.save_path, 'predicted')
    
    #check whether we need to load an existing model, or train from scratch.
    load_model_checkpt = args.load_model
    # get full path for the pre-loaded model
    if load_model_checkpt:
        args.preloaded_model_path = os.path.join(os.getcwd(),'model','model_PIPPI.pt')
    # random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    #kernel size for boundary estimation
    k = int(args.boundary_kernel)
    kernel_size = (k,k,k)
    # flag to use weighted BCE
    use_weights = args.weighted_bce
    
    # create the dataset splits
    subjs_dict = {}
    data_dir_list = util.listdir_nohidden_sort_numerical(args.data_path, list_dir=True, sort_digit=True)
    N_subj = len(data_dir_list)
    train_split, val_split, test_split = DATA_SPLITS[0], DATA_SPLITS[1], DATA_SPLITS[2]
    num_test_subj = math.ceil(test_split*N_subj)
    num_rem_subj = N_subj - num_test_subj
    num_train_subj = math.floor(train_split*N_subj)
    num_val_subj = num_rem_subj - num_train_subj
    # randomly sort the dataset and split
    rem_split, test_split = torch.utils.data.random_split(data_dir_list, [num_rem_subj, num_test_subj],generator=torch.Generator().manual_seed(0))
    subjs_dict['test'] =  [test_split.dataset[i] for i in test_split.indices]
    subjs_rem = [rem_split.dataset[i] for i in rem_split.indices]
    # split remaining subjects into train/val
    train_split, val_split = torch.utils.data.random_split(subjs_rem, [num_train_subj, num_val_subj],generator=torch.Generator().manual_seed(0))
    subjs_dict['train'] = [train_split.dataset[i] for i in train_split.indices]
    subjs_dict['val'] = [val_split.dataset[i] for i in val_split.indices]

    #point to the appropriate dataset subdirectories
    img_dir = IMG_DIR_NAME
    label_dir = LABEL_DIR_NAME

    # data loader parameter: randomly samples 1 of N_l images per subject during each epoch of training
    randomize_img_dataloader = args.randomize_image_dataloader
    
    # save the folds to load in later when evaluating the model.
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.save(os.path.join(model_path, 'model-folds.npy'), subjs_dict)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save the argparse arguments
    f = open(os.path.join(os.path.dirname(loss_output), 'args.txt'), 'w')
    f.write(repr(args))
    f.close()

    # train
    main(args, args.data_path, model_path, output_path, int(args.epochs), float(args.lr), args.transform, use_weights, loss_output, img_dir, label_dir, subjs_dict, 
        args.aug_severity, randomize_img_dataloader, float(args.inner_boundary_weight), float(args.outer_boundary_weight), kernel_size, load_model_checkpt)
