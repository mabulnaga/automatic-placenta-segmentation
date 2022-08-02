import numpy as np
import os
import torch
import torch.nn as nn
from unet_3d import UNet
import torch.nn as nn
import util
from train_placenta import split_train_val
import argparse
import csv
import postprocess
import metrics
from metrics import dice
from medpy.metric.binary import assd as ASSD
from medpy.metric.binary import hd as Hausdorff_Distance
from medpy.metric.binary import hd95 as Hausdorff_Distance_95

HAUSDORFF_PCTILE = 95
IMG_DIR_NAME = 'image'
LABEL_DIR_NAME = 'segmentation'
PAD_FACTOR = 16 #factor to make images divisible by.


def evaluate(model, test, device, save_dir):
    """
    Evaluates the given model on the given dataset. 

    Params:
    model: the model to evaluate
    test: the test dataset
    device: the device to run the evaluation on
    save_dir: the path to the directory where the results should be saved
    """
    model.eval()
    dice_score = 0
    hausd = 0
    hausd_pct = 0
    total_dice = 0
    total_hausd = 0
    sig = nn.Sigmoid()
    dices = []
    hausdorff = []
    num_island = []
    hausdorff_pct = []
    assd_list = []
    mean_bold_list = []
    assd = 0
    names = []
    voxel_to_mm = 1.

    with torch.no_grad():
        for batch in test:
            # run prediction
            print("NEW BATCH")
            images, labels, fn, factor, low, image_fn_path, img_affine, label_affine, subj_name, pad_amnt = batch['img']['data'], batch['label']['data'], batch['fn'], batch['90_pct'], batch['low'], batch['fn_img_path'], batch['affine'], batch['label_affine'], batch['subj_name'], batch['pad_amnt']
            images, labels = images.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)
            outputs = model(images)

            predicted_probs = sig(outputs)
            images, predicted_probs, labels = images.to('cpu'), predicted_probs.to('cpu'), labels.to('cpu')
            predicted = (predicted_probs > 0.5).float()

            if len(np.shape(predicted)) == 5:
                predicted = torch.squeeze(predicted,1)
                predicted_probs = torch.squeeze(predicted_probs,1)
                labels = torch.squeeze(labels,1)
                images = torch.squeeze(images,1)
            # convert to 0/1 labels
            for i in range(predicted.shape[0]):
                # clean the predicted image
                subject = subj_name[i]
                predicted_img = predicted[i].numpy()
                label = labels[i].numpy()
                img = images[i].numpy()
                # post process and clean
                cleaned_img = postprocess.remove_small_objects(predicted_img)
                cleaned_img, _ = postprocess.remove_islands(cleaned_img)
                predicted_img = cleaned_img
                total_dice += 1
                total_hausd +=1
                # print information
                print("IMAGE: {}".format(fn[i]))
                d = dice(predicted_img, label)
                if np.sum(predicted_img) > 0:
                    hd = Hausdorff_Distance(predicted_img, label,voxelspacing=voxel_to_mm)
                    hd_pctile = Hausdorff_Distance_95(predicted_img, label, voxelspacing=voxel_to_mm)
                    assd_ind = ASSD(predicted_img,label,voxel_to_mm)
                else:
                    hd = np.nan
                    hd_pctile = np.nan
                    assd_ind = np.nan
                # compute relative mean BOLD difference
                bold_diff = metrics.mean_BOLD_difference(util.unnormalize_img(img,low=low[i].numpy(),high=factor[i].numpy()),(label>0.5).astype(bool), predicted_img.astype(bool))
                dice_score += d
                hausd += hd
                hausd_pct += hd_pctile
                assd +=assd_ind
                print("dice: {}".format(d))
                dices.append(d)
                names.append(fn[i])
                hausdorff.append(hd)
                hausdorff_pct.append(hd_pctile)
                assd_list.append(assd_ind)
                mean_bold_list.append(bold_diff)

                # save the images -- makes it better for later processing results.
                save_dir_subject = os.path.join(save_dir,subject)
                #make directories for image and segmentation
                if not os.path.exists(os.path.join(save_dir_subject,'image')):
                    os.makedirs(os.path.join(save_dir_subject,'image'))
                if not os.path.exists(os.path.join(save_dir_subject,'predicted_segmentation')):
                    os.makedirs(os.path.join(save_dir_subject,'predicted_segmentation'))
                if not os.path.exists(os.path.join(save_dir_subject,'true_segmentation')):
                    os.makedirs(os.path.join(save_dir_subject,'true_segmentation'))
                
                img_pad_amnt = np.concatenate(pad_amnt[i].numpy(),axis=0)
                # unpad images
                img = util.unpad_img(util.unnormalize_img(img, low[i].numpy(), factor[i].numpy()),img_pad_amnt)
                predicted_img = util.unpad_img(predicted_img, img_pad_amnt)
                label = util.unpad_img(label,img_pad_amnt)
                # save 3d prediction to subject folder
                util.save_img(predicted_img, os.path.join(save_dir_subject,'predicted_segmentation'), 'predicted_segmentation_'+fn[i],img_affine[i])
                util.save_img(img, os.path.join(save_dir_subject,'image'), fn[i],img_affine[i])
                util.save_img(label, os.path.join(save_dir_subject,'true_segmentation'), 'true_segmentation_'+fn[i],label_affine[i])

    print('Average dice of the network on the test images: {}'.format(
        dice_score / total_dice))
    print('Average hausdorff of the network on the test images: {}'.format(
        hausd / total_hausd))
    # write to file the dice scores per subject
    with open(os.path.join(save_dir, 'stats','stats.csv'), mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',')
        csv_file.writerow(['subj','dice','hausdorff', 'hausdorff_'+str(HAUSDORFF_PCTILE), 'assd', 'mean_bold_diff'])
        for j in range(len(names)):
            csv_file.writerow([names[j],dices[j],hausdorff[j], hausdorff_pct[j], assd_list[j], mean_bold_list[j]])
        csv_file.writerow(['Average',np.mean(dices),np.mean(hausdorff), np.mean(hausdorff_pct), np.mean(assd_list), np.mean(mean_bold_list)])
        csv_file.writerow(['Std',np.std(dices),np.std(hausdorff), np.std(hausdorff_pct), np.std(assd_list), np.std(mean_bold_list)])
    

def main(model_path, save_dir, data_path, img_dir, label_dir, subj_folds, test_only=False):
    """
    sets up the model to be evaluated, and evaluates the model on both the train and test
    datasets. 

    Params:
    model_path: the path to the saved model file to be loaded in
    save_dir: the path to the directory where results should be saved
    data_path: the path to the directory where data is saved
    img_dir: subjdirectory for the images
    label_dir: subdirectory for the labels
    subj_folds: a dictionary with the text files for subject folds (train/val/test)
    test_only: bool. whether to only create a test set or not
    """
    pad_factor = PAD_FACTOR
    device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu")
    #device = torch.device("cpu")
    model = UNet(1)
    model = model.to(device)
    model, _, epoch, _ = util.load_checkpt(model_path, model)
    data = split_train_val(data_path, img_dir, label_dir, "none", subj_folds, batch_size=1, pad_factor = pad_factor, randomize_img_dataloader=False, store_images=False,test_only=test_only)
    train, test, val = data.train, data.test, data.val
    if not test_only:
        print(' TRAINING SET SIZE: ' + str(len(train)), flush=True)
        save_dir_train = os.path.join(save_dir,'train')
        #create directories for saving
        try:
            os.makedirs(os.path.join(save_dir_train,'stats'))
        except OSError as error:
            print(error)
        evaluate(model, train, device, save_dir_train)
        
        print("START VAL")
        save_dir_val = os.path.join(save_dir,'val')
        try:
            os.makedirs(os.path.join(save_dir_val,'stats'))
        except OSError as error:
            print(error)
        evaluate(model, val, device, save_dir_val)
 
    print("START TEST")
    save_dir_test = os.path.join(save_dir,'test')
    try:
        os.makedirs(os.path.join(save_dir_test,'stats'))
    except OSError as error:
        print(error)
    evaluate(model, test, device, save_dir_test)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate trained unet model')
    parser.add_argument('--save_path', dest='save_path', default='~/Documents/model-output/test001', 
        help='full path to location where experiment outputs will go')
    parser.add_argument('--model_name', dest='model_name', default='model_PIPPI')
    parser.add_argument('--data_path', dest='data_path', 
        default='~/Documents/data/')
    parser.add_argument('--eval_existing_folds', action='store_true')

    args = parser.parse_args()
    
    model_folder = os.path.join(args.save_path,'model') #need to get parent directory.
    model_path = os.path.join(model_folder,args.model_name+'.pt')
    save_path = os.path.join(args.save_path,'results',args.model_name+'_output')

    img_dir = IMG_DIR_NAME
    label_dir = LABEL_DIR_NAME

    # evaluate previously defined folds
    if args.eval_existing_folds: 
        subj_folds = np.load(os.path.join(model_folder, 'model-folds.npy'), allow_pickle='TRUE').item()
        test_only = False
    else:
        subj_folds = dict()
        # create a new dataset based on the data directory
        dir_list = util.listdir_nohidden_sort_numerical(args.data_path, list_dir=True, sort_digit=True)
        subj_folds['test'] = dir_list
        subj_folds['train'] = []
        subj_folds['val'] = []
        test_only = True

    print(subj_folds)
    main(model_path, save_path, args.data_path, img_dir, label_dir, subj_folds, test_only)
