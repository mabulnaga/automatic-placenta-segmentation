import numpy as np
import os
import torch
import torch.nn as nn
from unet_3d import UNet
import util
import argparse
import csv
from data_loader import DataLoader as DataLoaderInference
import postprocess
import metrics

MODEL_NAME = 'model_PIPPI.pt'
IMG_DIR_NAME = 'image'
LABEL_DIR_NAME = 'image'
PAD_FACTOR = 16 #factor to make images divisible by.

def evaluate(model, test, device, save_dir):
    """
    Evaluates the given model on the given dataset. 

    Params:
    model: the model to evaluate
    test: the test dataset
    device: the device to run the evaluation on
    criterion: the loss function
    save_dir: the path to the directory where the results should be saved
    """
    model.eval()
    sig = nn.Sigmoid()
    names = []
    predictions_dict_4d = dict()
    with torch.no_grad():
        for batch in test:
            # run prediction
            print("NEW BATCH")
            # grab images and labels
            images, labels, fn, img_high, img_low, image_fn_path, affine, pad_amnt, subj_name = batch['img']['data'], batch['label']['data'], batch['fn'], batch['90_pct'], batch['low'], batch['fn_img_path'], batch['affine'], batch['pad_amnt'], batch['subj_name']

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
            
            # save predicted segmentation and image
            for i in range(predicted.shape[0]):
                # clean the predicted image
                predicted_img = predicted[i].numpy()
                predicted_img= postprocess.remove_small_objects(predicted_img)
                predicted_img, _ = postprocess.remove_islands(predicted_img)
                #make directory
                subject = subj_name[i]
                save_dir_subject = os.path.join(save_dir,subject)
                #make directories for image and segmentation
                if not os.path.exists(os.path.join(save_dir_subject,'image')):
                    os.makedirs(os.path.join(save_dir_subject,'image'))
                if not os.path.exists(os.path.join(save_dir_subject,'predicted_segmentation')):
                    os.makedirs(os.path.join(save_dir_subject,'predicted_segmentation'))
                # print information
                print("IMAGE: {}".format(fn[i]))
                names.append(fn[i])
                #unnormalize the image
                img = images[i].numpy()
                img_high_normalize = img_high[i].numpy()
                img_low_normalize = img_low[i].numpy()
                img = util.unnormalize_img(img, img_low_normalize, img_high_normalize)
                # stack this to the dictionary, used later to compute statistics.
                predictions_dict_4d = util.append_img_4d_dict(predictions_dict_4d,subject,predicted_img)
                # unpad images, then save.
                img_pad_amnt = np.concatenate(pad_amnt[i].numpy(),axis=0)
                predicted_img = util.unpad_img(predicted_img,img_pad_amnt)
                img = util.unpad_img(img,img_pad_amnt)
                # save predicted segmentation
                util.save_img(predicted_img, os.path.join(save_dir_subject,'predicted_segmentation'), 'segmentation_' + fn[i],affine[i].cpu().numpy())
                # save image
                util.save_img(img, os.path.join(save_dir_subject,'image'), 'image_'+fn[i],affine[i].numpy())
    
    # combine the 4D predictions and collect stats, then output to a CSV.
    hdr = ['subj_ID','start_frame','dice','hd','hd_95','assd']
    for key in predictions_dict_4d:
        # make folder to save
        subj_name = key
        save_dir_subject = os.path.join(save_dir,subj_name)
        dice_time = metrics.metric_time_series(predictions_dict_4d[subj_name],metric="dice",voxel_spacing=1)
        hd_time = metrics.metric_time_series(predictions_dict_4d[subj_name],metric="hausdorff",voxel_spacing=3)
        hd95_time = metrics.metric_time_series(predictions_dict_4d[subj_name],metric="hausdorff_95",voxel_spacing=3)
        assd_time = metrics.metric_time_series(predictions_dict_4d[subj_name],metric="assd",voxel_spacing=3)
        # write to csv
        with open(os.path.join(save_dir_subject, 'timeseries_stats.csv'), mode='w') as csv_file:
            csv_file = csv.writer(csv_file, delimiter=',')
            csv_file.writerow(hdr)
            for j in range(0,len(dice_time)):
                csv_file.writerow([subj_name,j,dice_time[j],hd_time[j], hd95_time[j], assd_time[j]])
            csv_file.writerow(['Average',np.mean(dice_time),np.mean(hd_time), np.mean(hd95_time), np.mean(assd_time)])
            csv_file.writerow(['Std',np.std(dice_time),np.std(hd_time), np.std(hd95_time), np.std(assd_time)])


def main(args, model_path, save_dir, data_path, img_dir_name, label_dir_name, subj_folds):
    """
    sets up the model to be evaluated, and evaluates the model on both the train and test
    datasets. 

    Params:
    args: all argparse arguments
    model_path: the path to the saved model file to be loaded in
    save_dir: the path to the directory where results should be saved
    data_path: the path to the directory where data is saved
    img_dir_name: subjdirectory for the images
    label_dir_name: subdirectory for the labels
    img_dir_name: subjdirectory for the images for inference
    label_dir_name: subdirectory for the labels for inference
    subj_folds: a dictionary with the text files for subject folds (train/val/test)
    """
    model = UNet(1)
    pad_factor = PAD_FACTOR
    data = DataLoaderInference(data_path, subj_folds, img_dir_name, label_dir_name, pad_factor=pad_factor, test_only=True, store_images=False)
    test = data.test
    device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu")
    print(device)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    model = model.to(device)
    model, _, epoch, _ = util.load_checkpt(model_path, model)
    
    print("START EVALUATION")
    save_dir_test = os.path.join(save_dir,'test_inference')
    evaluate(model, test, device, save_dir_test)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate saved unet model')
    parser.add_argument('--data_path', dest='data_path', default='~/Documents/data/')
    parser.add_argument('--save_dir', dest='save_dir', default='~/Documents/model-output/')
    parser.add_argument('--subject_id', dest='subject_id', type=int, default=-1) # if -1, evals the entire dataset directory.
    args = parser.parse_args()

    base_dir = os.getcwd()
    model_name = MODEL_NAME
    subj_folds = dict()
    
    args.model_path = os.path.join(base_dir,'model',model_name)
    save_dir = os.path.join(args.save_dir,'model_output')

    img_dir_name = IMG_DIR_NAME
    label_dir_name = LABEL_DIR_NAME
    
    # create a list of subjects based on data in the directory.
    dir_list = util.listdir_nohidden_sort_numerical(args.data_path, list_dir=True, sort_digit=True)
    subj_folds['test'] = dir_list
    # index specific subject if necessary
    if args.subject_id != -1:
        subj_folds['test'] = [subj_folds['test'][args.subject_id]]
    print(subj_folds['test'])
    
    # load the model
    model_path = os.path.abspath(os.path.join(args.model_path, os.pardir)) #need to get parent directory.    
    # inference
    main(args, args.model_path, save_dir, args.data_path, img_dir_name, label_dir_name, subj_folds)
