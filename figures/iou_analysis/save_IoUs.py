import os
import time
import skimage.morphology
import torch
import skimage
from tqdm import tqdm
import argparse
import numpy as np
# from cellpose import io, models, train
# from typing import Union, Tuple, Optional
# from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation 
# from huggingface_hub import hf_hub_download
# from mndino import mnmodel, evaluation, mnds


# def run_automatic_instance_segmentation(
#         image: np.ndarray,
#         ndim: int,
#         checkpoint_path: Optional[Union[os.PathLike, str]] = None,
#         model_type: str = "vit_b_lm",
#         device: Optional[Union[str, torch.device]] = None,
#         tile_shape: Optional[Tuple[int, int]] = None,
#         halo: Optional[Tuple[int, int]] = None,
#     ):
#         """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

#         NOTE: AIS is supported only for `µsam` models.

#         Args:
#             image: The input image.
#             ndim: The number of dimensions for the input data.
#             checkpoint_path: The path to stored checkpoints.
#             model_type: The choice of the `µsam` model.
#             device: The device to run the model inference.
#             tile_shape: The tile shape for tiling-based segmentation.
#             halo: The overlap shape on each side per tile for stitching the segmented tiles.

#         Returns:
#             The instance segmentation.
#         """
#         # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
#         predictor, segmenter = get_predictor_and_segmenter(
#             model_type=model_type,  # choice of the Segment Anything model
#             checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
#             device=device,  # the device to run the model inference.
#             # amg='ais',  # set the automatic segmentation mode to AIS.
#             is_tiled=(tile_shape is not None),  # whether to run automatic segmentation with tiling.
#         )

#         # Step 2: Get the instance segmentation for the given image.
#         prediction = automatic_instance_segmentation(
#             predictor=predictor,  # the predictor for the Segment Anything model.
#             segmenter=segmenter,  # the segmenter class responsible for generating predictions.
#             input_path=image,  # the filepath to image or the input array for automatic segmentation.
#             ndim=ndim,  # the number of input dimensions.
#             tile_shape=tile_shape,  # the tile shape for tiling-based prediction.
#             halo=halo,  # the overlap shape for tiling-based prediction.
#         )

#         return prediction
    
def read_image(image_path, scale=1.0):
    im = skimage.io.imread(image_path)
    if scale != 1.:
        im = skimage.transform.rescale(im, scale)
    return im

def compare_two_labels(label_model, label_gt, return_IoU_matrix, debug=False):
    
    # get number of detected nuclei
    nb_nuclei_gt = np.max(label_gt)
    nb_nuclei_model = np.max(label_model)
    
    # catch the case of an empty picture in model and gt
    if nb_nuclei_gt == 0 and nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, 0, 1, np.empty(0)]     
        else:
            return [0, 0, 1]
    
    # catch the case of empty picture in model
    if nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, nb_nuclei_gt, 0, np.empty(0)]     
        else:
            return [0, nb_nuclei_gt, 0]
    
    # catch the case of empty picture in gt
    if nb_nuclei_gt == 0:
        if(return_IoU_matrix):
            return [nb_nuclei_model, 0, 0, np.empty(0)]     
        else:
            return [nb_nuclei_model, 0, 0]
    
    # build IoU matrix
    IoUs = np.full((nb_nuclei_gt, nb_nuclei_model), -1.0, dtype = np.float32)

    # calculate IoU for each nucleus index_gt in GT and nucleus index_pred in prediction    
    # TODO improve runtime of this algorithm
    for index_gt in range(1,nb_nuclei_gt+1):

        nucleus_gt = label_gt == index_gt
        number_gt = np.sum(nucleus_gt)

        for index_model in range(1,nb_nuclei_model+1):
            
            if debug:
                print(index_gt, "/", index_model)
            
            nucleus_model = label_model == index_model 
            number_model = np.sum(nucleus_model)
            
            same_and_1 = np.sum((nucleus_gt == nucleus_model) * nucleus_gt)
            
            IoUs[index_gt-1,index_model-1] = same_and_1 / (number_gt + number_model - same_and_1)
    
    # get matches and errors
    detection_map = (IoUs > 0.5)
    nb_matches = np.sum(detection_map)

    detection_rate = IoUs * detection_map
    
    nb_overdetection = nb_nuclei_model - nb_matches
    nb_underdetection = nb_nuclei_gt - nb_matches
    
    mean_IoU = np.mean(np.sum(detection_rate, axis = 1))
    
    if(return_IoU_matrix):
        result = [nb_overdetection, nb_underdetection, mean_IoU, IoUs]
    else:
        result = [nb_overdetection, nb_underdetection, mean_IoU]
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save IoU matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--path', type=str, help='mnDINO dataset path')
    parser.add_argument('--save_path', type=str, help='Path to save IoU matrices')
    # parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    # parser.add_argument('--model_type', default='dino', choices=['dino', 'unet'], help='Choose model type over DINO-based architecture or simple 2D Unet.')
    parser.add_argument('--step', type=int, default=32, help='Step size of prediction box, larger value will decrease inference time but harm accuracy.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of crops from one image that are predicted simultaneously.')
    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Probability threshold to classify if each pixel is micronucleus.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for aligning microscopy magnification.')
    return parser.parse_args()

def main():
    # REPO_ID = "CaicedoLab/mnDINO"
    # FILENAME = "mnDINO_v1.pth"
    # model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    args = parse_args()
    
    # PATH = os.path.join(args.path, 'test/images')
    PATH = args.path
        
    # Load Model
    device = f"cuda" if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    # model = mnmodel.MicronucleiModel(device=device)
    # model.load(model_path)
    
    # model_type='cpsam_v2'
    # model = models.CellposeModel(pretrained_model=model_type, device=torch.device(device))
    # model_path = '/hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/baselines/cpsam_fintuned_predictions/models/cellpose_finetuned.pth'
    # model = models.CellposeModel(gpu=True, 
    #                              pretrained_model=model_path, 
    #                              device=torch.device(device))
    # model_type = "vit_b_lm"
    # CKPT_PATH = os.path.join('/scr/yren/microsam_predictions/', 'models', 'checkpoints')
    # checkpoint_name = "sam_finetuned_all"
    # best_checkpoint = os.path.join(CKPT_PATH, checkpoint_name, 'best.pt')
    
    files = os.listdir(PATH)
    filelist = sorted([file for file in files if not file.startswith('.')])
    filelist = sorted([file for file in files if file.endswith('.npy')])
    annot_files = sorted(filelist.copy()) # npy files for UNet
    
    # gts_files = sorted(os.listdir(PATH.replace('images', 'mn_masks')))
    # imgs = [io.imread(os.path.join(PATH, f)) for f in filelist]
    # gts = [io.imread(os.path.join(PATH.replace('images', 'mn_masks'), f)) for f in gts_files]
    
    # pretrained_model_path = '/hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/baselines/cp3_fintuned_predictions/models/cellpose_finetuned.pth'
    # model = models.CellposeModel(gpu=True, 
    #                                      pretrained_model=str(pretrained_model_path),
    #                                      device=torch.device(device))
    # model = mnmodel.MicronucleiModel(
    #     device=device,
    #     data_dir=PATH,
    #     model_type='unet'
    # )
    # model_name = f'mnDINO_unet.pth'
    # models_dir = os.path.join(PATH, 'models')
    # # model_name = f'mnDINO_{INDEX}.pth'
    # model.load(model_path=os.path.join(models_dir, model_name))
    MICRON_AREA_THRESHOLD = 300
    # DIAMETER = model.diam_labels.copy()
    # channels = [0,0]
    for i in tqdm(range(len(annot_files))):
        test_file = annot_files[i]
        prediction = np.load(os.path.join(PATH, test_file))
        # im = skimage.io.imread(os.path.join(PATH, test_file))
        # im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
        # H, W = im.shape
        # if (H > 1024) and (W > 1024):
        #     prediction = run_automatic_instance_segmentation(
        #         image=im,
        #         ndim=2,
        #         checkpoint_path=best_checkpoint,
        #         model_type=model_type,
        #         device=device,
        #         tile_shape=(1024, 1024), 
        #         halo=(256, 256) # step size
        #     )
        # else:
        #     prediction = run_automatic_instance_segmentation(
        #         image=im,
        #         ndim=2,
        #         checkpoint_path=best_checkpoint,
        #         model_type=model_type,
        #         device=device,
        #         halo=(256, 256) # step size, probably smaller than image size
        #     )
        # prediction = model.eval(im, channels=channels, diameter=DIAMETER)[0]
        # prediction = np.asarray(prediction, dtype='uint16')
        # labels = skimage.morphology.label(prediction)
        micro_mask = skimage.morphology.label(prediction)
        
        # micron_labels = []
        # for i in range(1, len(np.unique(labels))):
        #     area = np.sum(labels == i)
        #     if area < MICRON_AREA_THRESHOLD:
        #         micron_labels.append(i)
                
        # micro_mask = np.zeros_like(labels)
        # for i in micron_labels:
        #     micro_mask += (labels == i)
            
        gt_path = '/scr/data/annotated_mn_datasets/test/mn_masks/'
        mn_gt = read_image(os.path.join(gt_path, test_file.replace('._probabilities.npy', '.png')), scale=args.scale)
        mn_gt = mn_gt > 0
        mn_gt = skimage.morphology.label(mn_gt)
        

        labeled_mn = skimage.morphology.label(micro_mask)
        labeled_mn = np.asarray(labeled_mn, dtype='uint16')
        nb_overdetection, nb_underdetection, mean_IoU, IoUs = compare_two_labels(labeled_mn, mn_gt, True, False)
        
        # Save IoUs
        imid = test_file.split('._probabilities.npy')[0] # UNet & MNFinder
        np.save(os.path.join(args.save_path, imid + '.npy'), IoUs)
        
    
    print(f'IoU matrices saved successfully to {args.save_path}')
    
    

if __name__ == '__main__':
    main()