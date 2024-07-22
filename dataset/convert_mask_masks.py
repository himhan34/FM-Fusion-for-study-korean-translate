import os, glob 
import numpy as np
import cv2
from scannetv2.sync_files import read_scans

def convert_scan(scan_prediction_folder):
    prediction_frames = glob.glob(os.path.join(scan_prediction_folder, '*.npy'))
    for prediction_frame in sorted(prediction_frames):
        masks = np.load(prediction_frame) # (K, H, W)
        K = masks.shape[0]
        mask_instances = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8) # (H, W), instance=[1,K]
        for k_ in np.arange(K):
            mask_instances[masks[k_]>1e-3] = k_+1
        cv2.imwrite(prediction_frame.replace('.npy', '.png'), mask_instances)
        # break

if __name__=='__main__':
    
    data_root = '/data2/scenenn' #'/media/lch/SeagateExp/dataset_/ScanNet'
    split='val'
    split_file = 'val_bk'
    PREDICTION_FOLDER = 'prediction_no_augment' #'prediction_forward'

    scans = read_scans(os.path.join(data_root, 'splits', split_file + '.txt'))
    
    # scan = 'scene0025_01'
    for scan in scans:
        print('Converting {}'.format(scan))
        scan_dir = os.path.join(data_root,split, scan)    
        convert_scan(os.path.join(scan_dir, PREDICTION_FOLDER))
        print('finished')
        # break


