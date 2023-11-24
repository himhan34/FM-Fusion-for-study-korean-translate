# Move scane folders to the scan_swap folder or reverse
import os, glob 
import shutil

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

def InSwap(scan_dir,swap_dir,TARGET_FOLDERS):
    if os.path.exists(swap_dir):
        shutil.rmtree(swap_dir)
    os.makedirs(swap_dir)
    
    for folder in TARGET_FOLDERS:
        if os.path.exists(os.path.join(scan_dir, folder)):        
            shutil.copytree(os.path.join(scan_dir, folder), os.path.join(swap_dir, folder))

def OutSwap(swap_dir, scan_dir, TARGET_FOLDERS):
    for folder in TARGET_FOLDERS:
        if(os.path.exists(os.path.join(scan_dir, folder))):
            # continue
            shutil.rmtree(os.path.join(scan_dir, folder))
        
        if os.path.exists(os.path.join(swap_dir, folder)):        
            shutil.copytree(os.path.join(swap_dir, folder), os.path.join(scan_dir, folder))
            print('update ', os.path.join(scan_dir, folder))

def CleanNumpyMask(scan_dir, prediction_folder):
    prediction_frames = glob.glob(os.path.join(scan_dir, prediction_folder, '*.npy'))
    for prediction_frame in sorted(prediction_frames):
        os.remove(prediction_frame)
    print('remove {} numpy masks'.format(len(prediction_frames)))

if __name__=="__main__":
    dataroot = '/media/lch/SeagateExp/dataset_/ScanNet' # '/data2/ScanNet'
    split = 'val'
    split_file = 'val_clean'
    swap_folder = os.path.join(dataroot, 'scans_swap')
    TARGET_FOLDERS = ['prediction_no_augment'] # ['color','depth','pose','prediction_no_augment']
    
    if os.path.exists(swap_folder)==False:
        os.makedirs(swap_folder)
    
    scans = read_scans(os.path.join(dataroot, 'splits', split_file + '.txt'))
    
    for scan in scans:
        scan_dir = os.path.join(dataroot, split, scan)
        swap_scan_folder = os.path.join(swap_folder, scan)
        print('-- processing {}'.format(scan))
        # if os.path.exists(os.path.join(scan_dir, 'prediction_no_augment')):
        #     CleanNumpyMask(scan_dir, 'prediction_no_augment')
        # InSwap(scan_dir,swap_scan_folder,TARGET_FOLDERS)
        
        if os.path.exists(swap_scan_folder):
            OutSwap(swap_scan_folder, scan_dir, TARGET_FOLDERS)
        
        print('-- finished {}'.format(scan))
        # break
    



