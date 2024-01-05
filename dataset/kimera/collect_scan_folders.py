import os,glob,sys
import shutil
sys.path.append('/home/cliuci/code_ws/OpensetFusion/python')
import fuse_detection

if __name__=='__main__':
    dataroot = '/data2/ScanNet'
    target_root = '/data2/ScanNet/target_scans'
    split = 'val'
    
    scans = fuse_detection.read_scans(os.path.join(dataroot,'splits','val_clean.txt'))

    print('collecting {} scans'.format(len(scans)))

    for scan in scans:
        source_folder = os.path.join(dataroot,split, scan)
        target_folder = os.path.join(target_root, scan)
        if os.path.exists(target_folder)==False:
            os.mkdir(target_folder)

        # shutil.copytree(os.path.join(source_folder, 'color'), os.path.join(target_folder, 'color'))
        # shutil.copytree(os.path.join(source_folder, 'depth'), os.path.join(target_folder, 'depth'))
        # shutil.copytree(os.path.join(source_folder, 'pose'), os.path.join(target_folder, 'pose'))
        # shutil.copytree(os.path.join(source_folder, 'pred_maskrcnn_color_rf'), os.path.join(target_folder, 'pred_maskrcnn_color_rf'))
        shutil.rmtree(os.path.join(target_folder, 'pred_gsam_color'))
        # print(target_folder)
        print('remove to {}/pred_maskrcnn_color'.format(target_folder))
        # break