import os, sys
import subprocess
from subprocess import Popen, PIPE

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

if __name__ == '__main__':
    exe_dir = "build/cpp/IntegrateInstanceMap"
    
    # args
    config_file = 'config/scannet.yaml'
    dataroot = '/data2/ScanNet' # '/media/lch/SeagateExp/dataset_/ScanNet'
    split= 'train'
    # scene_name = 'scene0064_00'        
    output_folder =  '/data2/ScanNetGraph/train' #'/media/lch/SeagateExp/dataset_/ScanNet/output_new'
    
    scans = read_scans(os.path.join(dataroot, 'splits', 'train_micro.txt'))
    
    cmd_list = ['{} --config {} --root {} --output {} --prediction prediction_no_augment --frame_gap 5 --max_frames 500'.format(
        exe_dir,config_file,os.path.join(dataroot, split, scene_name),output_folder) for scene_name in scans]
    
    # exit(0)
    # run
    for scene_name in scans:
        print('processing {}'.format(scene_name))
        
        cmd = "{} --config {} --root {} --output {} --prediction prediction_no_augment --frame_gap 5 --max_frames 5000".format(
            exe_dir, 
            config_file,
            os.path.join(dataroot,split,scene_name),
            output_folder)
        #
        # ret = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE)
        ret = subprocess.run(cmd,
                            stdin=subprocess.PIPE,shell=True)
    
        break

