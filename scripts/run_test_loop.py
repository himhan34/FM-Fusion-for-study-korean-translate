import os, sys
import subprocess


def read_scan_pairs(dir):
    pairs = []
    with open(dir) as f:
        for line in f.readlines():
            ele0 = line.split(' ')[0].strip()
            ele1 = line.split(' ')[1].strip()
            pairs.append([ele0, ele1])
        f.close()
    return pairs

if __name__ == '__main__':
    exe_dir = '/home/cliuci/code_ws/OpensetFusion/build/cpp/TestLoop'
    
    # args
    config_file = '/home/cliuci/code_ws/OpensetFusion/config/realsense.yaml'
    dataroot = '/data2/sgslam'
    split = 'val'
    split_file = 'val.txt'
    
    # 
    output_folder = os.path.join(dataroot,'output','testloop')
    scan_pairs = read_scan_pairs(os.path.join(dataroot, 'splits', split_file))
    
    # run
    for pair in scan_pairs:
        print('processing {} {}'.format(pair[0], pair[1]))
        src_folder = os.path.join(dataroot, split, pair[0])
        ref_folder = os.path.join(dataroot, split, pair[1])

        # continue
        cmd = "{} --config {} --weights_folder torchscript --ref_scene {} --src_scene {} --output_folder {}".format(
            exe_dir, 
            config_file,
            ref_folder,
            src_folder,
            output_folder)
        
        ret = subprocess.run(cmd,
                            stdin=subprocess.PIPE,shell=True)
    
        # break

