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
    # args
    config_file = 'config/realsense.yaml'
    dataroot = '/data2/sgslam'
    split_file = 'val_bk.txt'

    # TestLoop
    # split = 'val'
    # output_folder = os.path.join(dataroot,'output','v3+')
    # RUNPROG = 'TestLoop' # 'InstanceMap', 'TestLoop'
    
    ## Instance mapping
    split = 'val'
    output_folder = os.path.join(dataroot,'output','v4')
    RUNPROG = 'TestLoop' # 'InstanceMap', 'TestLoop'

    # 
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder)
    scan_pairs = read_scan_pairs(os.path.join(dataroot, 'splits', split_file))
    
    # run
    for pair in scan_pairs:
        print('processing {} {}'.format(pair[0], pair[1]))
        src_folder = os.path.join(dataroot, split, pair[0])
        ref_folder = os.path.join(dataroot, split, pair[1])
        
        if RUNPROG =='TestLoop':
            exe_dir = 'build/cpp/TestLoop'
            cmd = "{} --config {} --weights_folder torchscript --ref_scene {} --src_scene {} --output_folder {}".format(
                exe_dir, 
                config_file,
                ref_folder,
                src_folder,
                output_folder)
            cmd += " --prune_instance"
            cmd += " --dense_match"
            
        elif RUNPROG == 'InstanceMap':
            exe_dir = 'build/cpp/IntegrateInstanceMap'
            cmd = "{} --config {} --root {} --output {} --prediction prediction_no_augment --verbose 2 --frame_gap 2 --save_frame_gap 20".format(
                exe_dir, 
                config_file,
                src_folder,
                output_folder)
            # cmd += "--save_frame_gap 50"

        ret = subprocess.run(cmd,
                            stdin=subprocess.PIPE,shell=True)    
        # break

