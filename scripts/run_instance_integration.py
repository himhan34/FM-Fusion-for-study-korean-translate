import os, sys
import subprocess

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

if __name__ == '__main__':
    exe_dir = "/home/lch/Code_ws/OpensetFusion/build/cpp/IntegrateInstanceMap"
    
    # args
    config_file = '/home/lch/Code_ws/OpensetFusion/config/scannet.yaml'
    dataroot = '/media/lch/SeagateExp/dataset_/ScanNet'
    split= 'val'
    # scene_name = 'scene0064_00'        
    output_folder = '/media/lch/SeagateExp/dataset_/ScanNet/output_new'
    
    scans = read_scans(os.path.join(dataroot, 'splits', 'val_clean.txt'))
    
    # run
    for scene_name in scans:
        print('processing {}'.format(scene_name))
        
        cmd = "{} --config {} --root {} --output {} --frame_gap 5 --max_frames 5000".format(
            exe_dir, 
            config_file,
            os.path.join(dataroot,split,scene_name),
            output_folder)
        
        ret = subprocess.run(cmd,
                            stdin=subprocess.PIPE,shell=True)
    
        # break

