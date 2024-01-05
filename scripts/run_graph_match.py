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
    exe_dir = "/home/lch/Code_ws/OpensetFusion/build/cpp/GraphMatchNode"
    
    # args
    config_file = '/home/lch/Code_ws/OpensetFusion/config/scannet.yaml'
    dataroot = '/media/lch/SeagateExp/dataset_/ScanNetGraph'
    output_folder = os.path.join(dataroot,'train')
    
    scans = read_scans(os.path.join(dataroot, 'splits', 'train_mini.txt'))
    
    # run
    for scene_name in scans:
        print('processing {}'.format(scene_name))
        
        cmd = "{} --config {} --map_folder {} --src_sequence {} --tar_sequence {}".format(
            exe_dir, 
            config_file,
            output_folder,
            scene_name+'a',
            scene_name+'b')
        
        ret = subprocess.run(cmd,
                            stdin=subprocess.PIPE,shell=True)
    
        # break

