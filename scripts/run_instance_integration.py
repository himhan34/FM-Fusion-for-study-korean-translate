import os, sys
import subprocess
from subprocess import Popen, PIPE
import multiprocessing as mp

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

def process_scene(args):
    exe_dir,config_file,scene_dir,output_folder,subseq = args
    print('processing {}'.format(os.path.basename(scene_dir)))
    data_association = 'data_association_{}.txt'.format(subseq)
    trajectory = 'trajectory_{}.log'.format(subseq)
    
    cmd = "{} --config {} --root {} --output {} --subseq {} --association {} --trajectory {} --prediction prediction_no_augment --frame_gap 5 --max_frames 8000".format(
        exe_dir, 
        config_file,
        scene_dir,
        output_folder,
        subseq,
        data_association,
        trajectory)
    
    #
    ret = subprocess.run(cmd,
                        stdin=subprocess.PIPE,shell=True)
    print('------- finished {} -------'.format(os.path.basename(scene_dir)))

if __name__ == '__main__':
    exe_dir = "build/cpp/IntegrateInstanceMap"
    
    # args
    config_file = 'config/scannet.yaml'
    dataroot = '/data2/ScanNet' # '/media/lch/SeagateExp/dataset_/ScanNet'
    graphroot = '/data2/ScanNetGraph'
    split= 'val'
    output_folder =  os.path.join(graphroot,split) #'/media/lch/SeagateExp/dataset_/ScanNet/output_new'
    subseq_list = ['c','d']
    
    scans = read_scans(os.path.join(dataroot, 'splits', '{}.txt'.format(split)))
    # scans = ['scene0064_00','scene0025_00']
    print('find {} scans to process'.format(len(scans)))
    
    arg_list=[]
    for scene_name in scans:
        for subseq in subseq_list:
            arg_list.append((exe_dir,config_file,os.path.join(dataroot,split,scene_name),output_folder,subseq))

    pool = mp.Pool(processes=32)
    pool.map(process_scene, arg_list)
    # pool.map(process_scene, [(exe_dir,config_file,os.path.join(dataroot,split,scene_name),output_folder,subseq) for scene_name in scans])
    pool.join()
    
    
    # run in seperate process
    # for idx, scene_name in enumerate(scans):
    #     process_scene(exe_dir,config_file,os.path.join(dataroot,split,scene_name),output_folder)
    #     break