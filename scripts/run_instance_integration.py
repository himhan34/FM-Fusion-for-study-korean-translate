import os, sys
import time
import subprocess
import numpy as np
from subprocess import Popen, PIPE
import multiprocessing as mp

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

def instance_integration_process(args):
    exe_dir,config_file,scene_dir,output_folder,subseq = args
    print('processing {}{}, export to {}'.format(os.path.basename(scene_dir),subseq,output_folder))
    # if subseq=='':
    #     data_association = 'data_association.txt'
    #     trajectory = 'trajectory.log'
    # else:
    #     data_association = 'data_association_{}.txt'.format(subseq)
    #     trajectory = 'trajectory_{}.log'.format(subseq)

    cmd = "{} --config {} --root {} --output {} --prediction prediction_no_augment --frame_gap 2 --global_tsdf --max_frames 8000".format(
        exe_dir, 
        config_file,
        scene_dir,
        output_folder)
    
    if subseq!='':
        cmd += ' --subseq {}'.format(subseq)
        cmd += ' --data_association data_association_{}.txt'.format(subseq)
        cmd += ' --trajectory trajectory_{}.log'.format(subseq)

    # print(cmd)
    #
    subprocess.run(cmd,stdin=None,shell=True)
    print('------- finished {} -------'.format(os.path.basename(scene_dir)))

def dense_integration_process(args):
    exe_dir,scene_dir = args
    print('processing {}, export to {}'.format(os.path.basename(scene_dir),output_folder))
    # data_association = 'data_association_{}.txt'.format(subseq)
    # trajectory = 'trajectory_{}.log'.format(subseq)

    cmd = "{} --root {} --intrinsic intrinsic --depth_scale 1000 --max_depth 6.0 --resolution 256 --max_frames 8000 --save_mesh".format(
        exe_dir, 
        scene_dir)
        
    #
    subprocess.run(cmd,stdin=None,shell=True)
    print('------- finished dense integration {} -------'.format(os.path.basename(scene_dir)))

if __name__ == '__main__':
    instance_exe_dir = "build/cpp/IntegrateInstanceMap"
    dense_exe_dir = "build/cpp/IntegrateRGBD"
    DATASET= 'realsense'

    split= 'val'
    if DATASET=='scannet':
        # ScanNet settings
        config_file = 'config/scannet.yaml'
        dataroot = '/data2/ScanNet' # '/media/lch/SeagateExp/dataset_/ScanNet'
        graphroot = '/data2/ScanNetGraph'
        # split= 'val'
        split_file = split
        scene_folder = os.path.join(dataroot,split)
        output_folder =  os.path.join(graphroot,split) 
        # subseq_list = [''] #['c','d']
    elif DATASET=='3rscan':
        # 3rscan settings
        config_file = 'config/rio.yaml'
        dataroot = '/data2/3rscan_raw' # '/media/lch/SeagateExp/dataset_/ScanNet'
        graphroot = dataroot
        split_file= 'serials80' #'scans_ver2_'
        scene_folder = dataroot
        output_folder = os.path.join(dataroot,'output')
    elif DATASET=='realsense':
        # Realsense D515 setting
        config_file = 'config/realsense.yaml'
        dataroot = '/data2/sgslam'
        graphroot = dataroot
        split_file = 'tmp'
        scene_folder = os.path.join(dataroot,'scans')
        output_folder = os.path.join(graphroot,'val_swap')
        
    #
    scans = read_scans(os.path.join(dataroot, 'splits', '{}.txt'.format(split_file)))
    # sample_indices = np.random.choice(len(scans), 32, replace=False)
    # sample_scans = [scans[i] for i in sample_indices]
    sample_scans = scans
    

    arg_list=[]
    arg_list2 = []
    for scene_name in sample_scans:
        print(scene_name)
        # for subseq in subseq_list:
        arg_list.append((instance_exe_dir,config_file,os.path.join(scene_folder,scene_name),output_folder,''))
        arg_list2.append((dense_exe_dir,os.path.join(scene_folder,scene_name)))
        instance_integration_process((instance_exe_dir,config_file,os.path.join(scene_folder,scene_name),output_folder,''))
        # dense_integration_process((dense_exe_dir,os.path.join(scene_folder,scene_name)))
    print('find {} scans and {} sub-sequences to process'.format(len(scans),len(arg_list)))
    
    # print(arg_list2)
    exit(0)
    # process_scene(arg_list[0])
    pool = mp.Pool(processes=4)
    pool.map(instance_integration_process, arg_list)
    pool.close()
    pool.join()
