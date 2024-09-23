import os, sys
import glob
import numpy as np
import subprocess
from scipy.spatial.transform import Rotation as R

def read_all_poses(pose_folder):
    pose_files = glob.glob(os.path.join(pose_folder, '*.txt'))
    pose_map = {}
    for pose_f in sorted(pose_files):
        frame_name = os.path.basename(pose_f).split('.')[0]
        pose = np.loadtxt(pose_f)
        pose_map[frame_name] = pose
    
    return pose_map

def rewrite_pnp_constraints(scene_loop_folder):
    pnp0_folder = os.path.join(scene_loop_folder, 'pnp_0')
    pnp1_folder = os.path.join(scene_loop_folder, 'pnp')
    if os.path.exists(pnp1_folder)==False:
        os.makedirs(pnp1_folder)
    
    pnp0_files = glob.glob(os.path.join(pnp0_folder, '*.txt'))
    
    for file0 in pnp0_files:
        pair = os.path.basename(file0).split('.')[0]
        src_frame = pair.split('_')[0]
        ref_frame = pair.split('_')[1]
        pose = np.loadtxt(file0)
        t = pose[0:3,3]
        rot = pose[0:3,0:3]
        quat = R.from_matrix(rot).as_quat()
        
        with open(os.path.join(pnp1_folder,'{}.txt'.format(src_frame)), 'w') as f:
            f.write('# src_frame ref_frame tx ty tz qx qy qz qw\n')
            f.write('{} {} '.format(src_frame, ref_frame))
            f.write('{:.3f} {:.3f} {:.3f} '.format(t[0], t[1], t[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(quat[0], quat[1], quat[2], quat[3]))
            
            f.close()
        
    

def write_pose2_single_file(pose_map, output_dir):
    header = '# frame x y z qx qy qz qw\n'
    with open(output_dir, 'w') as f:
        f.write(header)
        for frame_name, pose in pose_map.items():
            position = pose[0:3,3]
            rot = pose[0:3,0:3]
            quat = R.from_matrix(rot).as_quat()
            f.write('{} '.format(frame_name))
            f.write('{:.3f} {:.3f} {:.3f} '.format(position[0], position[1], position[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(quat[0], quat[1], quat[2], quat[3]))
        
        f.close()    


if __name__=='__main__':
    ########## CONFIG ##########
    dataroot = '/data2/sgslam'
    EXE_DIR = 'build/cpp/TestHydra'
    # ORB_VOC_DIR = '/home/cliuci/code_ws/DBoW2/vocabulary/ORBvoc.yml'
    ORB_VOC_DIR = '/home/cliuci/code_ws/DBoW2/build/small_voc.yml.gz'
    RUN_LCD = False
    PREPARE_POSE_AVERAGE = False
    
    ############################
    scan_pairs = [
        # ["uc0110_00a", "uc0110_00b"], # similar trajectory
        ["uc0204_00a", "uc0204_00c"],  # opposite trajectory
        ["uc0110_00a", "uc0110_00c"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00b"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00c"],  # opposite trajectory
        ["uc0204_00a", "uc0204_00b"],  # opposite trajectory
        ["uc0111_00a", "uc0111_00b"],
        ["ab0201_03c", "ab0201_03a"], # opposite trajectory
        ["ab0302_00a", "ab0302_00b"],
        ["ab0401_00a", "ab0401_00b"], # opposite trajectory
        ["ab0403_00c", "ab0403_00d"], # opposite trajectory
    ]
    
    summary_time = []
    summary_orb_count = []

    #
    for pair in scan_pairs:
        src_scene=pair[0]
        ref_scene=pair[1]
        print('-------------- {} and {} --------------'.format(src_scene, ref_scene))
    
        scene_output = os.path.join(dataroot, 'output', 'hydra_lcd', '{}-{}'.format(src_scene, ref_scene))
        if os.path.exists(scene_output)==False:
            os.makedirs(scene_output)
            os.makedirs(os.path.join(scene_output, 'viz'))
            os.makedirs(os.path.join(scene_output, 'pnp'))
        os.makedirs(os.path.join(scene_output, 'orb'), exist_ok=True)
        # import shutil
        # shutil.move(os.path.join(scene_output,'pnp'),
        #             os.path.join(scene_output,'pnp_0'))
        
        cmd ='{} '.format(EXE_DIR)
        cmd +='--src_scene {} '.format(os.path.join(dataroot, 'scans', src_scene))
        cmd +='--ref_scene {} '.format(os.path.join(dataroot, 'scans', ref_scene))
        cmd +='--output_folder {} '.format(scene_output)
        cmd +='--distance_range {} '.format(6.0)
        cmd +='--obj_thd {} '.format(0.01)
        cmd +='--bow_thd {} '.format(0.1)
        cmd +='--feat_thd {} '.format(0.85)
        cmd +='--orb_voc_dir {} '.format(ORB_VOC_DIR)
        cmd +='--max_frames {} '.format(2000)
        cmd +='--frame_gap {} '.format(10)
        # cmd +='--verbose true'

        if RUN_LCD:
            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

        with open(os.path.join(scene_output, 'time.txt'), 'r') as f:
            lines = f.readlines()
            eles = lines[1].strip().split(' ')
            eles = [float(e) for e in eles]
            summary_time.append(eles)

        with open(os.path.join(scene_output, 'orb_count.txt'), 'r') as f:
            lines = f.readlines()
            eles = lines[1].strip().split(' ')
            eles = [int(e) for e in eles]
            summary_orb_count += eles

        if PREPARE_POSE_AVERAGE: # Write the files to pg_folder
            print('--- Prepare data structure for pose average')
            src_scene_dir = os.path.join(dataroot,'scans',src_scene)
            ref_scene_dir = os.path.join(dataroot,'scans',ref_scene)
            pg_folder = os.path.join(scene_output, 'pose_graph')
            if os.path.exists(pg_folder)==False:
                os.makedirs(pg_folder)
            src_pose_map = read_all_poses(os.path.join(src_scene_dir, 'pose'))
            ref_pose_map = read_all_poses(os.path.join(ref_scene_dir, 'pose'))
            
            print('Read {} poses from {}'.format(len(src_pose_map), src_scene_dir))
            print('Read {} poses from {}'.format(len(ref_pose_map), ref_scene_dir))
            
            write_pose2_single_file(src_pose_map, os.path.join(pg_folder, 'src_poses.txt'))
            write_pose2_single_file(ref_pose_map, os.path.join(pg_folder, 'ref_poses.txt'))

            rewrite_pnp_constraints(scene_output)

            # break
                    
    print('------ Finished all------')
    if len(summary_time)>0:
        summary_time = np.array(summary_time) # (N, 6)
        summary_time = np.sum(summary_time, axis=0) # (6,)
        header = ['features', 'globalMatch', 'ORB', 'bfMatch', 'PnP']
        mean_time = summary_time[1:] / summary_time[0]
        total_time = mean_time.sum()
        true_frames = summary_time[0]
        
        for i, key in enumerate(header):
            print('{}:{:.3f} ms '.format(key, mean_time[i]))
        print('Total frames: {}, Total time: {:.3f} ms'.format(summary_time[0],
                                                               total_time))
        
    if True:
        print('------- Bandwidth analysis -------')
        frame_number = 1131
        
        summary_orb_count = np.array(summary_orb_count)
        # scale-up due to those compensate frames are missing
        total_orb_count = summary_orb_count.sum()
        total_orb_count = total_orb_count * frame_number / true_frames
        print('Scale orb features by {:.3f}'.format(frame_number/true_frames))
        
        sg_desc_length = 256 + 83 # bow+sg
        orb_desc_length = 32 
        
        sg_bandwidth = frame_number * sg_desc_length * 4
        orb_bandwidth = total_orb_count * orb_desc_length * 4
        
        
        print('SG descriptor: {} frames, {:.3f} KB'.format(frame_number,
                                                           sg_bandwidth/1024))
        print('ORB descriptor: {} features, {:.3f} KB'.format(round(total_orb_count),
                                                              orb_bandwidth/1024))