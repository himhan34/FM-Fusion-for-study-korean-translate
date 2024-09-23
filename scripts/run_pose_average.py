import os, glob
import numpy as np
import subprocess
from scipy.spatial.transform import Rotation as R

def summary_write_pgo(scene_folder, folder_name='pgo', out_file='summary_pgo.txt', src_frames=None, ref_frames=None):
    frame_dirs = glob.glob(os.path.join(scene_folder,folder_name,'*.txt'))
    frame_dirs = sorted(frame_dirs)
    out_file = os.path.join(scene_folder,'pose_graph',out_file)
    ofile = open(out_file,'w')
    ofile.write('# src_frame, srf_frame, tx, ty, tz, qx, qy, qz, qw\n')
    
    count = 0
    for dir in frame_dirs:
        if 'summary' in dir:
            continue
        frame_name = os.path.basename(dir).split('.')[0]
        # frame_id = int(frame_name.split('-')[-1])
        T_ref_src = np.loadtxt(dir)
        if (T_ref_src.shape[0]==0):
            T_ref_src = np.eye(4)
        # if (np.allclose(T_ref_src, np.eye(4))==True):
        #     continue
        if src_frames is not None:
            if frame_name not in src_frames:
                continue
        
        if ref_frames is None:
            ref_frame = frame_name
        else:
            idx = src_frames.index(frame_name)
            ref_frame = ref_frames[idx]
        
        quat = R.from_matrix(T_ref_src[:3,:3]).as_quat()
        translation = T_ref_src[:3,3]
        
        ofile.write('{} {} '.format(frame_name, ref_frame))
        ofile.write('{:.3f} {:.3f} {:.3f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(translation[0],
                                                                                translation[1],
                                                                                translation[2],
                                                                                quat[0],
                                                                                quat[1],
                                                                                quat[2],
                                                                                quat[3]))
        count += 1
    ofile.close()
    print('Summarize {} PGO transformations'.format(count))

def summary_pnp_transformations(scene_folder, overwrite=False):
    frame_dirs = glob.glob(os.path.join(scene_folder,'pnp','*.txt'))
    frame_dirs = sorted(frame_dirs)
    lines = []
    valid_src_frames = []
    ref_frames = []
    
    for dir in frame_dirs:
        if 'summary' in dir:
            continue
        frame_name = os.path.basename(dir).split('.')[0]
        frame_f = open(dir,'r')
        frame_lines = frame_f.readlines()
        if (len(frame_lines)>1):
            lines += frame_lines[1:]
            ref_frame_name = frame_lines[1].split(' ')[1]
            valid_src_frames.append(frame_name)
            ref_frames.append(ref_frame_name)
            
    if overwrite:
        out_filename = 'loop_transformations.txt'
        out_file = os.path.join(scene_folder,'pose_graph',out_filename)
        ofile = open(out_file,'w')
        ofile.write('# src_frame, srf_frame, tx, ty, tz, qx, qy, qz, qw\n')
        for line in lines:
            ofile.write(line)
        ofile.close()
        print('Summarize {} PnP transformations'.format(len(lines)))
    
    return valid_src_frames, ref_frames

def check_valid_transformation_file(dir):
    if os.path.exists(dir)==False:
        return False
    with open(dir,'r') as f:
        lines = f.readlines()
        if len(lines)<2:
            return False
        eles = lines[1].split(' ')
        if len(eles)>7:
            return True
        else:
            return False

def create_hloc_frame_windows(anchor_id, all_frames, window_size, pnp_folder):
    
    # anchor_id = int(anchor_frame.split('-')[-1])
    window_frames = []
    window_frames_str = ''
    SAMPLING_GAP = 10

    for i in range(1, window_size):
        frame_id = anchor_id - i * SAMPLING_GAP
        frame_name = 'frame-{:06d}'.format(frame_id)
        frame_dir = os.path.join(pnp_folder, '{}.txt'.format(frame_name))
        if frame_dir in all_frames:
            window_frames.append(frame_name)
            window_frames_str += '{}'.format(frame_name)
            
    # print(anchor_id, window_frames)
    
    return window_frames_str 

def create_hydra_frame_windows(anchor_id, all_query_frames, all_frame_pairs, window_size):
    
    SAMPLING_GAP = 10
    window_frame_pairs = []
    window_frame_pairs_str = ''
    
    for i in range(1, window_size):
        frame_id = anchor_id - i * SAMPLING_GAP
        frame_name = 'frame-{:06d}'.format(frame_id)
        if frame_name not in all_query_frames:
            continue
        query_idx = all_query_frames.index(frame_name)
        frame_pair = all_frame_pairs[query_idx]
        window_frame_pairs.append(frame_pair)
        window_frame_pairs_str += '{} '.format(frame_pair)
    
    print('Anchor frame:', anchor_id)
    print('Window pairs:', window_frame_pairs_str)
    
    return window_frame_pairs_str
   

def create_our_frame_windows(anchor_frame, all_frames, window_size):
    
    anchor_id = int(anchor_frame.split('-')[-1])
    anchor_index = all_frames.index(anchor_frame)
    SAMPLING_GAP = 10
    window_frames = []
    window_frames_str = ''

    for i in range(1, window_size):
        frame_index = anchor_index - i
        if frame_index<0:break
        frame_name = all_frames[frame_index]
        frame_id = int(frame_name.split('-')[-1])
        if frame_id< (anchor_id-window_size*SAMPLING_GAP):
            break
        
        window_frames.append(frame_name)
        window_frames_str += '{}'.format(frame_name)
        
    return window_frames_str

def read_pose_file(dir):
    loop_transformations = []
    
    with open(dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line:
                continue
            eles = line.split(' ')
            src_frame = eles[0]
            ref_frame = eles[1]
            t = np.array([float(eles[2]), float(eles[3]), float(eles[4])])
            q = np.array([float(eles[5]), float(eles[6]), float(eles[7]), float(eles[8])])
            rot = R.from_quat(q).as_matrix()
            T_ref_src = np.eye(4)
            T_ref_src[:3,:3] = rot
            T_ref_src[:3,3] = t
            loop_transformations.append({'src':src_frame, 'ref':ref_frame, 
                                         'pose':T_ref_src})
        f.close()
        return loop_transformations

def write_pose_file(loop_dict, output_dir):
    with open(output_dir,'w') as f:
        f.write('# src_frame ref_frame tx ty tz qx qy qz qw\n')
        for loop in loop_dict:
            src_frame = loop['src']
            ref_frame = loop['ref']
            pose = loop['pose']
            t = pose[:3,3]
            rot = pose[:3,:3]
            quat = R.from_matrix(rot).as_quat()
            f.write('{} {} '.format(src_frame, ref_frame))
            f.write('{:.3f} {:.3f} {:.3f} '.format(t[0], t[1], t[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(quat[0], quat[1], quat[2], quat[3]))
        f.close()

if __name__ == "__main__":
    # Data structure
    # |-- SRC_SCENE-REF_SCENE
    # |   |-- pnp: observation list [T_c1_c0]
    # |   |-- pose_graph
    # |     |-- src_pose.txt
    # |     |-- ref_pose.txt
    # |   |-- pose_average
    # |     |-- frame-000001.txt: T_ref_src
    # |     |-- ...
    # |   |-- pnp_averaged: loop constrains after averaging
    
    
    #### args ####
    dataroot = "/data2/sgslam"
    sfm_datroot = "/data2/sfm/multi_agent"
    hydra_result_folder = os.path.join(dataroot, "output", "hydra_lcd")
    gt_folder = os.path.join(dataroot, "gt")
    our_result_folder = os.path.join(dataroot, "output", "v9")
    ##############

    # cfg_file = "config/realsense.yaml"
    # export_folder = os.path.join(dataroot, "output", "offline_register_quatro")
    scan_pairs = [
        ["uc0110_00a", "uc0110_00b"],
        ["uc0110_00a", "uc0110_00c"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00b"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00c"],
        ["uc0204_00a", "uc0204_00b"],  # opposite trajectory
        ["uc0204_00a", "uc0204_00c"],  # opposite trajectory
        ["uc0111_00a", "uc0111_00b"],
        ["ab0201_03c", "ab0201_03a"], # opposite trajectory
        ["ab0302_00a", "ab0302_00b"],
        ["ab0401_00a", "ab0401_00b"], # opposite trajectory
        ["ab0403_00c", "ab0403_00d"], # opposite trajectory
    ]
    RUN_PGO = False
    RUN_POSE_AVG = True
    CREATE_PNP_AVERAGED = True
    POSE_AVERAGE_WINDOW = 3
    METHOD = 'hydra' # hydra, hloc or ours

    for pair in scan_pairs:
        src_scene_folder = os.path.join(dataroot,'scans',pair[0])
        ref_scene_folder = os.path.join(dataroot,'scans',pair[1])
        sfm_scene_folder = os.path.join(sfm_datroot,'{}-{}'.format(pair[0],pair[1]))
        hydra_scene_folder = os.path.join(hydra_result_folder,'{}-{}'.format(pair[0],pair[1]))
        
        if METHOD=='hloc':
            lcd_frames = glob.glob(os.path.join(sfm_scene_folder,'pnp','*.txt'))
            scene_output_folder = sfm_scene_folder
        elif METHOD=='hydra':
            lcd_frames = glob.glob(os.path.join(hydra_scene_folder,'pnp','*.txt'))
            scene_output_folder = hydra_scene_folder
        else:
            assert METHOD=='ours'
            lcd_frames = glob.glob(os.path.join(our_result_folder,pair[0],pair[1],'*.txt'))
            lcd_frames = [frame_dir for frame_dir in lcd_frames if len(os.path.basename(frame_dir))==16]
        
        lcd_frames = sorted(lcd_frames)
        print('------- Eval {}-{} {} frames -------'.format(pair[0],pair[1],len(lcd_frames)))
        
        gt_T_ref_src = np.loadtxt(os.path.join(gt_folder,'{}-{}.txt'.format(pair[0],pair[1])))
        gt_orientation = R.from_matrix(gt_T_ref_src[:3,:3]).as_euler('zyx',degrees=True)
        gt_translation = gt_T_ref_src[:3,3]
        
        np.savetxt(os.path.join(sfm_scene_folder,'pose_graph','gt.txt'), gt_T_ref_src)
        
        output_folder = os.path.join(sfm_scene_folder,'pgo')
        if os.path.exists(output_folder)==False:
            os.makedirs(output_folder)
        
        if RUN_PGO:
            exe_dir = "build/cpp/testTwoAgentsPGO"
            for frame in lcd_frames:
                frame_id = os.path.basename(frame).split('.')[0].split('-')[-1]
                frame_id = int(frame_id)
            
                cmd = '{} --scene_folder {} --frame_id {} --mode 1'.format(exe_dir,
                                                                sfm_scene_folder,
                                                                frame_id)

                subprocess.run(cmd, shell=True)
                print('Gt orientation:\n ', gt_orientation)
                print('Gt translation:\n ', gt_translation)
                print('Done: {}'.format(frame))
            
        if RUN_POSE_AVG:
            exe_dir = "build/cpp/robustPoseAvg"
            
            if METHOD=='hloc':
                input_folder = sfm_scene_folder
                pos_ave_folder = os.path.join(sfm_scene_folder,'pose_average')
                if os.path.exists(pos_ave_folder)==False:
                    os.makedirs(pos_ave_folder)
            elif METHOD=='hydra':
                input_folder = hydra_scene_folder
                pos_ave_folder = os.path.join(input_folder,'pose_average')
                if os.path.exists(pos_ave_folder)==False:
                    os.makedirs(pos_ave_folder)
            else:
                assert METHOD=='ours'
                input_folder = os.path.join(our_result_folder, pair[0], pair[1])
                pos_ave_folder = os.path.join(our_result_folder,pair[0],'pose_average_{}'.format(pair[1]))
                if os.path.exists(pos_ave_folder)==False:
                    os.makedirs(pos_ave_folder)

            for frame in lcd_frames:
                frame_name = os.path.basename(frame).split('.')[0]
                if METHOD=='hydra':
                    frame_name = frame_name.split('_')[0].strip()
                frame_id = int(frame_name.split('-')[-1])
                valid_frame = check_valid_transformation_file(frame)
                
                print('---', frame_name)
                if METHOD=='hloc' and valid_frame==False:
                    continue    
                            
                cmd = '{} --input_folder {} --output_folder {} '.format(exe_dir,
                                                                input_folder,
                                                                pos_ave_folder)

                cmd += '--anchor_frame {} '.format(frame_name)
                if (METHOD=='hloc'):
                    window_frames = create_hloc_frame_windows(frame_id, 
                                                            lcd_frames, 
                                                            POSE_AVERAGE_WINDOW,
                                                            os.path.join(sfm_scene_folder,'pnp'))                    
                    cmd += '--sfm '
                elif (METHOD=='hydra'):
                    window_frames = create_hloc_frame_windows(frame_id,
                                                              lcd_frames,
                                                              POSE_AVERAGE_WINDOW,
                                                              os.path.join(hydra_scene_folder,'pnp'))
                    cmd += '--sfm '
                else:
                    window_frames = create_our_frame_windows(frame_name,
                                                             [os.path.basename(frame_dir).split('.')[0] for frame_dir in lcd_frames],
                                                             POSE_AVERAGE_WINDOW)
                
                if window_frames!='':
                    cmd += '--frame_list {}'.format(window_frames)

                if window_frames=='':
                    continue
                print(cmd)
                
                subprocess.run(cmd, shell=True)
                # print('Pose Average: {}'.format(frame))
                # break
        
        if CREATE_PNP_AVERAGED:
            pnp_files = glob.glob(os.path.join(scene_output_folder,'pnp','*.txt'))
            pnp_out_folder = os.path.join(scene_output_folder,'pnp_averaged_{}'.format(POSE_AVERAGE_WINDOW))
            if os.path.exists(pnp_out_folder)==False:
                os.makedirs(pnp_out_folder)
            # pa_files  = glob.glob(os.path.join(scene_output_folder,'pose_average','*.txt'))
            
            for pnp_f in pnp_files:
                query_frame_name = os.path.basename(pnp_f).split('.')[0]
                pa_file = os.path.join(scene_output_folder,'pose_average','{}.txt'.format(query_frame_name))
                pnp_dict = read_pose_file(pnp_f)[0]
                if os.path.exists(pa_file):
                    T_ref_c1 = np.loadtxt(os.path.join(ref_scene_folder,'pose','{}.txt'.format(pnp_dict['ref'])))
                    T_src_c0 = np.loadtxt(os.path.join(src_scene_folder,'pose','{}.txt'.format(query_frame_name)))
                    T_ref_src = np.loadtxt(pa_file)
                    T_c1_c0 = np.dot(np.linalg.inv(T_ref_c1), np.dot(T_ref_src, T_src_c0))
                    write_pose_file([{'src':query_frame_name, 'ref':pnp_dict['ref'], 'pose':T_c1_c0}],
                                    os.path.join(pnp_out_folder, os.path.basename(pnp_f)))
                    
                else: # Copy the original pnp file
                    import shutil
                    shutil.copyfile(pnp_f, 
                                    os.path.join(pnp_out_folder, os.path.basename(pnp_f)))
        
        if METHOD=='hloc':
            src_frames, ref_frames = summary_pnp_transformations(sfm_scene_folder, overwrite=False)
            summary_write_pgo(sfm_scene_folder, 'pose_average', 
                              'summary_pose_average_pa5.txt', 
                              src_frames, ref_frames)
