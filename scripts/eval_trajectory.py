import os, glob
import numpy as np
from scipy.spatial.transform import Rotation as R 
import argparse

from eval_loop import read_match_centroid_result
from run_test_loop import read_scan_pairs

def relative_pos_error(pred_pos:np.ndarray, gt_pos:np.ndarray):
    err = pred_pos - gt_pos # (N,3)
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2] # (N,)
    if np.isnan(norm2).any():
        print("RPE has nan")
    return np.sqrt(norm2)

def position_ate(pred_pos:np.ndarray, gt_pos:np.ndarray):
    err = pred_pos - gt_pos # (N,3)
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2] # (N,)
    if np.isnan(norm2).any():
        print("ATE_POS has nan")

    return np.sqrt(np.mean(norm2)), np.sqrt(norm2) # sqrt(mean(||pred-gt||^2))

def load_camera_tag_poses(folder_dir, inverse=False):
    '''
        T_camera_tag or T_tag_camera
    '''
    frames = glob.glob(folder_dir+'/*.txt')
    camera_tag_poses = {}
    
    for frame_dir in frames:
        if 'frame' in frame_dir:
            frame_name = os.path.basename(frame_dir).split('.')[0]
            T_cam_tag = np.loadtxt(frame_dir)
            T_cam_tag[3,:3] = 0
            if inverse:
                T_cam_tag = np.linalg.inv(T_cam_tag)
            camera_tag_poses[frame_name] = T_cam_tag

    return camera_tag_poses

def load_vins_tag_poses(folder_dir):
    ''' T_vins_tag, vins is the local coordinate vins is initialized. '''
    tag_pose_dir = os.path.join(folder_dir,'T_vins_tag.txt')
    if os.path.exists(tag_pose_dir):
        T_vins_tag = np.loadtxt(tag_pose_dir)
        return T_vins_tag
    else:
        return None

def load_pose_file(pose_file):
    frames = []
    poses  = []
    frame_poses = {}
    with open(pose_file,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            eles = line.strip().split(' ')
            frame = eles[0]
            tx, ty, tz = float(eles[1]), float(eles[2]), float(eles[3])
            qx, qy, qz, qw = float(eles[4]), float(eles[5]), float(eles[6]), float(eles[7])
            quat = R.from_quat([qx, qy, qz, qw])
            T_c1_c0 = np.eye(4)
            T_c1_c0[:3,:3] = quat.as_matrix()
            T_c1_c0[:3,3] = [tx, ty, tz]
            frames.append(frame)
            poses.append(T_c1_c0)
            frame_poses[frame] = T_c1_c0
            
        f.close()
        print('Load {} poses from {}'.format(len(frames), pose_file))
        return frame_poses

def write_scene_result(frames, ate_array, output_dir, ate_threshold=0.2):
    with open(output_dir, 'w') as f:
        f.write('# frame ate\n')
        count_tp = 0
        for i, frame in enumerate(frames):
            f.write('{} {:.3f}\n'.format(frame, ate_array[i]))
            if ate_array[i]<ate_threshold:
                count_tp += 1
                
        f.write('TP: {}/{} frames, threshold: {}m \n'.format(count_tp, len(frames),ate_threshold))
        f.close()
        
def compute_success_rate(frames, pes, threshold=0.2):
    print('Compute success rate')
    N = len(frames)
    frames_success_rate = []
    frame_numbers = []
    
    for i in range(N):
        frame_idx = int(frames[i][6:12])
        frame_number = frame_idx
        frame_numbers.append(frame_number)
        
        # count_tp = pes[:i+1]<threshold
        # success_rate = count_tp.sum()/(i+1)
    
        # frames_success_rate.append([frame_number, success_rate])
    
    frame_numbers = np.array(frame_numbers)
    tp_masks = pes<threshold
    frames_success_rate = np.concatenate([frame_numbers.reshape(-1,1), tp_masks.reshape(-1,1), pes.reshape(-1,1)], axis=1)
    
    # sorted by frame number
    # frames_success_rate = frames_success_rate[frames_success_rate[:,0].argsort()]
    
    return frames_success_rate
    
def compute_iou(out_result_folder, src_scene, ref_scene, src_frames, gt_pose, SFM_MODE=False):
    import open3d as o3d
    from eval_loop import read_frames_map, find_closet_index, read_match_centroid_result, compute_cloud_overlap
    
    result_folder = os.path.join(out_result_folder, src_scene, ref_scene)
    ious = []
    
    ref_maps = read_frames_map(os.path.join(out_result_folder, ref_scene, 'fakeScene'))
    if SFM_MODE: # calibrate src frames
        src_maps = read_frames_map(result_folder, '_centroids.txt')
        cal_src_frames = []
        for src_frame in src_frames:
            src_frame_id = int(src_frame[6:12])
            
            src_centroid_dir = find_closet_index(src_maps["indices"], src_maps["dirs"], src_frame_id)
            # cal_src_frame = src_centroid_dir.replace('_centroids.txt', '.txt')
            cal_src_frame = os.path.basename(src_centroid_dir)[:12]
            cal_src_frames.append(cal_src_frame)
        src_frames = cal_src_frames
    
    for src_frame in src_frames:
        frame_dir = os.path.join(result_folder, src_frame+'.txt')
        if os.path.exists(frame_dir)==False:
            print('No src frame found in ', frame_dir)
            ious.append(0.0)
            continue
        
        _, _, _, ref_frame_timestamp = read_match_centroid_result(frame_dir)
        assert ref_frame_timestamp is not None
        ref_frame_id = int((ref_frame_timestamp - 12000.0) / 0.1)
        # print('{}-{}'.format(src_frame, ref_frame_id))

        ref_map_dir = find_closet_index(
            ref_maps["indices"], ref_maps["dirs"], ref_frame_id
        )
        
        src_pcd = o3d.io.read_point_cloud(os.path.join(result_folder, src_frame+'_src.ply'))
        if src_pcd.has_points()==False: 
            print('src pcd empty in ', src_frame)
            ious.append(0.0)
            continue
        ref_pcd = o3d.io.read_point_cloud(ref_map_dir)
        assert src_pcd.has_points() and ref_pcd.has_points()
        src_pcd.transform(gt_pose)
        iou, _= compute_cloud_overlap(src_pcd, ref_pcd, 0.2)
        ious.append(iou)
    
    ious = np.array(ious)
    return ious

class TrajectoryAnalysis:
    def __init__(self, scene_dir, gt_source, rpe_threshold) -> None:
        self.src_scene = os.path.basename(scene_dir)
        self.src_poses = self.load_camera_poses(os.path.join(scene_dir,'pose')) # {frame_name: T_src_c}
        self.pred_poses = {} # pred poses, {frame_name: T_ref_c}
        self.gt_poses = {} # gt poses, {frame_name: T_ref_c}
        self.gt_source = gt_source # icp, tag or optitrack
        self.rpe_threshold = rpe_threshold
        
    def get_pred_poses_list(self, sample_ratio=0.2):
        pred_poses_list = []
        sample_indices = np.random.choice(len(self.pred_poses), int(len(self.pred_poses)*sample_ratio), replace=False)
        
        for i, (frame, pose) in enumerate(self.pred_poses.items()):
            if i in sample_indices:
                pred_poses_list.append(pose)
        return np.array(pred_poses_list)

    def get_gt_poses_listt(self, sample_ratio=0.2):
        gt_poses_list = []
        sample_indices = np.random.choice(len(self.gt_poses), int(len(self.gt_poses)*sample_ratio), replace=False)
        
        for i, (frame, pose) in enumerate(self.gt_poses.items()):
            if i in sample_indices:
                gt_poses_list.append(pose)
        return np.array(gt_poses_list)
            
    def load_camera_poses(self, dir):
        frames_pose_dirs = glob.glob(dir+'/*.txt')
        frames_pose_dirs = sorted(frames_pose_dirs)
        poses = {}
        for frame in frames_pose_dirs:
            frame_name = os.path.basename(frame).split('.')[0]
            pose = np.loadtxt(frame)
            poses[frame_name] = pose
            
        print('Load {} camera poses in {}'.format(len(poses), self.src_scene))
        return poses

    def update_aligned_poses(self, output_folder, ref_scene):
        ''' Update pred poses in reference frame. Only the last loop frame is used. '''
        src_scene_output = os.path.join(output_folder, self.src_scene)
        loop_frames = glob.glob(src_scene_output+'/'+ref_scene+'/*.txt')
        loop_frames = sorted(loop_frames)
        if len(loop_frames)==0:
            print('No loop frames found in ', src_scene_output)
            return None

        T_ref_src, _, _, _ = read_match_centroid_result(loop_frames[-2]) # predicted T_ref_src
        
        for frame_name, pose in self.src_poses.items():
            self.pred_poses[frame_name] = T_ref_src @ pose # T_ref_c = T_ref_src @ T_src_c
        print('update pred pose graph using: ', loop_frames[-1])
        
    def update_ours_framewise_poses(self, output_folder, ref_scene, compensate=True):
        ''' Each loop frame is used 
            @param compensate: if true, set the poses from frame-000000 to the beginning frame to be identity matrix
        '''
        src_scene_output = os.path.join(output_folder, self.src_scene)
        loop_frames = glob.glob(src_scene_output+'/'+ref_scene+'/*.txt')
        loop_frames = sorted(loop_frames)
        if len(loop_frames)==0:
            print('No loop frames found in ', src_scene_output)
            return None
        for frame in loop_frames:
            if 'centroids' in frame or 'cmatches' in frame: continue
            frame_name = os.path.basename(frame).split('.')[0]
            T_ref_src, _, _ ,_ = read_match_centroid_result(frame)
            self.pred_poses[frame_name] = T_ref_src @ self.src_poses[frame_name] # T_ref_c
            # self.pred_poses[frame_name] = T_ref_src
        
        if compensate:
            beginning_frame = loop_frames[0]
            beginning_frame_name = os.path.basename(beginning_frame).split('.')[0]
            beginning_frame_id = int(beginning_frame_name[6:12])
            
            idx = beginning_frame_id
            while(idx>9):
                frame_name = 'frame-{:06d}'.format(idx)
                self.pred_poses[frame_name] = np.eye(4)
                idx -=10
            
        print('update pred pose graph using {} loop frames'.format(len(self.pred_poses)))
        
    def update_our_average_poses(self, pose_Average_folder, compensate=True):
        loop_frames = glob.glob(pose_Average_folder+'/*.txt')
        loop_frames = sorted(loop_frames)
        
        if len(loop_frames)==0:
            print('No loop frames found in ', pose_Average_folder)
            return None
        
        for frame_dir in loop_frames:
            frame_name = os.path.basename(frame_dir).split('.')[0]
            T_ref_src = np.loadtxt(frame_dir)
            self.pred_poses[frame_name] = T_ref_src @ self.src_poses[frame_name] # T_ref_c = T_ref_src @ T_src_c
        
        if compensate:
            beginning_frame = loop_frames[0]
            beginning_frame_name = os.path.basename(beginning_frame).split('.')[0]
            beginning_frame_id = int(beginning_frame_name[6:12])
            idx = beginning_frame_id-10
            while(idx>9):
                frame_name = 'frame-{:06d}'.format(idx)
                self.pred_poses[frame_name] = np.eye(4)
                idx -=10
                
        print('update {} averaged poses.'.format(len(self.pred_poses)))

    def update_hloc_framewise_poses(self, pg_folder, file_name='loop_transformations.txt', compensate=True):
        ''' Update pred poses using HLoc pose graph 
            @param compensate: if true, all the src poses without a loop transformation are set to identity matrix
        '''
        
        pred_file = os.path.join(pg_folder, file_name)
        
        self.src_poses.clear()
        self.src_poses = load_pose_file(os.path.join(pg_folder, 'src_poses.txt'))
        self.ref_poses = load_pose_file(os.path.join(pg_folder, 'ref_poses.txt'))
        
        with open(pred_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                # if 'frame' in line:
                eles = line.split(' ')
                src_frame = eles[0]
                ref_frame = eles[1]

                tx, ty, tz = float(eles[2]), float(eles[3]), float(eles[4])
                qx, qy, qz, qw = float(eles[5]), float(eles[6]), float(eles[7]), float(eles[8])
                quat = R.from_quat([qx, qy, qz, qw])

                
                if 'loop_transformations' in file_name:
                    if ref_frame not in self.ref_poses: continue

                    T_c1_c0 = np.eye(4)
                    T_c1_c0[:3,:3] = quat.as_matrix()
                    T_c1_c0[:3,3] = [tx, ty, tz]        
                    # T_src_c0 = self.src_poses[src_frame]            
                    self.pred_poses[src_frame] = self.ref_poses[ref_frame] @ T_c1_c0  # T_ref_c0 = T_ref_c1 @ T_c1_c0
                elif 'pose_average' in file_name or 'pgo' in file_name:
                    if src_frame not in self.src_poses: continue
                    T_ref_src = np.eye(4)
                    T_ref_src[:3,:3] = quat.as_matrix()
                    T_ref_src[:3,3] = [tx, ty, tz]
                    # T_src_ref = np.linalg.inv(T_ref_src)
                    self.pred_poses[src_frame] = T_ref_src @ self.src_poses[src_frame] # T_ref_c = T_ref_src @ T_src_c
                else:
                    raise ValueError('Unknown pose graph file name')
            
            
            print(' update {} pred poses from HLoc'.format(len(self.pred_poses)))
            
            if compensate:
                for frame_name, pose in self.src_poses.items():
                    if frame_name not in self.pred_poses:
                        self.pred_poses[frame_name] = self.src_poses[frame_name]

    def update_icp_gt_poses(self, gt_pose_file):
        
        T_ref_src = np.loadtxt(gt_pose_file) # gt T_ref_src
        for frame_name, pose in self.src_poses.items():
            self.gt_poses[frame_name] = T_ref_src @ pose
            # self.gt_poses[frame_name] = T_ref_src
            
        print('Update all gt poses using ICP gt pose from', gt_pose_file) 
        return T_ref_src  
        
    def update_tag_gt_poses(self, src_tag_folder, ref_tag_folder):
        # print('Update gt pose graph using tag')
        src_camera_tag_poses = load_camera_tag_poses(src_tag_folder, True) # T_tag_c
        T_reg_tag_f = load_vins_tag_poses(ref_tag_folder) # T_vins_tag
        if T_reg_tag_f is None: return False
        
        for frame, T_tag_c in src_camera_tag_poses.items():
            if 'pose' in frame:
                frame_name_t = frame.split('_')[0]
            else:
                frame_name_t = frame.split('.')[0]
            
            gt_ref_c = T_reg_tag_f @ T_tag_c
            self.gt_poses[frame_name_t] = gt_ref_c
            
            if frame_name_t not in self.pred_poses:# try not use it. have bug.
                frame_name_reg = list(self.pred_poses.keys())[-1]
                T_src_reg_f = self.src_poses[frame_name_reg]
                T_src_tag_f = self.src_poses[frame_name_t]
                T_reg_tag_f = np.linalg.inv(T_src_reg_f) @ T_src_tag_f
                T_pred_tag_f = self.pred_poses[frame_name_reg] @ T_reg_tag_f # T_ref_predtag = T_ref_reg @ T_reg_predtag
                self.pred_poses[frame_name_t] = T_pred_tag_f
            
        print('Update {} gt poses in ref:{}, src:{}'.format(len(self.gt_poses), ref_tag_folder, self.src_scene))
        return True
        
    def evaluate(self):
        pred_pos_arr = []
        gt_pos_arr = []
        frames = []
        if(len(self.pred_poses)==0 or len(self.gt_poses)==0):
            print('No pred or gt poses found')
            return None, None, None
        
        for frame, pred_pose in self.pred_poses.items():
            if frame in self.gt_poses:            
                pred_pos_arr.append(pred_pose[:3,3])
                gt_pos_arr.append(self.gt_poses[frame][:3,3])
                frames.append(frame)
            
        pred_pos_arr = np.array(pred_pos_arr)
        gt_pos_arr = np.array(gt_pos_arr)
        # rpe_array = relative_pos_error(pred_pos_arr, gt_pos_arr)
        rpe_array = np.linalg.norm(pred_pos_arr-gt_pos_arr, axis=1)
        rpe_tp = rpe_array<self.rpe_threshold

        pred_pos_arr = pred_pos_arr[rpe_tp]
        gt_pos_arr = gt_pos_arr[rpe_tp]
        ate, _ = position_ate(pred_pos_arr, gt_pos_arr)
        
        print('threshold: ', self.rpe_threshold)
        print('{} kf have gt. ATE: {:.3f}m, {}/{} tp'.format(gt_pos_arr.shape[0],ate, rpe_tp.sum(), rpe_tp.shape[0]))
        return ate, frames, pred_pos_arr, gt_pos_arr, rpe_array

def compensate_our_frames(our_result_folder, GAP=10):
    frame_dirs = glob.glob(our_result_folder+'/*.txt')
    frame_dirs = [frame_dir for frame_dir in frame_dirs if 'centroid' not in frame_dir]
    frame_dirs = [frame_dir for frame_dir in frame_dirs if 'cmatches' not in frame_dir]
    frame_dirs = sorted(frame_dirs)
    
    beginning_frame = frame_dirs[0]
    with open(beginning_frame, 'r') as f:
        lines = f.readlines()
        beginning_timestamp = (lines[0].strip().split(':')[-1].split(';')[0])
        beginning_timestamp = beginning_timestamp.strip()
        beginning_timestamp = int(beginning_timestamp)
        print(beginning_timestamp)
        
    beginning_frame_name = os.path.basename(beginning_frame).split('.')[0]
    beginning_frame_id = int(beginning_frame_name[6:12])
    
    print('Compensate from beginning frame: ', beginning_frame_name)
    idx = beginning_frame_id - GAP
    count = 1
    while(idx>100):
        frame_name = 'frame-{:06d}'.format(idx)
        import shutil
        shutil.copy(beginning_frame, os.path.join(our_result_folder, frame_name+'.txt'))
        
        # with open(os.path.join(our_result_folder, frame_name+'.txt'), 'w') as f:
        #     f.write('# timestamp: {}; pose\n'.format(beginning_timestamp - count))
        #     for line in lines[1:]:
        #         f.write(line)
            
        #     f.close()
        
        idx -= GAP
        count += 1

def organize_our_pose_avg(our_scene_result, ref_scene):
    
    rawfolder = os.path.join(our_scene_result,'{}'.format(ref_scene))
    infolder = os.path.join(our_scene_result,'pose_average_{}'.format(ref_scene))
    outfolder = os.path.join(our_scene_result,'optimized_{}'.format(ref_scene))
    
    if os.path.exists(outfolder)==False:
        os.makedirs(outfolder)
    
    frame_files = glob.glob(rawfolder+'/*.txt')
    frame_files = [frame_dir for frame_dir in frame_files if 'centroid' not in frame_dir]
    frame_files = [frame_dir for frame_dir in frame_files if 'cmatches' not in frame_dir]
    
    for frame_dir in frame_files:
        frame_name = os.path.basename(frame_dir).split('.')[0]
        frame_name = frame_name.split('_')[0]  
        if os.path.exists(os.path.join(infolder, frame_name+'.txt')):
            T_ref_src = np.loadtxt(os.path.join(infolder, frame_name+'.txt'))
        else:
            T_ref_src = np.eye(4)
              
        with open(frame_dir, 'r') as f:
            lines = f.readlines()

            timestamp = lines[0].strip().split(':')[-1].split(';')[0]
            timestamp = float(timestamp.strip())
            match_results = None
            
            for row, line in enumerate(lines):
                if '# src, ref' in line:
                    match_results = lines[row:]
                    break
        
        with open(os.path.join(outfolder,frame_name+'.txt'), 'w') as f:
            f.write('# timetstamp: {}; pose\n'.format(timestamp))
            
            for i in range(4):
                for j in range(4):
                    f.write('{:.6f} '.format(T_ref_src[i,j]))
                f.write('\n')

            for line in match_results:
                f.write(line)

            f.close()
        
        # break
        

if __name__=='__main__':
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/realsense.yaml")
    parser.add_argument("--dataroot", type=str, default="/data2/sgslam")
    parser.add_argument(
        "--output_folder", type=str, default="/data2/sgslam/output/v9")
    parser.add_argument("--split_file", type=str, default="multi_agent.txt")
    parser.add_argument("--gt_source", type=str, default="icp", help="icp, tag or optitrack")
    args = parser.parse_args()
    PE_THREAHOLD = 0.2
    SFM_DATAROOT = '/data2/sfm/multi_agent'
    EVAL_SFM = False
    POSE_AVERAGE = True
    CONSIDER_IOU = False
    SR_FILENAME = 'success_rate_pa5.txt'

    if POSE_AVERAGE:
        SFM_PG_FILE = 'summary_pose_average_pa5.txt'
        # SFM_PG_FILE = 'summary_pgo.txt'
        output_posfix = 'ave'
    else:
        SFM_PG_FILE = 'loop_transformations.txt'
        output_posfix = 'raw'

    # scan_pairs = read_scan_pairs(os.path.join(args.dataroot,'splits', args.split_file))  
    
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
    
    summary_pred_poses = []
    summary_gt_poses = []
    summary_pes = []
    summary_frames_success = [] # (N,2) frame_number, success_rate
    count_sequence = 0

    for pair in scan_pairs:
        print('---- Evaluate loop src:{}, ref:{} ----'.format(pair[0], pair[1]))
        scene_trajectory = TrajectoryAnalysis(os.path.join(args.dataroot, 'scans', pair[0]),'icp',PE_THREAHOLD)
        # organize_our_pose_avg(os.path.join(args.output_folder, pair[0]), pair[1])
        # compensate_our_frames(os.path.join(args.output_folder, pair[0], pair[1]), GAP=10)

        if EVAL_SFM:
            scene_folder = os.path.join(SFM_DATAROOT, '{}-{}'.format(pair[0], pair[1]))
            loop_file = os.path.join(SFM_DATAROOT, '{}-{}'.format(pair[0], pair[1]), 
                                     'pose_graph', 'loop_transformations.txt')
            if os.path.exists(loop_file)==False: continue
            scene_trajectory.update_hloc_framewise_poses(os.path.join(scene_folder,'pose_graph'), SFM_PG_FILE)
        else:
            scene_folder = os.path.join(args.output_folder, pair[0])
            pair_output_folder = os.path.join(args.output_folder, pair[0], pair[1])
            if os.path.exists(pair_output_folder)==False:
                continue
            if POSE_AVERAGE:
                scene_trajectory.update_our_average_poses(os.path.join(scene_folder, 'pose_average_{}'.format(pair[1])))
            else:            
                scene_trajectory.update_ours_framewise_poses(args.output_folder, pair[1])
                
        if args.gt_source=='icp':
            gt_pose = scene_trajectory.update_icp_gt_poses(os.path.join(args.dataroot, 
                                                        'gt', '{}-{}.txt'.format(pair[0], pair[1])))
            
        elif args.gt_source=='tag':
            scene_trajectory.update_aligned_poses(args.output_folder, pair[1])
            exist_tag = scene_trajectory.update_tag_gt_poses(os.path.join(args.dataroot, 'scans', pair[0], 'apriltag'),
                                                 os.path.join(args.dataroot, 'scans', pair[1], 'apriltag'))
            if exist_tag==False:
                print('Skip tag evaluation')
                continue
        # break
        ate, frames, pred_poses, gt_poses, ate = scene_trajectory.evaluate()
        if ate is None:continue
        
        if args.gt_source=='icp':
            scene_frames_success = compute_success_rate(frames, ate, PE_THREAHOLD)
            
            if CONSIDER_IOU:
                ious = compute_iou(args.output_folder, pair[0], pair[1], frames, gt_pose, EVAL_SFM)
                scene_frames_success = np.concatenate([scene_frames_success, ious.reshape(-1,1)], axis=1)
            scene_frames_success = scene_frames_success[scene_frames_success[:,0].argsort()]
            
            summary_frames_success.append(scene_frames_success)
            print('max frames: ', scene_frames_success[-1,0])
            # write_scene_result(frames, rte, os.path.join(scene_folder, 'ate_{}-{}_{}.txt'.format(pair[0],pair[1],output_posfix)))
        summary_pred_poses.append(pred_poses)
        summary_gt_poses.append(gt_poses)
        summary_pes.append(ate)
        count_sequence += 1
        # break        

    #
    if (count_sequence>0):
        print('------- Summary -------')
        summary_pred_poses = np.concatenate(summary_pred_poses, axis=0)
        summary_gt_poses = np.concatenate(summary_gt_poses, axis=0)
        summary_pes = np.concatenate(summary_pes, axis=0)
        
        summary_pes_tp = summary_pes<PE_THREAHOLD
        final_ate, _ = position_ate(summary_pred_poses, summary_gt_poses)

        print('Evaluate {} pair of scenes {} poses. Final ATE: {:.3f}m'.format(
            count_sequence,summary_gt_poses.shape[0],final_ate))
        print('{}/{} true predictions'.format(np.sum(summary_pes_tp), summary_pes_tp.shape[0]))
    
    if len(summary_frames_success)>0:
        summary_succes_rate = np.concatenate(summary_frames_success, axis=0)
        if EVAL_SFM:
            np.savetxt(os.path.join(SFM_DATAROOT, SR_FILENAME), 
                       summary_succes_rate, fmt='%.3f')
        else:
            np.savetxt(os.path.join(args.output_folder, SR_FILENAME), 
                       summary_succes_rate, fmt='%.3f')
        success_rate = summary_succes_rate[:,1].sum() / summary_succes_rate.shape[0]
        print('Save {}/{} success frames. success rate {:.3f}'.format(
                        summary_succes_rate[:,1].sum(), summary_succes_rate.shape[0], success_rate))

