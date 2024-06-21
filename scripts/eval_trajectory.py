import os, glob
import numpy as np 
import argparse

from eval_loop import read_match_centroid_result
from run_test_loop import read_scan_pairs

def position_ate(pred_pos:np.ndarray, gt_pos:np.ndarray):
    err = pred_pos - gt_pos # (N,3)
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2] # (N,)
    if np.isnan(norm2).any():
        print("ATE_POS has nan")
    return np.sqrt(np.mean(norm2)) # sqrt(mean(||pred-gt||^2))

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

class TrajectoryAnalysis:
    def __init__(self, scene_dir, gt_source='icp') -> None:
        self.src_scene = os.path.basename(scene_dir)
        self.poses = self.load_camera_poses(os.path.join(scene_dir,'pose')) # {frame_name: T_src_c}
        self.pred_poses = {} # pred poses, {frame_name: T_ref_c}
        self.gt_poses = {} # gt poses, {frame_name: T_ref_c}
        self.gt_source = gt_source # icp, tag or optitrack
        
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
        src_scene_output = os.path.join(output_folder, self.src_scene)
        loop_frames = glob.glob(src_scene_output+'/'+ref_scene+'/*.txt')
        loop_frames = sorted(loop_frames)
        if len(loop_frames)==0:
            print('No loop frames found in ', src_scene_output)
            return None
        
        T_ref_src, _, _ = read_match_centroid_result(loop_frames[-1]) # predicted T_ref_src
        
        for frame_name, pose in self.poses.items():
            self.pred_poses[frame_name] = T_ref_src @ pose # T_ref_c = T_ref_src @ T_src_c
        print('update pred pose graph using: ', loop_frames[-1])

    def update_icp_gt_poses(self, gt_pose_file):
        
        T_ref_src = np.loadtxt(gt_pose_file) # gt T_ref_src
        for frame_name, pose in self.poses.items():
            self.gt_poses[frame_name] = T_ref_src @ pose
            
        print('Update gt pose graph using ICP gt pose from', gt_pose_file)   
        
    def update_tag_gt_poses(self, src_tag_folder, ref_tag_folder):
        # print('Update gt pose graph using tag')
        src_camera_tag_poses = load_camera_tag_poses(src_tag_folder, True) # T_tag_c
        T_ref_tag = load_vins_tag_poses(ref_tag_folder) # T_vins_tag
        if T_ref_tag is None: return None
        
        for frame, T_tag_c in src_camera_tag_poses.items():
            if 'pose' in frame:
                frame_name = frame.split('_')[0]
            else:
                frame_name = frame.split('.')[0]
            
            gt_ref_c = T_ref_tag @ T_tag_c
            self.gt_poses[frame_name] = gt_ref_c
        print('Update {} gt poses in ref:{}, src:{}'.format(len(self.gt_poses), ref_tag_folder, self.src_scene))
        
    def evaluate(self):
        pred_pos_arr = []
        gt_pos_arr = []
        if(len(self.pred_poses)==0 or len(self.gt_poses)==0):
            print('No pred or gt poses found')
            return None, None, None
        
        for frame, pred_pose in self.pred_poses.items():
            if frame in self.gt_poses:            
                pred_pos_arr.append(pred_pose[:3,3])
                gt_pos_arr.append(self.gt_poses[frame][:3,3])
            
        pred_pos_arr = np.array(pred_pos_arr)
        gt_pos_arr = np.array(gt_pos_arr)
        
        ate = position_ate(pred_pos_arr, gt_pos_arr)
        print('{} kf have gt. ATE: {:.3f}m'.format(gt_pos_arr.shape[0],ate))
        return ate, pred_pos_arr, gt_pos_arr
        
        
if __name__=='__main__':
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/realsense.yaml")
    parser.add_argument("--dataroot", type=str, default="/data2/sgslam")
    parser.add_argument(
        "--output_folder", type=str, default="/data2/sgslam/output/online_coarse+")
    parser.add_argument("--split_file", type=str, default="val_bk.txt")
    parser.add_argument("--gt_source", type=str, default="tag", help="icp, tag or optitrack")
    args = parser.parse_args()  
    
    scan_pairs = read_scan_pairs(os.path.join(args.dataroot,'splits', args.split_file))  
    summary_pred_poses = []
    summary_gt_poses = []
    count_sequence = 0

    for pair in scan_pairs:
        print('---- Evaluate loop src:{}, ref:{} ----'.format(pair[0], pair[1]))
        
        scene_trajectory = TrajectoryAnalysis(os.path.join(args.dataroot, 'scans', pair[0]))
        scene_trajectory.update_aligned_poses(args.output_folder, pair[1])
        if args.gt_source=='icp':
            scene_trajectory.update_icp_gt_poses(os.path.join(args.dataroot, 
                                                        'gt', '{}-{}.txt'.format(pair[0], pair[1])))
        elif args.gt_source=='tag':
            scene_trajectory.update_tag_gt_poses(os.path.join(args.dataroot, 'scans', pair[0], 'apriltag'),
                                                 os.path.join(args.dataroot, 'scans', pair[1], 'apriltag'))

        
        ate, pred_poses, gt_poses = scene_trajectory.evaluate()
        if ate is None:continue
        
    
        summary_pred_poses.append(pred_poses)
        summary_gt_poses.append(gt_poses)
        count_sequence += 1
        
    
    #
    if (count_sequence>0):
        summary_pred_poses = np.concatenate(summary_pred_poses, axis=0)
        summary_gt_poses = np.concatenate(summary_gt_poses, axis=0)
        final_ate = position_ate(summary_pred_poses, summary_gt_poses)
        print('Evaluate {} pair of scenes. Final ATE: {:.3f}m'.format(count_sequence,final_ate))


