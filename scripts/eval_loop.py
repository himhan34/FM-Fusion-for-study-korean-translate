import os, sys
import torch
# from typing import Optional

# import torch
# import torch.nn.functional as F

import numpy as np
import open3d as o3d
from run_test_loop import read_scan_pairs

def read_match_result(dir):
    pose = None
    pred_nodes = []
    pred_scores = []
    
    with open(dir) as f:
        lines = f.readlines()
        pose_lines = lines[1:5]
        pred_lines = lines[6:]
        f.close()
        
        pose = np.array([[float(x.strip()) for x in line.strip().split(' ')] for line in pose_lines])   
        for line in pred_lines:
            nodes_pair = line.split(' ')[0].strip()[1:-1]
            pred_nodes.append([int(x) for x in nodes_pair.split(',')])
            pred_scores.append(float(line.split(' ')[1].strip()))
        
        pred_nodes = np.array(pred_nodes)
        pred_scores = np.array(pred_scores)
        
        return pose, pred_nodes, pred_scores

def eval_instance_match(pred_nodes, iou_map, min_iou=0.2):
    # print('pred_nodes\n', pred_nodes)
    # print('gt src names\n', iou_map['src_names'])
    # print('gt ref names\n', iou_map['tar_names'])
    iou_mat = iou_map['iou'].numpy()
    
    pred_mask = np.zeros_like(pred_scores).astype(bool)
    
    i=0
    for pair in pred_nodes:
        if pair[0] in iou_map['src_names'] and pair[1] in iou_map['tar_names']:
            src_node_id = iou_map['src_names'].index(pair[0])
            ref_node_id = iou_map['tar_names'].index(pair[1])
            gt_iou = iou_mat[src_node_id, ref_node_id]
            # print(pair[0],pair[1],gt_iou)
            # print(iou_mat[src_node_id,:])
            if gt_iou> min_iou:
                pred_mask[i] = True
        else:
            print('WARNING: {} {} not in gt'.format(pair[0], pair[1]))
        i+=1

    # print(pred_mask)

    return pred_mask





def apply_transform(points: torch.Tensor, transform: torch.Tensor):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )

    return points

class InstanceEvaluator:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.gt = 0
        
    def update(self, pred_mask, gt_pairs):
        self.tp += np.sum(pred_mask)
        self.fp += np.sum(~pred_mask)
        self.gt += gt_pairs.shape[0]
        
    def get_metrics(self):
        precision = self.tp/(self.tp+self.fp)
        recall = self.tp/self.gt
        f1 = 2*precision*recall/(precision+recall)
        
        return recall, precision, f1
    
def eval_registration_error(src_cloud, pred_tf, gt_tf):
    src_points = np.asarray(src_cloud.points)
    src_points = torch.from_numpy(src_points).float()
    pred_tf = torch.from_numpy(pred_tf).float()
    gt_tf = torch.from_numpy(gt_tf).float()
    
    realignment_transform = torch.matmul(torch.inverse(gt_tf), pred_tf)
    realigned_src_points_f = apply_transform(src_points, realignment_transform)
    rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()    
    return rmse
    
    # rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
    # recall = torch.lt(rmse, self.acceptance_rmse).float()


    # src_corr_points = apply_transform(src_corr_points, transform)
    # corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
    # precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
    # return precision, corr_distances

    
        
if __name__ == '__main__':
    exe_dir = '/home/cliuci/code_ws/OpensetFusion/build/cpp/TestLoop'
    
    # args
    config_file = '/home/cliuci/code_ws/OpensetFusion/config/realsense.yaml'
    dataroot = '/data2/sgslam'
    split = 'val'
    split_file = 'val_bk.txt'
    ACCEPTANCE_RMSE = 0.2
    
    # 
    output_folder = os.path.join(dataroot,'output','testloop')
    scan_pairs = read_scan_pairs(os.path.join(dataroot, 'splits', split_file))
    inst_evaluator = InstanceEvaluator()
    registration_tp = 0
    
    # run
    for pair in scan_pairs:
        print('processing {} {}'.format(pair[0], pair[1]))
        scene_name = pair[1][:-1]
        src_subid = pair[0][-1]
        ref_subid = pair[1][-1]
        src_folder = os.path.join(dataroot, split, pair[0])
        ref_folder = os.path.join(dataroot, split, pair[1])
        gt_match_file = os.path.join(dataroot, 'matches', scene_name, 'matches_{}{}.pth'.format(src_subid, ref_subid))
        
        if os.path.exists(gt_match_file):
            gt_pairs, iou_map, _ = torch.load(gt_match_file) # gt data
            pred_pose, pred_nodes, pred_scores = read_match_result(os.path.join(output_folder,'{}-{}.txt'.format(pair[0], pair[1])))
            pred_tp_mask = eval_instance_match(pred_nodes, iou_map)
            inst_evaluator.update(pred_tp_mask, gt_pairs.numpy())
            gt_tf = np.loadtxt(os.path.join(dataroot, split, pair[0], 'transform.txt')) # src to ref
            src_pcd = o3d.io.read_point_cloud(os.path.join(output_folder, '{}.ply'.format(pair[0])))
            
            rmse = eval_registration_error(src_pcd, pred_pose, gt_tf)
            if rmse<ACCEPTANCE_RMSE:
                registration_tp += 1
            
            print('registration rmse: {:.3f}'.format(rmse))
            # print('gt matches',gt_pairs)
        else:
            print('WARNING: {} does not exist'.format(gt_match_file))

        # break

    #
    print('---------- Summary {} pairs ---------'.format(len(scan_pairs)))
    print('instance recall: {:.3f}, instance precision: {:.3f}, f1: {:.3f}'.format(*inst_evaluator.get_metrics()))
    print('registration recall:{}/{}'.format(registration_tp,len(scan_pairs)))
