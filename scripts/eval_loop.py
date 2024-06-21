import os, sys
import glob
import torch
import argparse

import numpy as np
import open3d as o3d
from run_test_loop import read_scan_pairs

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_match_result(dir):
    pose = None
    pred_nodes = []
    pred_scores = []

    with open(dir) as f:
        lines = f.readlines()
        pose_lines = lines[1:5]
        pred_lines = lines[6:]
        f.close()

        pose = np.array(
            [[float(x.strip()) for x in line.strip().split(" ")] for line in pose_lines]
        )
        for line in pred_lines:
            nodes_pair = line.split(" ")[0].strip()[1:-1]
            pred_nodes.append([int(x) for x in nodes_pair.split(",")])
            pred_scores.append(float(line.split(" ")[1].strip()))

        pred_nodes = np.array(pred_nodes)
        pred_scores = np.array(pred_scores)

        return pose, pred_nodes, pred_scores

def read_match_centroid_result(dir):
    pose = None
    src_centroids = []
    ref_centroids = []


    with open(dir) as f:
        lines = f.readlines()
        pose_lines = lines[1:5]
        pred_lines = lines[6:]
        f.close()

        pose = np.array(
            [[float(x.strip()) for x in line.strip().split(" ")] for line in pose_lines]
        )
        for line in pred_lines:
            eles = line.split(' ')
            src_centroid = [float(x) for x in eles[1:4]]
            ref_centroid = [float(x) for x in eles[4:7]]
            src_centroids.append(src_centroid)
            ref_centroids.append(ref_centroid)
            # print(src_centroids, ref_centroids)

        #
        src_centroids = np.array(src_centroids)
        ref_centroids = np.array(ref_centroids)
            
        return pose, src_centroids, ref_centroids

def read_instance_match_result(dir):
    pred_nodes = np.loadtxt(dir, dtype=int)
    pred_scores = np.ones_like(pred_nodes[:,0])
    pose = np.eye(4)
    
    return pose, pred_nodes, pred_scores

def eval_instance_match(pred_nodes, iou_map, min_iou=0.2):
    # print('pred_nodes\n', pred_nodes)
    # print('gt src names\n', iou_map['src_names'])
    # print('gt ref names\n', iou_map['tar_names'])
    iou_mat = iou_map['iou'].numpy()
    M = pred_nodes.shape[0]
    pred_mask = np.zeros(M, dtype=bool)
    
    i=0
    for pair in pred_nodes:
        if pair[0] in iou_map["src_names"] and pair[1] in iou_map["tar_names"]:
            src_node_id = iou_map["src_names"].index(pair[0])
            ref_node_id = iou_map["tar_names"].index(pair[1])
            gt_iou = iou_mat[src_node_id, ref_node_id]
            # print(pair[0],pair[1],gt_iou)
            # print(iou_mat[src_node_id,:])
            if gt_iou > min_iou:
                pred_mask[i] = True
        else:
            print("WARNING: {} {} not in gt".format(pair[0], pair[1]))
        i += 1

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
            "Incompatible shapes between points {} and transform {}.".format(
                tuple(points.shape), tuple(transform.shape)
            )
        )

    return points

def check_sequence_output(scene_dir):
    if os.path.exists(os.path.join(scene_dir,'config.txt')):
        return True
    else: return False
    

def compute_cloud_overlap(cloud_a:o3d.geometry.PointCloud,cloud_b:o3d.geometry.PointCloud,search_radius=0.2):
    # compute point cloud overlap 
    Na = len(cloud_a.points)
    Nb = len(cloud_b.points)
    correspondences = []
    cloud_a_occupied = np.zeros((Na,1))
    pcd_tree_b = o3d.geometry.KDTreeFlann(cloud_b)
    
    for i in range(Na):
        [k,idx,_] = pcd_tree_b.search_radius_vector_3d(cloud_a.points[i],search_radius)
        # [k,idx,dists] = pcd_tree_b.search_knn_vector_3d(cloud_a.points[i],1)
        if k>1:
        # if dists[0]<search_radius:
            cloud_a_occupied[i] = 1
            correspondences.append([i,idx[0]])
    assert len(correspondences)==len(np.argwhere(cloud_a_occupied==1))
    iou = len(correspondences)/(Na+Nb-len(correspondences))
    return iou, correspondences

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
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / self.gt
        f1 = 2 * precision * recall / (precision + recall)

        return recall, self.tp, self.gt, precision, self.tp, self.tp + self.fp
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


def evaluate_fine(src_corr_points, ref_corr_points, transform, acceptance_radius=0.1):
    if src_corr_points.shape[0] == 0 or ref_corr_points.shape[0] == 0:
        return 0.0, np.array([])
    src_corr_points = torch.from_numpy(src_corr_points).float()
    ref_corr_points = torch.from_numpy(ref_corr_points).float()
    transform = torch.from_numpy(transform).float()

    src_corr_points = apply_transform(src_corr_points, transform)
    corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
    # print('dist range: [{:.3f}, {:.3f}]'.format(corr_distances.min(), corr_distances.max()))
    corr_true_mask = torch.lt(corr_distances, acceptance_radius)
    precision = torch.lt(corr_distances, acceptance_radius).float().mean()
    return precision, corr_true_mask


def rerun_registration(config_file, dataroot, output_folder, scan_pairs):
    exec_file = os.path.join(ROOT_DIR, "build/cpp/TestLoop")
    for (ref_scene, src_scene) in scan_pairs:
        cmd = "{} --config {} --weights_folder {} --src_scene {} --ref_scene {} --output_folder {} --prune_instance --dense_match".format(
            exec_file,
            config_file,
            "./torchscript",
            os.path.join(dataroot, "val/{}".format(ref_scene)),
            os.path.join(dataroot, "val/{}".format(src_scene)),
            output_folder,
        )
        os.system(cmd)

def eval_offline_loop(dataroot, scan_pairs, output_folder):
    inst_evaluator = InstanceEvaluator()
    registration_tp = 0
    inliner_count = 0
    corr_count = 1e-6
    print("Evaluate results in {}".format(os.path.basename(output_folder)))
    
    for pair in scan_pairs:
        print("--- processing {} {} ---".format(pair[0], pair[1]))
        scene_name = pair[1][:-1]
        src_subid = pair[0][-1]
        ref_subid = pair[1][-1]
        gt_match_file = os.path.join(dataroot,
                                "matches",
                                scene_name,
                                "matches_{}{}.pth".format(src_subid, ref_subid))
        
        output_file = os.path.join(output_folder,'{}-{}.txt'.format(pair[0], pair[1]))

        if os.path.exists(output_file):
            gt_pairs, iou_map, _ = torch.load(gt_match_file)  # gt data
            # pred_pose, pred_nodes, pred_scores = read_instance_match_result(os.path.join(output_folder,'{}-{}.txt'.format(pair[0], pair[1])))
            pred_pose, pred_nodes, pred_scores = read_match_result(output_file)
            pred_tp_mask = eval_instance_match(pred_nodes, iou_map)
            inst_evaluator.update(pred_tp_mask, gt_pairs.numpy())
            gt_tf = np.loadtxt(
                os.path.join(dataroot, split, pair[0], "transform.txt")
            )  # src to ref

            src_pcd_dir = os.path.join(output_folder, "{}.ply".format(pair[0]))
            if os.path.exists(src_pcd_dir): # instance matched
                src_pcd = o3d.io.read_point_cloud(src_pcd_dir)
                rmse = eval_registration_error(src_pcd, pred_pose, gt_tf)
            else:
                rmse = 100

            if args.dense:
                corr_src_pcd = o3d.io.read_point_cloud(
                    os.path.join(
                        output_folder, "{}-{}_csrc.ply".format(pair[0], pair[1])
                    )
                )
                corr_ref_pcd = o3d.io.read_point_cloud(
                    os.path.join(
                        output_folder, "{}-{}_cref.ply".format(pair[0], pair[1])
                    )
                )

                inliner_rate, corr_true_mask = evaluate_fine(
                    np.asarray(corr_src_pcd.points),
                    np.asarray(corr_ref_pcd.points),
                    gt_tf,
                    acceptance_radius=INLINER_RADIUS,
                )
                inliner_count += corr_true_mask.sum()
                corr_count += corr_true_mask.shape[0]
            else:
                inliner_rate = 0.0

            if rmse < RMSE_THRESHOLD:
                registration_tp += 1
            print(
                "Instance match, tp: {}, fp:{}".format(
                    pred_tp_mask.sum(), (~pred_tp_mask).sum()
                )
            )
            print(
                "PIR: {:.3f}, registration rmse:{:.3f}, {}".format(
                    inliner_rate, rmse, "success" if rmse < RMSE_THRESHOLD else "fail"
                )
            )

    #
    print("---------- Summary {} pairs ---------".format(len(scan_pairs)))
    print(
        "instance recall: {:.3f}({}/{}), instance precision: {:.3f}({}/{})".format(
            *inst_evaluator.get_metrics()
        )
    )
    print("points inliner rate: {:.3f}".format(inliner_count / corr_count))
    print("registration recall:{}/{}".format(registration_tp, len(scan_pairs)))


def read_frames_map(gt_scene_folder):
    ''' The folder contains a sequence of frames that have the src map stored.
        Return,
        frames_map: {'indices':[], 'dirs':[]}
    '''
    gt_maps = glob.glob(gt_scene_folder+'/*.ply')
    gt_maps = sorted(gt_maps)
    frames_map = {'indices':[], 'dirs':[]}
    
    for file in gt_maps:
        frame_name = os.path.basename(file).split('.')[0]
        frame_id = int(frame_name.split('-')[-1])
        # frames_map[frame_name] = file
        frames_map['indices'].append(frame_id)
        frames_map['dirs'].append(file)
        
    frames_map['indices'] = np.array(frames_map['indices'])
    # print(frames_map['indices'])
    
    return frames_map

def find_closet_index(ref_indices:np.ndarray, ref_dirs:list, query_index:int):
    ''' Find the closest index in the ref_indices to the query_index'''
    diff = np.abs(ref_indices-query_index)
    idx = np.argmin(diff)
    return ref_dirs[idx]


def read_timinig_record(dir, verbose=False):
    '''
    Return dict:
        'mapping_runtime': np.ndarray, (K,)
        'loop_header': list, (S,)
        'loop_runtime': np.ndarray, (K',S)
        
    '''
    
    if(verbose):
        print('read timing record from ',dir)
    
    loop_headers = ['SG-Create','SG-Net','Comm','Shape','C-Match','P-Match','Pose','IO']
    mapping_runtime = []
    loop_runtime = []
    with open(dir,'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            eles = line.strip().split(' ')
            eles = [ele.strip() for ele in eles]
            if(len(eles)<2):continue
            frame_id = eles[0]
            load_data = eles[1]
            
            #
            mapping_runtime.append(float(eles[2]))
            
            #
            if(len(eles)>3):
                frame_loop = np.array([float(x) for x in eles[3:]])
                loop_runtime.append(frame_loop)
        
        #
        mapping_runtime = np.array(mapping_runtime)
        loop_runtime = np.array(loop_runtime)
        f.close()

        #   
        mean_mapping_time = np.mean(mapping_runtime)
        mean_loop_time_details = np.mean(loop_runtime,axis=0)
        mean_loop_time = np.sum(mean_loop_time_details[:-1])
        
        print('Mappping frames: {}, Loop frames: {}'.format(len(mapping_runtime), len(loop_runtime)))
        msg = 'Mapping '
        for header in loop_headers:
            msg += '{}   '.format(header)
        msg += '   Sum\n'
        msg += '{:.3f}   '.format(mean_mapping_time)
        for i in range(len(mean_loop_time_details)):
            msg += '{:.3f}   '.format(mean_loop_time_details[i])
        msg += '{:.3f}\n'.format(mean_loop_time)

        if(verbose): print(msg)
        
        return {'mapping_runtime':mapping_runtime,
                'loop_header':loop_headers,
                'loop_runtime':loop_runtime}


def write_eval_data(outdir, tp_instances, count_instances, rmse_list, iou):
    with open(outdir, 'w') as f:
        f.write('# tp_instance, count_instance, rmse, iou\n')
        for i in range(len(tp_instances)):
            f.write('{},{},{:.3f},{:.3f}\n'.format(tp_instances[i], count_instances[i], rmse_list[i], iou[i]))
        f.close()
        return True

def eval_online_loop(dataroot, scan_pairs, output_folder, consider_iou, write_log=False):
    '''
        Evaluate online loop results.
        If CONSIDER_IOU is ON: it requires a GT_IOU folder, 
        so each pair of scan is evaluated with their 3D IOU. 
        Summarized registration results is saved to each scene output folder.
        Runtime result is directly printed.
    '''
    from timing import TimeAnalysis
    
    INSTANCE_RADIUS = 1.0
    CONSIDER_IOU = consider_iou
    print('Evaluate online loop results')
    
    # mapping_runtime = TimeAnalysis()
    loop_runtime = TimeAnalysis()
    mapping_runtime = []
    loop_runtime.header = ['SG-Create','SG-Net','Comm','  Shape','C-Match','P-Match','  Pose','   IO']
        
    for pair in scan_pairs:
        if(check_sequence_output(os.path.join(output_folder, pair[0]))==False): continue
        if(check_sequence_output(os.path.join(output_folder, pair[1]))==False): continue
        
        print("--- processing {} {} ---".format(pair[0], pair[1]))
        scene_runtime_dict = read_timinig_record(os.path.join(output_folder, pair[0],'timing.txt'),True)
        mapping_runtime.append(scene_runtime_dict['mapping_runtime'])
        loop_runtime.add_frame_data(scene_runtime_dict['loop_runtime'])    
        # continue
        
        if CONSIDER_IOU:
            ref_maps = read_frames_map(os.path.join(dataroot, 'output', 'gt_iou', pair[1]))

        loop_frames = glob.glob(os.path.join(output_folder, pair[0], pair[1], '*.txt'))
        loop_frames = sorted(loop_frames)

        gt_pose = np.loadtxt(os.path.join(dataroot, 'gt', '{}-{}.txt'.format(pair[0], pair[1]))) 
        # src_pcd = o3d.io.read_point_cloud(os.path.join(output_folder, pair[0], "instance_map.ply"))

        # output evaluation
        ious = []
        instance_tp = []
        instance_count = []
        rmse_list = []
        

        for frame in loop_frames:
            frame_name = os.path.basename(frame).split('.')[0]
            src_pcd = o3d.io.read_point_cloud(os.path.join(output_folder, pair[0], pair[1], '{}_src.ply'.format(frame_name)))

            pred_pose, src_centroids, ref_centroids = read_match_centroid_result(frame)
            inst_pre, inst_mask =evaluate_fine(src_centroids, ref_centroids, gt_pose, acceptance_radius=INSTANCE_RADIUS)
            rmse = eval_registration_error(src_pcd, pred_pose, gt_pose)            
            
            if CONSIDER_IOU:
                frame_id = int(frame_name.split('-')[-1])
                # src_map_dir = find_closet_index(src_maps['indices'], src_maps['dirs'], frame_id)
                ref_map_dir = find_closet_index(ref_maps['indices'], ref_maps['dirs'], frame_id)
                # print('{} find gt src file {}, ref file {}'.format(frame_name, src_map_dir, ref_map_dir))
                # src_pcd = o3d.io.read_point_cloud(src_map_dir)
                ref_pcd = o3d.io.read_point_cloud(ref_map_dir)
                src_pcd.transform(gt_pose)
                iou, _ = compute_cloud_overlap(src_pcd, ref_pcd, 0.2)
                # print('  {}'.format(iou))
            else:
                iou = -1.0
                        
            print('{}: {}/{} true instance matches, rmse {:.3f}, iou {:.3f}'.format(
                frame_name, inst_mask.sum(), len(src_centroids), rmse, iou))

            #I/O
            instance_tp.append(inst_mask.sum())
            instance_count.append(len(src_centroids))
            rmse_list.append(rmse)
            ious.append(iou)
            
        if write_log:
            write_eval_data(os.path.join(output_folder, pair[0], pair[1],'eval.log'), 
                            instance_tp, instance_count, rmse_list, ious)
        # break
    # Runtime summary
    mapping_runtime = np.concatenate(mapping_runtime)
    loop_runtime.analysis(verbose=True)
    print('mapping time: {:.3f}'.format(mapping_runtime.mean()))

def summary_registration_result(scan_pairs, output_folder):
    from registration_anaysis import RegistrationEvaluation, load_eval_file
    print('*** Read each scene result and summarize the registration result ***')
    ious_splits = [0.1,0.4,0.7,1.0]
    reg_summary = RegistrationEvaluation(ious_splits)
    
    for pair in scan_pairs:
        print("--- processing {} {} ---".format(pair[0], pair[1]))
        eval_file = os.path.join(output_folder,pair[0],pair[1],'eval.log')
        if os.path.exists(eval_file)==False: continue
        scene_loop_frames = load_eval_file(eval_file)
        reg_summary.record_loop_frames(scene_loop_frames)
    
    #
    reg_summary.analysis()    

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/realsense.yaml")
    parser.add_argument("--dataroot", type=str, default="/data2/sgslam")
    parser.add_argument("--match_folder", type=str, default="/data2/sgslam/matches")
    parser.add_argument(
        "--output_folder", type=str, default="/data2/sgslam/output/online_coarse"
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--split_file", type=str, default="val_bk.txt")
    parser.add_argument(
        "--dense",
        type=bool,
        default=True,
        help="evaluate point inliner rate if turned on",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="whether to rerun the registration",
    )
    parser.add_argument("--consider_iou",action="store_true",
                        help="whether to consider GT 3D IOU")
    parser.add_argument("--run_mode", type=str, default="offline", help="offline, online, reg")
    args = parser.parse_args()

    config_file = args.config_file
    dataroot = args.dataroot
    output_folder = args.output_folder
    split = args.split
    split_file = args.split_file
    RMSE_THRESHOLD = 0.2
    INLINER_RADIUS = 0.1
    RUN_MODE = args.run_mode # 'offline', 'online','reg'
     
    scan_pairs = read_scan_pairs(os.path.join(dataroot, "splits", split_file))
    if args.rerun:
        rerun_registration(config_file, dataroot, output_folder, scan_pairs) # offline registration

    # run
    if RUN_MODE=='offline':
        eval_offline_loop(args.dataroot, scan_pairs, args.output_folder)
    elif RUN_MODE =='online':
        eval_online_loop(args.dataroot, scan_pairs, args.output_folder, args.consider_iou)
    elif RUN_MODE =="reg":
        summary_registration_result(scan_pairs, args.output_folder)
    
    