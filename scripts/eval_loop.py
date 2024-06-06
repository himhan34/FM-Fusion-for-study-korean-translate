import os, sys
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


def eval_instance_match(pred_nodes, iou_map, min_iou=0.2):
    # print('pred_nodes\n', pred_nodes)
    # print('gt src names\n', iou_map['src_names'])
    # print('gt ref names\n', iou_map['tar_names'])
    iou_mat = iou_map["iou"].numpy()

    pred_mask = np.zeros_like(pred_scores).astype(bool)

    i = 0
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
    src_corr_points = torch.from_numpy(src_corr_points).float()
    ref_corr_points = torch.from_numpy(ref_corr_points).float()
    transform = torch.from_numpy(transform).float()

    src_corr_points = apply_transform(src_corr_points, transform)
    corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
    corr_true_mask = torch.lt(corr_distances, acceptance_radius)
    precision = torch.lt(corr_distances, acceptance_radius).float().mean()
    return precision, corr_true_mask


def rerun_registration(config_file, dataroot, output_folder, scan_pairs):
    exec_file = os.path.join(ROOT_DIR, "build/cpp/TestLoop")
    for (ref_scene, src_scene) in scan_pairs:
        cmd = "{} --config {} --weights_folder {} --src_scene {} --ref_scene {} --output_folder {} --dense_match  --prune_instance ".format(
            exec_file,
            config_file,
            "./torchscript",
            os.path.join(dataroot, "val/{}".format(ref_scene)),
            os.path.join(dataroot, "val/{}".format(src_scene)),
            output_folder,
        )
        os.system(cmd)


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/realsense.yaml")
    parser.add_argument("--dataroot", type=str, default="/data2/sgslam")
    parser.add_argument(
        "--output_folder", type=str, default="/data2/sgslam/output/coarse_register2"
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--split_file", type=str, default="val_bk.txt")
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="whether to rerun the registration",
    )
    args = parser.parse_args()

    config_file = args.config_file
    dataroot = args.dataroot
    output_folder = args.output_folder
    split = args.split
    split_file = args.split_file
    RMSE_THRESHOLD = 0.2
    INLINER_RADIUS = 0.1

    scan_pairs = read_scan_pairs(os.path.join(dataroot, "splits", split_file))
    if args.rerun:
        rerun_registration(config_file, dataroot, output_folder, scan_pairs)
    inst_evaluator = InstanceEvaluator()
    registration_tp = 0
    inliner_count = 0
    corr_count = 1e-6
    print("Evaluate results in {}".format(os.path.basename(output_folder)))

    # run
    for pair in scan_pairs:
        print("--- processing {} {} ---".format(pair[0], pair[1]))
        scene_name = pair[1][:-1]
        src_subid = pair[0][-1]
        ref_subid = pair[1][-1]
        src_folder = os.path.join(dataroot, split, pair[0])
        ref_folder = os.path.join(dataroot, split, pair[1])
        gt_match_file = os.path.join(
            dataroot,
            "matches",
            scene_name,
            "matches_{}{}.pth".format(src_subid, ref_subid),
        )

        if os.path.exists(gt_match_file):
            gt_pairs, iou_map, _ = torch.load(gt_match_file)  # gt data
            pred_pose, pred_nodes, pred_scores = read_match_result(
                os.path.join(output_folder, "{}-{}.txt".format(pair[0], pair[1]))
            )
            pred_tp_mask = eval_instance_match(pred_nodes, iou_map)
            inst_evaluator.update(pred_tp_mask, gt_pairs.numpy())
            gt_tf = np.loadtxt(
                os.path.join(dataroot, split, pair[0], "transform.txt")
            )  # src to ref

            src_pcd_dir = os.path.join(output_folder, "{}.ply".format(pair[0]))
            if os.path.exists(src_pcd_dir):  # instance matched
                src_pcd = o3d.io.read_point_cloud(src_pcd_dir)
                rmse = eval_registration_error(src_pcd, pred_pose, gt_tf)
                inliner_rate = 0.0

                # corr_src_pcd = o3d.io.read_point_cloud(os.path.join(output_folder, '{}-{}_csrc.ply'.format(pair[0],pair[1])))
                # corr_ref_pcd = o3d.io.read_point_cloud(os.path.join(output_folder, '{}-{}_cref.ply'.format(pair[0],pair[1])))

                # inliner_rate, corr_true_mask = evaluate_fine(np.asarray(corr_src_pcd.points),
                #                                           np.asarray(corr_ref_pcd.points),
                #                                           pred_pose,
                #                                           acceptance_radius=INLINER_RADIUS)
                # inliner_count += corr_true_mask.sum()
                # corr_count += corr_true_mask.shape[0]
            else:
                rmse = 10.0
                inliner_rate = 0.0

            if rmse < RMSE_THRESHOLD:
                registration_tp += 1
            print(
                "Instance match, tp: {}, fp:{}".format(
                    pred_tp_mask.sum(), (~pred_tp_mask).sum()
                )
            )
            print("PIR: {:.3f}, registration rmse:{:.3f}".format(inliner_rate, rmse))
        else:
            print("WARNING: {} does not exist".format(gt_match_file))

        # break

    #
    print("---------- Summary {} pairs ---------".format(len(scan_pairs)))
    print(
        "instance recall: {:.3f}({}/{}), instance precision: {:.3f}({}/{})".format(
            *inst_evaluator.get_metrics()
        )
    )
    print("points inliner rate: {:.3f}".format(inliner_count / corr_count))
    print("registration recall:{}/{}".format(registration_tp, len(scan_pairs)))
