import os, glob
import subprocess
from sre_parse import parse
import time

from run_test_loop import read_scan_pairs
from eval_loop import eval_registration_error, evaluate_fine
import open3d as o3d
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nms_thd", type=float, default=0.05)
    parser.add_argument("--ds_num", type=int, default=1)
    return parser.parse_args()



def copy_instance_map(dataroot, scan_pairs, output_folder):

    for pair in scan_pairs:
        src_scene_folder = os.path.join(dataroot, "val", pair[0])
        ref_scene_folder = os.path.join(dataroot, "val", pair[1])

        src_pcd = o3d.io.read_point_cloud(
            os.path.join(src_scene_folder, "instance_map.ply")
        )
        ref_pcd = o3d.io.read_point_cloud(
            os.path.join(ref_scene_folder, "instance_map.ply")
        )

        o3d.io.write_point_cloud(
            os.path.join(
                output_folder, "{}-{}".format(pair[0], pair[1]), "src_instances.ply"
            ),
            src_pcd,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                output_folder, "{}-{}".format(pair[0], pair[1]), "ref_instances.ply"
            ),
            ref_pcd,
        )
        # break


if __name__ == "__main__":
    args = parse_args()
    ####### Rio data structure ########
    # |-RioGraph_Dataroot
    #   |-gt
    #   |-output
    #     |-ours
    #       |-src_scene-ref_scene
    #       |- ...
    #     |-ours_superpoint
    #       |-src_scene-ref_scene
    #       |- ...
    #   |-splits
    #     |-val.txt
    ########### Args ############
    dataroot = "/data2/RioGraph"
    output_setting = "ours_superpoint"  # ours_geotransformer_filter0009
    # output_setting = 'ours'
    # output_setting = 'sgpgm'
    #############################

    # hard-code
    SPLIT = "val_gtransformer"
    EXE_DIR = "build/cpp/Test3RScanRegister"
    CFG_FILE = "config/realsense.yaml"
    RMSE_THRESHOLD = 0.2
    RUN_CMD = True
    OVERWRITE_NODES = False
    EVAL = True

    #
    scan_pairs = read_scan_pairs(os.path.join(dataroot, "splits", SPLIT + ".txt"))
    # scan_pairs = scan_pairs[:10]

    count = 0
    rmse_array = []
    ir_array = []
    corr_number_array = []
    summary_tp_corr = []
    # copy_instance_map(dataroot, scan_pairs, os.path.join(dataroot, 'output', output_setting))
    # exit(0)

    for pair in scan_pairs:
        # print("******** processing pair: {} ***********".format(pair))
        pair_corr_folder = os.path.join(
            dataroot, "output", output_setting, "{}-{}".format(pair[0], pair[1])
        )

        if OVERWRITE_NODES:  # don't use it. keep it off
            import shutil

            if os.path.exists(
                os.path.join(
                    dataroot,
                    "output",
                    "ours",
                    "{}-{}".format(pair[0], pair[1]),
                    "node_matches.txt",
                )
            ):
                shutil.copy(
                    os.path.join(
                        dataroot,
                        "output",
                        "ours",
                        "{}-{}".format(pair[0], pair[1]),
                        "node_matches.txt",
                    ),
                    os.path.join(pair_corr_folder, "node_matches.txt"),
                )

        cmd = "{} --config {}".format(EXE_DIR, CFG_FILE)
        cmd += " --gt_folder {}".format(os.path.join(dataroot, "gt"))
        cmd += " --corr_folder {}".format(pair_corr_folder)
        cmd += " --src_scene {} --ref_scene {}".format(pair[0], pair[1])

        # register params
        # cmd += " --max_corr_number {}".format(300)
        cmd += " --inlier_threshold {}".format(0.5)
        # cmd += " --enable_icp"
        cmd += " --ds_num 1"
        cmd += " --nms_thd {}".format(args.nms_thd)
        if RUN_CMD:
            print(cmd)
            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

        if EVAL:
            # gt_pose = np.loadtxt(os.path.join(dataroot, 'gt', '{}-{}.txt'.format(pair[0], pair[1])))
            src_pcd = o3d.io.read_point_cloud(
                os.path.join(pair_corr_folder, "src_instances.ply")
            )
            # gt_pose = np.loadtxt(os.path.join(dataroot, 'val', pair[0], 'transform.txt')).astype(np.float32)
            gt_pose = np.loadtxt(os.path.join(pair_corr_folder, "gt_pose.txt")).astype(
                np.float32
            )

            if os.path.exists(os.path.join(pair_corr_folder, "pred_newpose.txt")):
                pred_pose = np.loadtxt(
                    os.path.join(pair_corr_folder, "pred_newpose.txt")
                )
            else:  # sgpgm setting
                pred_pose = np.eye(4)
                gt_pose = np.eye(4)

            corr_src_pcd = o3d.io.read_point_cloud(
                os.path.join(pair_corr_folder, "corr_src.ply")
            )
            corr_ref_pcd = o3d.io.read_point_cloud(
                os.path.join(pair_corr_folder, "corr_ref.ply")
            )
            ir, corr_tp_mask = evaluate_fine(
                np.asarray(corr_src_pcd.points),
                np.asarray(corr_ref_pcd.points),
                gt_pose,
            )
            # print(gt_pose)

            print("IR: {:.3f}".format(ir))
            rmse = eval_registration_error(src_pcd, gt_pose, pred_pose)
            rmse_array.append(rmse)
            ir_array.append(ir)
            corr_number_array.append(corr_tp_mask.shape[0])
            summary_tp_corr.append(corr_tp_mask)

        count += 1
        # break

    ######
    print("Registered {} pairs".format(count))

    if len(rmse_array) > 0:
        rmse_array = np.array(rmse_array)
        ir_array = 100 * np.array(ir_array)
        corr_number_array = np.array(corr_number_array)
        summary_tp_corr = np.concatenate(summary_tp_corr, axis=0)

        register_recall = rmse_array < RMSE_THRESHOLD
        print(
            "Register Recall: {:.3f}({}/{}), Inlier ratio: {:.1f}".format(
                register_recall.mean(),
                np.sum(register_recall),
                len(rmse_array),
                ir_array.mean(),
            )
        )
        # summary_tp_corr.mean()))

        out_result = np.hstack(
            (
                rmse_array.reshape(-1, 1),
                ir_array.reshape(-1, 1),
                corr_number_array.reshape(-1, 1),
            )
        )

        from IO import write_scenes_results

        write_scenes_results(
            os.path.join(dataroot, "output", output_setting, "registration.txt"),
            [pair[0] + "-" + pair[1] for pair in scan_pairs],
            out_result,
            header="# scene_pair rmse ir",
        )
