import os, glob
import subprocess
from sre_parse import parse
import time

from run_test_loop import read_scan_pairs
from eval_loop import eval_registration_error, evaluate_fine
from IO import read_nodes_matches
import open3d as o3d
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nms_thd", type=float, default=0.1)
    parser.add_argument("--ds_num", type=int, default=1)
    return parser.parse_args()


def copy_instance_map(dataroot, scan_pairs, output_folder, include_gt=False):

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
        
        if include_gt:
            gt_pose = np.loadtxt(os.path.join(dataroot, 'val', pair[0], 'transform.txt')).astype(np.float32)
            np.savetxt(os.path.join(dataroot,'gt','{}-{}.txt'.format(pair[0],pair[1])), 
                       gt_pose, fmt='%.6f')
        
        # break
        
def icp_register(src_pcd, ref_pcd,trans_init,icp_voxel=0.2, threshold=0.5):
    if icp_voxel > 0:
        src_icp_pcd = src_pcd.voxel_down_sample(icp_voxel)
        ref_icp_pcd = ref_pcd.voxel_down_sample(icp_voxel)
    else:
        src_icp_pcd = src_pcd
        ref_icp_pcd = ref_pcd
    
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_icp_pcd, 
        ref_icp_pcd, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return reg_p2p.transformation

def eval_pseudo_inlier(rmse_array, inlier_array, pseudo_inlier_array, rmse_thd):
    success_scenes = rmse_array < rmse_thd
    
    # success scenes
    A = pseudo_inlier_array[success_scenes]<inlier_array[success_scenes]
    B = pseudo_inlier_array[success_scenes]==inlier_array[success_scenes]
    C = pseudo_inlier_array[success_scenes]>inlier_array[success_scenes]
    
    # failed scenes
    D = pseudo_inlier_array[~success_scenes]<inlier_array[~success_scenes]
    E = pseudo_inlier_array[~success_scenes]==inlier_array[~success_scenes]
    F = pseudo_inlier_array[~success_scenes]>inlier_array[~success_scenes]
    
    print('[Success] pseudo inlier < inlier: {}, pseudo inlier = inlier: {}, psuedo > inlier: {}'.format(
        A.sum(), B.sum(), C.sum()))
    print('[Failed]  pseudo inlier < inlier: {}, pseudo inlier = inlier: {}, psuedo > inlier: {}'.format(
        D.sum(), E.sum(), F.sum()))
    

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
    # output_setting = "ours_optimized2"
    # output_setting = 'ours_locked'
    # output_setting = 'ours_gt_trained'
    # output_setting = 'sgpgm_optimized'
    output_setting = 'sgpgm_locked'
    # output_setting = 'geotransformer_ft_fixed'
    # output_setting = 'geotransformer_ft'
    #############################

    # hard-code
    SPLIT = 'small_scenes' # 'small_scenes','median_scenes', 'large_scenes', 
    # 'val_merged, 'val_gtransformer
    
    EXE_DIR = "build/cpp/Test3RScanRegister"
    CFG_FILE = "config/realsense.yaml"
    RMSE_THRESHOLD = 0.2
    RUN_CMD = False
    ICP_ONLY = False # pose-process sgpgm and geotransformer
    OVERWRITE_NODES = False
    EVAL = True
    EVAL_PSEUDO = True
    EVAL_TIMING = False
    
    POSE_FILE = 'pred_pose.txt' 
    # POSE_FILE = 'pred_newpose.txt'
    if 'ours' in output_setting:
        POSE_FILE = 'pred_newpose.txt'
    if ICP_ONLY:
        POSE_FILE = 'pred_pose_icp.txt'

    #
    scan_pairs = read_scan_pairs(os.path.join(dataroot, "splits", SPLIT + ".txt"))
    # scan_pairs = scan_pairs[:10]
    # scan_pairs = [['scene0104_00h', 'scene0104_00a']]

    count = 0
    rmse_array = []
    ir_array = []
    inlier_number_array = []
    corr_number_array = []
    pseudo_ir_array = []
    pseudo_inlier_number_array = []
    summary_tp_corr = []
    timing_matrix = []
    # copy_instance_map(dataroot, scan_pairs, 
    #                   os.path.join(dataroot, 'output', output_setting),
    #                   include_gt=False)
    # exit(0)

    for pair in scan_pairs:
        print("******** processing pair: {} ***********".format(pair))
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
        cmd += " --max_corr_number {}".format(300)
        cmd += " --inlier_threshold {}".format(0.5)
        # cmd += " --enable_icp"
        cmd += " --ds_num 1"
        cmd += " --nms_thd {}".format(args.nms_thd)
        if RUN_CMD:
            print(cmd)
            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

        if ICP_ONLY:
            src_pcd = o3d.io.read_point_cloud(
                os.path.join(dataroot,'val',pair[0], "instance_map.ply"))
            ref_pcd = o3d.io.read_point_cloud(
                os.path.join(dataroot,'val',pair[1], "instance_map.ply"))
            assert(len(src_pcd.points) > 0)
            if 'ours' in output_setting:
                init_pose = np.loadtxt(os.path.join(pair_corr_folder, "pred_newpose.txt"))
            else:
                init_pose = np.loadtxt(os.path.join(pair_corr_folder, "pred_pose.txt"))
            new_pose = icp_register(src_pcd, ref_pcd, init_pose, 
                                    icp_voxel=0.2, threshold=0.5)
            np.savetxt(os.path.join(pair_corr_folder, "pred_pose_icp.txt"), 
                       new_pose)

        eval_msg = ''
        if EVAL:
            src_pcd = o3d.io.read_point_cloud(
                os.path.join(pair_corr_folder, "src_instances.ply"))
            gt_pose = np.loadtxt(os.path.join(dataroot, 'val', pair[0], 'transform.txt')).astype(np.float32)
            # gt_pose = np.loadtxt(os.path.join(pair_corr_folder, "gt_pose.txt")
            #                      ).astype(np.float32)

            pred_pose = np.loadtxt(os.path.join(pair_corr_folder, POSE_FILE)
                                   ).astype(np.float32)

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
            
            # Export
            eval_msg += '{}-{}, IR: {:.3f}'.format(pair[0],pair[1],ir)
            rmse = eval_registration_error(src_pcd, gt_pose, pred_pose)
            rmse_array.append(rmse)
            ir_array.append(ir)
            corr_number_array.append(corr_tp_mask.shape[0])
            summary_tp_corr.append(corr_tp_mask)
            inlier_number_array.append(corr_tp_mask.sum())
        
        if EVAL_PSEUDO:    
            if output_setting=='ours': # Add centroids to the correspondence
                src_centroids, ref_centroids = read_nodes_matches(
                    os.path.join(pair_corr_folder, "node_matches.txt")
                )
                # print('src_centroids: ', src_centroids.shape)
                raw_corr_src = np.concatenate([np.asarray(corr_src_pcd.points), src_centroids], 
                                                axis=0)
                raw_corr_ref = np.concatenate([np.asarray(corr_ref_pcd.points), ref_centroids],
                                                axis=0)
                # print(raw_corr_ref.shape)
            else:
                raw_corr_src = np.asarray(corr_src_pcd.points)
                raw_corr_ref = np.asarray(corr_ref_pcd.points)
            
            _, filter_corr_mask = evaluate_fine(
                # np.asarray(corr_src_pcd.points),
                # np.asarray(corr_ref_pcd.points),
                raw_corr_src,
                raw_corr_ref,
                pred_pose,
                0.2
            )
            pseudo_corr_src = raw_corr_src[filter_corr_mask]
            pseudo_corr_ref = raw_corr_ref[filter_corr_mask]
            
            pseudo_ir, pseudo_inlier_mask = evaluate_fine(
                pseudo_corr_src,
                pseudo_corr_ref,
                gt_pose,
            )
            
            pseudo_corr_src_pcd = o3d.geometry.PointCloud(
                                                    points=o3d.utility.Vector3dVector(pseudo_corr_src))
            pseudo_corr_ref_pcd = o3d.geometry.PointCloud(
                                                    points=o3d.utility.Vector3dVector(pseudo_corr_ref))
            o3d.io.write_point_cloud(
                os.path.join(pair_corr_folder, "pseudo_corr_src.ply"),
                pseudo_corr_src_pcd,
            )
            o3d.io.write_point_cloud(
                os.path.join(pair_corr_folder, "pseudo_corr_ref.ply"),
                pseudo_corr_ref_pcd,
            )
            
            pseudo_ir_array.append(pseudo_ir)
            pseudo_inlier_number_array.append(pseudo_inlier_mask.sum())
            
            pseudo_inlier_mask = pseudo_inlier_mask.numpy().astype(np.int32)
            np.savetxt(os.path.join(pair_corr_folder, "pseudo_corres_pos.txt"), 
                       pseudo_inlier_mask, fmt='%d')
            eval_msg += ', Pseudo IR: {:.3f}'.format(pseudo_ir)
        
        if EVAL_TIMING:
            timing_data = np.loadtxt(os.path.join(pair_corr_folder, 
                                                  'g3reg_timing.txt'))
            timing_matrix.append(timing_data)
        
        print(eval_msg)
        count += 1
        # break

    ######
    print("******* Registered {} pairs ********".format(count))

    if len(rmse_array) > 0:
        rmse_array = np.array(rmse_array)
        ir_array = 100 * np.array(ir_array)
        corr_number_array = np.array(corr_number_array)
        summary_tp_corr = np.concatenate(summary_tp_corr, axis=0)
        inlier_number_array = np.array(inlier_number_array)
        pseudo_ir_array = 100 * np.array(pseudo_ir_array)
        pseudo_inlier_number_array = np.array(pseudo_inlier_number_array)

        register_recall = rmse_array < RMSE_THRESHOLD
        rmse_metric = rmse_array[register_recall]
        rmse_metric = rmse_metric.mean()
        
        print(
            "Register Recall: {:.3f}({}/{}), Inlier ratio: {:.1f}%, Inlier: {:.1f}, Psu. Inlier: {:.1f}, RMSE: {:.3f}".format(
                register_recall.mean(),
                np.sum(register_recall),
                len(rmse_array),
                100*inlier_number_array.sum()/corr_number_array.sum(),
                inlier_number_array.mean(),
                pseudo_inlier_number_array.mean(),
                # corr_number_array.mean(),
                rmse_metric
            )
        )


        out_result = np.hstack((
                100*rmse_array.reshape(-1, 1),
                ir_array.reshape(-1, 1),
                inlier_number_array.reshape(-1, 1),
                corr_number_array.reshape(-1, 1),
                # pseudo_ir_array.reshape(-1, 1),
                # pseudo_inlier_number_array.reshape(-1, 1),
            ))
        if EVAL_PSEUDO:
            print(
                'Pseudo Inlier ratio: {:.1f}, Pseudo Inlier num: {}'.format(
                pseudo_ir_array.mean(),
                pseudo_inlier_number_array.mean())
            )            
            out_result = np.hstack((
                    out_result,
                    pseudo_ir_array.reshape(-1, 1),
                    pseudo_inlier_number_array.reshape(-1, 1),
                ))
        

        from IO import write_scenes_results

        write_scenes_results(
            os.path.join(dataroot, "output", output_setting, "registration_{}.txt".format(SPLIT)),
            [pair[0] + "-" + pair[1] for pair in scan_pairs],
            out_result,
            header="# scene_pair rmse(cm) ir inliers corrs pir pseudo_inliers",
        )


    if EVAL_PSEUDO:
        assert(isinstance(summary_tp_corr, np.ndarray))
        eval_pseudo_inlier(rmse_array, 
                           inlier_number_array, 
                           pseudo_inlier_number_array, 
                           RMSE_THRESHOLD)
        
    if EVAL_TIMING:
        timing_matrix = np.vstack(timing_matrix)
        
        if False:
            detail_msg = '# Clique, Graph, Solver, Verify, Total\n'
            
            for pair, timing_data in zip(scan_pairs, timing_matrix):
                detail_msg += '{}-{}: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n'.format(
                    pair[0], pair[1], timing_data[0], timing_data[1], timing_data[2], timing_data[3], timing_data[4]
                )

            print(detail_msg)
        
        mean_timing = timing_matrix.mean(axis=0)
        print('Timing header: Clique, Graph, Solver, Verify, Total')
        print('Mean timing: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(
            mean_timing[0], mean_timing[1], mean_timing[2], mean_timing[3], mean_timing[4]
        ))  

