import os, glob
import subprocess
import time
import numpy as np
import open3d as o3d

from eval_loop import (
    eval_offline_register,
    read_frames_map,
    find_closet_index,
    read_match_centroid_result,
)

def evaluate_reg_time(output_folder):
    
    files = glob.glob(output_folder + "/*/*_timing.txt")
    coarse_time_array = []
    dense_time_array = []
    header = ['clique','graph','solver','verify','g3reg','icp']
    
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            eles = lines[1].strip().split(" ")
            time_array = np.array([float(e.strip()) for e in eles[1:]])
            if eles[0] == '0': # coarse
                coarse_time_array.append(time_array)
            else:
                dense_time_array.append(time_array)
                                        
    coarse_time_array = np.array(coarse_time_array)
    dense_time_array = np.array(dense_time_array)

    msg = ''.join([f"{key} " for key in header])
    msg += "\n"
    msg +='Coarse: {} frames '.format(coarse_time_array.shape[0])
    
    for i, key in enumerate(header):
        msg += f"{np.mean(coarse_time_array[:,i]):.1f} "
    msg += "\n"
    msg +='Dense: {} frames '.format(dense_time_array.shape[0])
    for i, key in enumerate(header):
        msg += f"{np.mean(dense_time_array[:,i]):.1f} "


    print(msg)

if __name__ == "__main__":
    # args
    dataroot = "/data2/sgslam"
    ########
    # The three folders should be downloaded from OneDrive
    output_folder = os.path.join(dataroot, "output", "v9")
    gt_folder = os.path.join(dataroot, "gt")
    TF_SOLVER = "gnc"  # 'gnc' or 'quadtro'
    RUN_LAZY_ICP = False
    ########

    cfg_file = "config/realsense.yaml"
    new_result_folder = os.path.join(dataroot, "output", "offline_refine")
    scan_pairs = [
        # ["uc0110_00a", "uc0110_00b"],
        # ["uc0110_00a", "uc0110_00c"],  # opposite trajectory
        # ["uc0115_00a", "uc0115_00b"],  # opposite trajectory
        # ["uc0115_00a", "uc0115_00c"],
        # ["uc0204_00a", "uc0204_00b"],  # opposite trajectory
        # ["uc0204_00a", "uc0204_00c"],
        # ["uc0111_00a", "uc0111_00b"],
        # ["ab0201_03c", "ab0201_03a"],
        # ["ab0302_00a", "ab0302_00b"],
        # ["ab0401_00a", "ab0401_00b"],
        # ["ab0403_00c", "ab0403_00d"],
    ]
    exe_dir = "build/cpp/TestRegister"
    coarse_icp_times = []
    dense_icp_times = []

    if os.path.exists(new_result_folder) == False:
        os.makedirs(new_result_folder)

    for pair in scan_pairs:
        print("******** processing pair: {} ***********".format(pair))
        src_scene = pair[0]
        ref_scene = pair[1]
        frames_dirs = glob.glob(
            os.path.join(output_folder, src_scene, ref_scene, "frame*.txt")
        )
        scene_new_result = os.path.join(new_result_folder, src_scene + "-" + ref_scene)
        if os.path.exists(scene_new_result) == False:
            os.makedirs(scene_new_result)

        # ref_maps_dir = glob.glob(output_folder + "/" + ref_scene + "/fakeScene/*_src.ply")
        ref_maps = read_frames_map(os.path.join(output_folder, ref_scene, "fakeScene"))

        for frame_dir in sorted(frames_dirs):
            if "cmatches" in frame_dir or "centroids" in frame_dir:
                continue
            print(frame_dir)
            _, _, _, ref_timestamp = read_match_centroid_result(frame_dir)
            ref_frame_id = int((ref_timestamp - 12000) / 0.1)
            frame_name = os.path.basename(frame_dir).split(".")[0]
            src_frame_id = int(frame_name.split("-")[-1])
            frame_cmatch_file = os.path.join(
                os.path.dirname(frame_dir), frame_name + "_cmatches.txt"
            )
            if os.path.exists(frame_cmatch_file):
                dense_mode = True
            else:
                dense_mode = False
            
            print(
                "--- processing src frame: {}, ref frame: {}---".format(
                    frame_name, ref_frame_id
                )
            )
            ref_map_dir = find_closet_index(
                ref_maps["indices"], ref_maps["dirs"], ref_frame_id
            )

            cmd = "{} --config {} ".format(exe_dir, cfg_file)
            cmd += "--result_folder {} --gt_folder {} --new_result_folder {} ".format(
                output_folder,
                gt_folder,
                scene_new_result
            )
            cmd += "--src_scene {} --ref_scene {} --frame_name {} --ref_frame_map_dir {} ".format(
                src_scene, ref_scene, frame_name, ref_map_dir
            )
            
            cmd +="--nms_thd 0.1 "
            cmd += "--icp_voxel 0.5 "
            cmd += "--inlier_threshold 0.3 "
            cmd += "--ds_num 1 "
            cmd += "--max_corr_number 300 "
            
            # cmd += "--enable_icp --search_radius 0.1 "
            cmd += "--downsample_corr "

            print(cmd)
            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

            # os.system(
            #     "cp {} {}".format(
            #         frame_dir, os.path.join(scene_new_result, frame_name + ".txt")
            #     )
            # )
            
            if RUN_LAZY_ICP:
                from register_3rscan import icp_register
                print('src map: ', os.path.join(output_folder, pair[0], pair[1], '{}_src.ply'.format(frame_name)))
                
                src_pcd = o3d.io.read_point_cloud(
                    os.path.join(output_folder, pair[0], pair[1], '{}_src.ply'.format(frame_name)))
                ref_pcd = o3d.io.read_point_cloud(ref_map_dir)
                src_pcd.estimate_normals()
                ref_pcd.estimate_normals()
                
                init_pose = np.loadtxt(os.path.join(new_result_folder, pair[0] + "-" + pair[1], frame_name + "_newpose.txt"))
                t0 = time.time()
                new_pose = icp_register(src_pcd, ref_pcd, init_pose, icp_voxel=0.2, threshold=0.5)
                icp_time = time.time() - t0
                print('ICP takes {:.2f} ms'.format(1000*icp_time))
                if dense_mode:
                    dense_icp_times.append(icp_time)
                else:
                    coarse_icp_times.append(icp_time)
            # break


    # Evaluation
    print("******** Evaluate All ***********")
    # eval_offline_register(export_folder, gt_folder, scan_pairs, True, output_folder)
    evaluate_reg_time(new_result_folder)
    
    if RUN_LAZY_ICP:
        print('------- Lazy ICP Time -------')
        dense_icp_times = np.array(dense_icp_times)
        coarse_icp_times = np.array(coarse_icp_times)
        print('Dense ICP: {:.2f} ms'.format(1000*np.mean(dense_icp_times)))
        print('Coarse ICP: {:.2f} ms'.format(1000*np.mean(coarse_icp_times)))
