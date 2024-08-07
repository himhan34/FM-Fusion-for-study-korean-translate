import os, glob
import subprocess
from eval_loop import (
    eval_offline_register,
    read_frames_map,
    find_closet_index,
    read_match_centroid_result,
)

if __name__ == "__main__":
    # args
    dataroot = "/data2/sgslam"
    ########
    # The three folders should be downloaded from OneDrive
    output_folder = os.path.join(dataroot, "output", "v6")
    gt_folder = os.path.join(dataroot, "gt")
    TF_SOLVER = 'gnc' # 'gnc' or 'quadtro'
    ########

    cfg_file = 'config/realsense.yaml'
    export_folder = os.path.join(dataroot, "output", "offline_register_quatro")
    scan_pairs = [
        ["uc0110_00a", "uc0110_00b"],
        ["uc0110_00a", "uc0110_00c"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00b"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00c"],
        ["uc0204_00a", "uc0204_00b"],  # opposite trajectory
        ["uc0204_00a", "uc0204_00c"],
        ["uc0111_00a", "uc0111_00b"],
        ["ab0201_03c", "ab0201_03a"],
        ["ab0302_00a", "ab0302_00b"],
        ["ab0401_00a", "ab0401_00b"],
        ["ab0403_00c", "ab0403_00d"],
    ]
    exe_dir = "build/cpp/TestRegister"

    if os.path.exists(export_folder) == False:
        os.makedirs(export_folder)

    for pair in scan_pairs:
        print("******** processing pair: {} ***********".format(pair))
        src_scene = pair[0]
        ref_scene = pair[1]
        frames_dirs = glob.glob(
            os.path.join(output_folder, src_scene, ref_scene, "frame*.txt")
        )
        pair_export_folder = os.path.join(export_folder, src_scene + "-" + ref_scene)
        if os.path.exists(pair_export_folder) == False:
            os.makedirs(pair_export_folder)

        # ref_maps_dir = glob.glob(output_folder + "/" + ref_scene + "/fakeScene/*_src.ply")
        ref_maps = read_frames_map(os.path.join(output_folder, ref_scene, "fakeScene"))

        for frame_dir in sorted(frames_dirs):
            if "cmatches" in frame_dir:
                continue
            _, _, _, ref_timestamp = read_match_centroid_result(frame_dir)
            ref_frame_id = int((ref_timestamp - 12000) / 0.1)
            frame_name = os.path.basename(frame_dir).split(".")[0]
            src_frame_id = int(frame_name.split("-")[-1])
            print(
                "--- processing src frame: {}, ref frame: {}---".format(
                    frame_name, ref_frame_id
                )
            )
            ref_map_dir = find_closet_index(
                ref_maps["indices"], ref_maps["dirs"], ref_frame_id
            )

            cmd = "{} --config {} --output_folder {} --gt_folder {} --export_folder {} --src_scene {} --ref_scene {} --frame_name {} --ref_frame_map_dir {}".format(
                exe_dir,
                cfg_file,
                output_folder,
                gt_folder,
                pair_export_folder,
                src_scene,
                ref_scene,
                frame_name,
                ref_map_dir,
            )
            cmd +=" --tf_solver {}".format(TF_SOLVER)
            
            print(cmd)
            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

            os.system(
                "cp {} {}".format(
                    frame_dir, os.path.join(pair_export_folder, frame_name + ".txt")
                )
            )

            # break


    # Evaluation
    print("******** Evaluate All ***********")
    # eval_offline_register(export_folder, gt_folder, scan_pairs, True, output_folder)
