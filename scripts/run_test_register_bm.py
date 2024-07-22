import os, glob
import subprocess
from eval_loop import (
    eval_offline_register,
    read_frames_map,
    find_closet_index,
    read_match_centroid_result,
)
from tqdm import tqdm
from multiprocessing import Pool


def process_pair(args):
    (
        pair,
        output_folder,
        gt_folder,
        gt_iou_folder,
        export_folder,
        exe_dir,
        verify_voxel,
        search_radius,
        icp_voxel,
        ds_voxel,
        ds_num,
    ) = args
    src_scene = pair[0]
    ref_scene = pair[1]
    frames_dirs = glob.glob(
        os.path.join(output_folder, src_scene, ref_scene, "frame*.txt")
    )
    pair_export_folder = os.path.join(export_folder, src_scene + "-" + ref_scene)
    if not os.path.exists(pair_export_folder):
        os.makedirs(pair_export_folder)

    ref_maps = read_frames_map(os.path.join(output_folder, ref_scene, "fakeScene"))

    for frame_dir in tqdm(sorted(frames_dirs), leave=False):
        if "cmatches" in frame_dir:
            continue
        _, _, _, ref_timestamp = read_match_centroid_result(frame_dir)
        ref_frame_id = int((ref_timestamp - 12000) / 0.1)
        frame_name = os.path.basename(frame_dir).split(".")[0]
        src_frame_id = int(frame_name.split("-")[-1])
        ref_map_dir = find_closet_index(
            ref_maps["indices"], ref_maps["dirs"], ref_frame_id
        )

        cmd = "{} --output_folder {} --gt_folder {} --export_folder {} --src_scene {} --ref_scene {} --frame_name {} --ref_frame_map_dir {} --verify_voxel {} --search_radius {} --icp_voxel {} --ds_voxel {} --ds_num {}".format(
            exe_dir,
            output_folder,
            gt_folder,
            pair_export_folder,
            src_scene,
            ref_scene,
            frame_name,
            ref_map_dir,
            verify_voxel,
            search_radius,
            icp_voxel,
            ds_voxel,
            ds_num,
        )

        subprocess.run(cmd, stdin=subprocess.PIPE, shell=True, stdout=subprocess.PIPE)

        os.system(
            "cp {} {}".format(
                frame_dir, os.path.join(pair_export_folder, frame_name + ".txt")
            )
        )


def test_register(
    scan_pairs,
    output_folder,
    gt_folder,
    gt_iou_folder,
    export_folder,
    exe_dir,
    verify_voxel,
    search_radius,
    icp_voxel,
    ds_voxel,
    ds_num,
):
    args = [
        (
            pair,
            output_folder,
            gt_folder,
            gt_iou_folder,
            export_folder,
            exe_dir,
            verify_voxel,
            search_radius,
            icp_voxel,
            ds_voxel,
            ds_num,
        )
        for pair in scan_pairs
    ]

    with Pool() as pool:
        list(tqdm(pool.imap(process_pair, args), total=len(args)))

    eval_offline_register(
        export_folder, gt_folder, scan_pairs, True, gt_iou_folder, verbose=False
    )


if __name__ == "__main__":
    # args
    dataroot = "/data2/sgslam"
    ########
    # The three folders should be downloaded from OneDrive
    output_folder = os.path.join(dataroot, "output", "two_agent_ver3")
    gt_iou_folder = os.path.join(
        dataroot, "output", "gt_iou"
    )  # Pre-saved map to compute iou.
    gt_folder = os.path.join(dataroot, "gt")
    ########

    export_folder = os.path.join(dataroot, "output", "offline_register")
    scan_pairs = [
        ["uc0110_00a", "uc0110_00b"],
        ["uc0110_00a", "uc0110_00c"],  # opposite trajectory
        ["uc0115_00a", "uc0115_00b"],  # opposite trajectory
        ["uc0204_00a", "uc0204_00b"],  # opposite trajectory
        ["uc0204_00a", "uc0204_00c"],
        ["uc0107_00a", "uc0107_00b"],
    ]
    exe_dir = "build/cpp/TestRegister"

    if os.path.exists(export_folder) == False:
        os.makedirs(export_folder)

    # verify_voxel_list = [0.5, 1.0, 1.5, 2]
    verify_voxel_list = [0.5]
    search_radius_list = [0.5]
    icp_voxel_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ds_voxel_list = [0.5]
    ds_num_list = [9]

    for verify_voxel in verify_voxel_list:
        for search_radius in search_radius_list:
            for icp_voxel in icp_voxel_list:
                for ds_voxel in ds_voxel_list:
                    for ds_num in ds_num_list:
                        print(
                            "verify_voxel: {}, search_radius: {}, icp_voxel: {}, ds_voxel: {}, ds_num: {}".format(
                                verify_voxel, search_radius, icp_voxel, ds_voxel, ds_num
                            )
                        )
                        test_register(
                            scan_pairs,
                            output_folder,
                            gt_folder,
                            gt_iou_folder,
                            export_folder,
                            exe_dir,
                            verify_voxel,
                            search_radius,
                            icp_voxel,
                            ds_voxel,
                            ds_num,
                        )
