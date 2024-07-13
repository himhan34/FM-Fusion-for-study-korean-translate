import os, glob
import subprocess
from eval_loop import eval_offline_register

if __name__ == "__main__":
    # args
    dataroot = "/data2/sgslam"
    ########
    # The three folders should be downloaded from OneDrive
    output_folder = os.path.join(dataroot, "output", "two_agent+")
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

        for frame_dir in sorted(frames_dirs):
            if "cmatches" in frame_dir:
                continue
            frame_name = os.path.basename(frame_dir).split(".")[0]
            print("   --- processing frame: {} ---".format(frame_name))

            cmd = "{} --output_folder {} --gt_folder {} --export_folder {} --src_scene {} --ref_scene {} --frame_name {}".format(
                exe_dir,
                output_folder,
                gt_folder,
                pair_export_folder,
                src_scene,
                ref_scene,
                frame_name,
            )

            subprocess.run(cmd, stdin=subprocess.PIPE, shell=True)

            os.system(
                "cp {} {}".format(
                    frame_dir, os.path.join(pair_export_folder, frame_name + ".txt")
                )
            )

            # break

        # break

    # Evaluation
    print("******** Evaluate All ***********")
    eval_offline_register(export_folder, gt_folder, scan_pairs, True, gt_iou_folder)
