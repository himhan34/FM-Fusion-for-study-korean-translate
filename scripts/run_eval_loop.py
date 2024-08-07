import os 
from eval_loop import eval_online_loop



if __name__=='__main__':
    ###### Paramters ######
    dataroot = "/data2/sgslam"
    output_folder = os.path.join(dataroot, "output", "v8")
    CONSIDER_IOU = True
    #######################
    
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
    
    eval_online_loop(dataroot, scan_pairs, output_folder, CONSIDER_IOU)
    
    
    