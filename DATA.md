# Data Format

## 1. Input RGB-D Sequence
In SgSlam and ScanNet, their data sequence are collected from RGB-D sensors.
The data format is as follow,
```bash
|---ab0201_03a
    |---color
    |---depth
    |---pose
    |---intrinsic
    |---prediction_no_augment
        |---frame-000000_label.json # Image taging and detection results
        |---frame-000000_mask.png # Pixel-wise instance mask
        |--- ...
```
In the sequence name ```ab0201_03a```, ```ab0201``` is a scene id and ```03a``` is a scanning index.

## 2. Output Results
At the end of each sequence, the mapping node save the results in a pre-defined output folder. The output format is as below.
```bash
|---output
    |---ab0201_03a
        |---instance_map.ply # points are colored instance-wise
        |---instance_info.txt
        |---instance_box.txt
        |---${idx}.ply (intanace-wise point cloud)
        |--- ...
```

## 3. Dataset Folder
We organize the dataset folder as below. 
```bash
${SGSLAM_DATAROOT}
|---splits
    |---val.txt
|---scans
    |---ab0201_03a
    |--- ...
|---output
    |---online_mapping # mapping results
```
```ScanNet``` dataset folder is organized in the same way.