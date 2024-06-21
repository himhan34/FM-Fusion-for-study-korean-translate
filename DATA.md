## SgSlam data structure
Construt the dataset folder as follows,
```bash
sgslam
|---splits
    |---val.txt
|---scans
    |---uc0110_00a
        |---rgb
        |---depth
        |---pose
        |---prediction
        |---intrinsic
|---output
    |---online_coarse # coarse scene graph registration results
    |---${SRC_SCENE}
        |---${REF_SCENE}
            |--- FRAME_XXXX.txt # loop results at the corresponding frame. Estimated pose is T_ref_src.
            |--- FRAME_XXXX_src.ply # local instance point cloud at the frame.
        |--- *.ply # Reconstucted instances of the source scene.
|---gt # 
    |---${SRC_SCENE}-${REF_SCENE}.txt # T_ref_src
|---val
    |---uc0110_00a
        |---instance_info.txt
        |---instance_box.txt
        |---transform.txt # ground-truth transformation to reference scene
        |---edges.pth # for sgnet pytorch 
        |---nodes.csv # for sgnet pytorch
        |---xyzi.pth # [x,y,z,instance] for sgnet pytorch
        |---instance_map.ply # global point cloud colored in instances
        |---global_pcd.ply # global rgb point cloud (denser than instance_map)
        |---${XXXX}.ply # instance point cloud
        |---${XXXX}.ply
        |---...
    |---...
|---matches # ground-truth instance matches
|---output # test loop result and mapping result
```

## FusionPortable data structure
Run traditional dense volumetric fusion using Open3d on FusionPortable dataset. 

Construt the dataset folder as follows,
```bash
FusionPortable
|---splits
    |---val.txt
|---scans
    |---bday_01
        |---color
        |---depth
        |---pose
        |---intrinsic
        |---prediction_no_augment(generate from RAM-Grounded-SAM)
    |---...
|---output
```
The ```split``` folder defines the list of scans. 

Run the dense mapping module as follows,

```bash
python scripts/dense_mapping.py --data_root /data2/FusionPortable/ --dataset fusionportable --resolution 256.0 --split scans --split_file val
```