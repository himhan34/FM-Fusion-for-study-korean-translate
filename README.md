# FM-Fusion

Bayesian fuse the object detectin from RAM-Grounded-SAM into consistent scene graph

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
## Compilation

```bash
mkdir build
cd build
cmake ..
make
```

## Run instance mapping node

Run the executable program as follow,
```bash
./cpp/IntegrateInstanceMap --config ../config/fusion_portable.yaml --root ${FusionPortableRoot}/scans/bday_03 --frame_gap 10 --output ${FusionPortableRoot}/output
```


The reconstructed instances should be saved int the output folder.
```bash
|---output
    |---bday_01
        |---instance_map.ply (aggregated instances point cloud)
        |---instance_info.txt
        |---${idx}.ply (intanace-wise point cloud)
        |--- ...
```

## Run Python version of instance mapping node 
The python verision is based on existed interface of Open3D. It is slower. 
The ```${SCANNET_ROOT}``` folder should be organized like FusionPortable data structure. Target scans should be put in ```split/${SPLIT_FILE_NAME}.txt``` file.
Run the mapping node,
```bash
python scripts/semantic_mapping.py --data_root ${SCANNET_ROOT} --prior_model measurement_model/bayesian --output_folder demo --prediction_folder prediction_no_augment --split val --split_file ${SPLIT_FILE_NAME}
```
The output files should be at ```${SCANNET_ROOT}/output/demo```.

To refine the instances volume,
```bash
python scripts/postfuse.py --dataroot ${SCANNET_ROOT} --split_file ${SPLIT_FILE_NAME} --debug_folder demo --prior_model bayesian --measurement_dir measurement model
```
The output files should be at ```${SCANNET_ROOT}/output/demo_refined```.


