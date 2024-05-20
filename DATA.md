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