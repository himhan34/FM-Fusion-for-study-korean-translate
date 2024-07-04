## Test registration

### 1. Evaluate
The online loop results are updated in [Onedrive space](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/Encm_4ETKV9EiZ2PRlCLVdEBy7PSe8TFejZo1LsV9Xydvg). Download and save them in a local directory ```${DATAROOT}```. 
- download ```two_agents+``` from ```${DATAROOT}/output```.
- download ```gt_iou``` from ```${DATAROOT}/output```.
- download ```gt``` folder from ```${DATAROOT}```.
- download ```multi_agents.txt``` from ```${DATAROOT}/splits```.

They should be organized,
```
|--- ${DATAROOT}
    |--- output
        |--- gt_iou
        |--- two_agents+
    |--- splits
        |--- multi_agent.txt
    |--- gt # ground-truth transformation.
```

Then, run evaluation

```
python scripts/eval_loop.py --consider_iou --run_mode online --split_file multi_agent.txt --dataroot ${DATAROOT} --output_folder ${DATAROOT}/output/two_agent+
```

Results should be printed.


### 2. Register
Using the downloaded data to refine the registration.

```bash
cd build
cmake .. & make -j12
python scripts/run_test_register.py
```

It should load the online loop results at each frame of each pair of scenes. Then the reigistration can be refined at ```cpp/TestRegister.cpp```.
