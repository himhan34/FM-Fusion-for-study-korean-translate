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


### Dependencies

Install G3Reg

```angular2html
git pull --recurse-submodules
```

Install our adopted Open3D from source

```bash
git clone git@github.com:glennliu/Open3d_CPU.git
mkdir build & cd build
cmake -DBUILD_SHARED_LIBS=ON -DGLIBCXX_USE_CXX11_ABI=ON -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_FMT=ON ..
make -j12
make install
```

Install GTSAM,

```bash
git clone https://github.com/borglab/gtsam.git
cd gtsam
mkdir build
cd build
cmake .. -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON
sudo make install
```

If loop closured module is not needed, the following the dependencies can be skipped.
Install glog,

```bash
git clone https://github.com/google/glog.git
cd glog && cmake -S . -B build -G "Unix Makefiles"
cmake --build build
sudo cmake --build build --target install
```

Install spdlog,

```bash
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=True .. && make -j
sudo make install
```

Install [utf8proc](github.com:JuliaStrings/utf8proc.git) follow its official repository.

```angular2html
git clone https://github.com/JuliaStrings/utf8proc.git
cd utf8proc && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=True .. && make -j
sudo make install
```

Install libtorch C++ from
Pytorch [download center](https://pytorch.org/get-started/locally/#supported-linux-distributions),

```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.3.0%2Bcu118.zip
or choose the abi version and add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1) in CMakeLists.txt
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu118.zip
```

Uncompresss the downloaded compress file. Our computer settings that related to the Libtorch are,

```
CUDA=11.8
LibTorch=2.3.0
GCC=9.4.0
```

It should also be compatible with other ```Pytorch>=2.0.0```. Depending on the cmake local environment, you may need to
install the ```CXX11 ABI``` version of LibTorch.

### Compilation

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${LIBTORCH_DIR} -DENABLE_TORCH=ON -DLOOP_DETECTION=ON ..
make
```

Set the ```${LIBTORCH_DIR}``` to the unzipped libtorch directory. If the ```Open3D``` and ```spdlog``` are installed in
customized directories, they should also be appended in ```-DCMAKE_INSTALL_PREFIX```.



## Run Loop Detection Node

The trained torchscript model can be downloaded from
the [OneDrive Link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/Encm_4ETKV9EiZ2PRlCLVdEBTCiuBYQ4yckF7SzFTDHg6g?e=oDsTHu).
Download and unzip the torchscript folder and
set the ```--weights_folder``` accordingly.

```bash
./build/cpp/TestLoop --config config/realsense.yaml --weights_folder ${TORCHSCRIPT_FOLDER} --ref_scene /data2/sgslam/val/uc0107_00a --src_scene /data2/sgslam/val/uc0107_00b --output_folder ${OUTPUT_FOLDER}
```

It has optional running options ```--prune_instance```, which prune instance match results by maximum clique. And
option ```--dense_match``` enable searching point cloud correspondences.
In [realsense.yaml](config/realsense.yaml), a few parameters can directly affect the final performance,

- ```LoopDetector.fuse_shape```: decide fuse shape features or not.
- ```Graph.ignore_labels```: incorporate "floor" to ignore them.

The ```ref_scene``` and ```src_scene``` options can be changed to any scene folder directories. The node read scene
graphs from the two scene, and generate instance-wise association results. Match and registration result will be saved
at ```OUTPUT_FOLDER```.

## Run all the scan pairs and evaluate

To run all the 14 scene pairs,

```bash
python scripts/run_test_loop.py
```

Notice to set ```OUTPUT_FOLDER``` before running.

Then, evaluate the match and registration results,

```bash
python scripts/eval_loop.py --dataroot ${DATAROOT} --output_folder ${OUTPUT_FOLDER} --match_folder ${MATCH_FOLDER}
```

The ```OUTPUT_FOLDER``` is the same as the option in running ```TestLoop```.

The latest evaluation result is saved [here](eval/v2_dense.txt).

## Run Loop Detection Node on ROS

```
source catkin_ws/devel/setup.bash
roslaunch sgloop_ros testloop.launch
roslaunch sgloop_ros visualize.launch
```

Currently, it is similar to the ```TestLoop``` program in the last step.
It read a pair of scene graph and register them. The result is visualized on Rviz.
