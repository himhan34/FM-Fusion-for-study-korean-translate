# FM-Fusion

Bayesian fuse the object detectin from RAM-Grounded-SAM into consistent scene graph

## Install

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
Pytorch=2.3.1
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
Set the ```${LIBTORCH_DIR}``` to the unzipped libtorch directory. If the ```Open3D``` and ```spdlog``` are installed in customized directories, they should also be appended in ```-DCMAKE_INSTALL_PREFIX```. 
Depends on your computer environment, you may comment or adjust ```set(CMAKE_CXX_STANDARD 17)``` in ```CMakeLists.txt```.

## Run Loop Detection Node

```bash
./build/cpp/TestLoop --config config/realsense.yaml --weights_folder ${TORCHSCRIPT_FOLDER} --ref_scene /data2/sgslam/val/uc0107_00a --src_scene /data2/sgslam/val/uc0107_00b
```

The ```ref_scene``` and ```src_scene``` options can be changed to any scene folder directories. The node read scene
graphs from the two scene, and generate instance-wise association results.

The trained torchscript model can be downloaded from
the [NutSpace Link](https://www.jianguoyun.com/p/DWQgL7oQpKe2Chi8ntEFIAA). Download and unzip the torchscript folder and
set the ```--weights_folder``` accordingly.

## Run instance mapping node

The mapping node is used to reconstructe 3D semantic instances. It does not considered any loop closures.
Run the executable program as follow,

```bash
./cpp/IntegrateInstanceMap --config ../config/fusion_portable.yaml --root ${FusionPortableRoot}/scans/bday_03 --prediction prediction_no_augment --frame_gap 10 --output ${FusionPortableRoot}/output
```

The reconstructed instances should be saved int the output folder.

```bash
|---output
    |---bday_01
        |---instance_map.ply (aggregated instances point cloud)
        |---instance_info.txt
        |---instance_box.txt
        |---${idx}.ply (intanace-wise point cloud)
        |--- ...
```

## Visualize reconstructed instances

```
python dataset/scannetv2/generate_gt_association.py --dataroot /data2/ScanNet --graphroot /data2/ScanNetGraph --split val --split_file val_clean --min_matches 4
```

<!-- 
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

 -->
