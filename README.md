<div align="center">
    <!-- <h1>⚽ FM-Fusion</h1> -->
    <h2>FM-Fusion: Instance-aware Semantic Mapping Boosted by Vision-Language Foundation Models</h2>
    <strong>IEEE RA-L 2024</strong>
    <br>
        <a href="https://uav.hkust.edu.hk/current-members/" target="_blank">Chuhao Liu</a><sup>1</sup>,
        <a href="https://uav.hkust.edu.hk/current-members/" target="_blank">Ke Wang</a><sup>2,†</sup>,
        <a href="https://jayceeshi.github.io/" target="_blank">Jieqi Shi</a><sup>1</sup>,
        <a href="https://qiaozhijian.github.io/" target="_blank">Zhijian Qiao</a><sup>1</sup>, and
        <a href="https://uav.hkust.edu.hk/group/" target="_blank">Shaojie Shen</a><sup>2</sup>
    <p>
        <h45>
            <sup>1</sup>HKUST Aerial Robotics Group &nbsp;&nbsp;
            <sup>2</sup>Chang'an University, China &nbsp;&nbsp;
            <br>
        </h5>
        <sup>†</sup>Corresponding Author
    </p>
    <a href="https://ieeexplore.ieee.org/abstract/document/10403989"> <img src="https://img.shields.io/badge/IEEE-RA--L-004c99"> </a>
    <a href='https://arxiv.org/abs/2402.04555'><img src='https://img.shields.io/badge/arXiv-2402.04555-990000' alt='arxiv'></a>
    <a href="https://www.youtube.com/watch?v=zrzcjj9-ydk&t=9s"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-Video-red"/></a>
</div>


<!-- ## Introduction -->

Bayesian fuse the object detectin from RAM-Grounded-SAM into consistent scene graph

## Tabel of Contents
1. Install
2. Download Data
3. Run Instance-aware Semantic Mapping
4. Use it in Your Application
5. Acknowledge

## 1. Install
Install dependency packages from Ubuntu Source
```bash
sudo apt-get install libboost-dev libomp-dev libeigen3-dev
```
Install Open3D from its source code.([Install Tutorial](https://www.open3d.org/docs/release/compilation.html#compilation))
```bash
git clone https://github.com/isl-org/Open3D
cd Open3D
make build & cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j12
sudo make install
```
Follow the official tutorials to install [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html), [GLOG](https://github.com/google/glog), [jsoncpp](https://github.com/open-source-parsers/jsoncpp/blob/master/README.md).

Compile FM-Fusion,
```bash
mkdir build & cd build
cmake ..
make -j12
make install
```

Install the ROS node program, which renders the semantic instance map in Rviz. Firstly, install ROS follow the [official guidance](http://wiki.ros.org/noetic/Installation/Ubuntu). Then, 
```bash
git submodule update --init --recursive
cd catkin_ws & catkin_make
source devel/setup.bash
```

## 2. Download Data
We provide two datasets to evaluate: SgSlam and ScanNet. Their sequences can be downloaded:
* [SgSlam_OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/EnIjnIl7a2NBtioZFfireM4B_PnZcIpwcB-P2-Fnh3FEPg?e=BKsgLQ).
* [ScanNet_OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/EhUu2zzwUGNKq8h2yUGHIawBpC8EI73YB_GTG9BUXXBoVA?e=o4TfsA).

The instruction of the data format can be found [here](DATA.md).
After download the ```scans``` folder in each dataset, go to [uncompress_data.py](scripts/uncompress_data.py) and set the data directories. Then, uncompress the sequence data.
```
python scripts/uncompress_data.py
```

## 3. Run Instance-aware Semantic Mapping
#### a. Run with Rviz.

Check the parameter settting. Then, launch the ROS  program,
```
roslaunch sgloop_ros visualize.launch % rviz
roslaunch sgloop_ros semantic_mapping.launch
```
It should incremental reconstruct the semantic map and render the results on Rviz. The output results are illustrated in the [data format](DATA.md).

#### b. Run without visualization.

If you do not need the ROS node to visualize, you can skip its install in the above instruction. Then, simply run the C++ executable program and the results will be saved at ```${SGSLAM_DATAROOT}/output```. The output directory can be set in running the program.
```bash
./build/cpp/IntegrateInstanceMap --config config/realsense.yaml --root ${SGSLAM_DATAROOT}/scans/ab0201_03a --prediction prediction_no_augment --frame_gap 2 --output ${SGSLAM_DATAROOT}/output
```

## 4. Use it in Your Application
In ```SgSlam``` dataset, we use Intel Realsense-D515 camera and DJI A3 flight controller to collect data sequence Details of the hardware suite can be found in this [paper](https://arxiv.org/abs/2201.03312). You can also collect your own dataset using a similar hardware suite. 
#### a. Prepare RGB-D and Camera poses.
We use [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) to compute visual-inertial odometry (VIO). We save the camera poses of its keyframes in a ```pose``` folder.

#### b. Run [RAM](https://github.com/xinyu1205/recognize-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything).

The three models are combined to run in [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). Please find our adopted Grounded-SAM [here](https://github.com/glennliu/Grounded-Segment-Anything). It should generate a ```prediction``` folder as explained in [data format](DATA.md). Then, you can run the semantic mapping on your dataset.

## 5. Acknowledge
The hardware is supported by [Luqi Wang](https://lwangax.wordpress.com).
We use [Open3D](https://www.open3d.org) to reconstruct instance sub-volume. The vision foundation models RAM, GroundingDINO, and SAM provide instance segmentation on images.

## 6. License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.
For technical issues, please contact Chuhao LIU (cliuci@connect.ust.hk). 
