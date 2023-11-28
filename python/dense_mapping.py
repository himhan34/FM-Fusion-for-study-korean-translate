import os, glob
import argparse
import open3d as o3d
import numpy as np
import cv2
import project_util
import fuse_detection
import render_result
import time

def integreate_tsdf_volume(scene_dir, dataset, frame_gap, resolution, visualize):
    scene_name = os.path.basename(scene_dir)
    
    print('Processing {}, with {} resolution'.format(scene_name, resolution))
    assert os.path.exists(os.path.join(scene_dir,'intrinsic')), 'intrinsic file not found'
    DATASET = dataset
    
    if dataset == 'scannet':
        RGB_FOLDER = 'color'
        RGB_POSFIX = '.jpg'
        DEPTH_SCALE = 1000.0
        DEPTH_SDF_TRUC = 4.0
        SDF_TRUCT = 0.04
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True)    
    elif dataset =='fusionportable':
        RGB_FOLDER = 'color'
        RGB_POSFIX = '.png'
        DEPTH_SCALE = 1000.0
        DEPTH_SDF_TRUC = 20.0
        SDF_TRUCT = 0.5
        
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True, verbose=True)    
    else:
        raise NotImplementedError

    # Init
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(depth_dim[1],depth_dim[0],K_depth[0,0],K_depth[1,1],K_depth[0,2],K_depth[1,2])
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / resolution,
        sdf_trunc=SDF_TRUCT,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    #
    tmp_dir = os.path.join(scene_dir,'tmp')
    if os.path.exists(tmp_dir)==False: os.makedirs(tmp_dir)
    
    # 
    depth_frames = sorted(glob.glob(os.path.join(scene_dir,'depth','*.png')))
    print('find {} frames'.format(len(depth_frames)))
    
    for depth_dir in depth_frames:
        frame_name = os.path.basename(depth_dir).split('.')[0] 
        if DATASET=='scannet':
            frame_stamp = float(frame_name.split('-')[-1])
        elif DATASET=='fusionportable':
            frame_stamp = float(frame_name)
        else:
            raise NotImplementedError
        if frame_stamp % frame_gap != 0:
            continue
        print('integrating frame {}'.format(frame_name))
        
        # Load RGB-D and pose
        rgbdir = os.path.join(scene_dir,RGB_FOLDER,frame_name+RGB_POSFIX)
        pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        if os.path.exists(pose_dir)==False:
            print('no pose file for frame {}. Stop the fusion.'.format(frame_name))
            break

        rgb = o3d.io.read_image(rgbdir)
        depth = o3d.io.read_image(depth_dir)
        T_wc = np.loadtxt(pose_dir)
        
        #
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,depth,depth_scale=DEPTH_SCALE,depth_trunc=DEPTH_SDF_TRUC,convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_wc))

        ###### Check depth cloud. for debug only #####
        # depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(rgbd.depth,intrinsic,np.linalg.inv(T_wc),depth_scale=DEPTH_SCALE,depth_trunc=DEPTH_SDF_TRUC)
        # print('depth_pcd has {} points'.format(np.asarray(depth_pcd.points).shape[0]))
        # o3d.io.write_point_cloud(os.path.join(tmp_dir,frame_name+'.ply'),depth_pcd)
        # break


    # Save volume
    # mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(os.path.join(scene_dir,'pcd_o3d_{:.0f}.ply'.format(resolution)),pcd)
    # o3d.io.write_triangle_mesh(os.path.join(scene_dir,'mesh_o3d_{:.0f}.ply'.format(resolution)),mesh)
    
    print('{} finished'.format(scene_name))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--dataset', help='scannet or fusionportable', default='scannet')
    parser.add_argument('--frame_gap', help='frame gap', default=10, type=int)
    parser.add_argument('--resolution', help='resolution of a block', default=256, type=float)
    parser.add_argument('--split', help='split', default='val')
    parser.add_argument('--split_file', help='split file name', default='val')
    opt = parser.parse_args()

    scans = fuse_detection.read_scans(os.path.join(opt.data_root,'splits','{}.txt'.format(opt.split_file)))
    print('Read {} scans to construct map'.format(len(scans)))
    
    for scan in scans:
        scan_dir = os.path.join(opt.data_root, opt.split, scan)
        integreate_tsdf_volume(scan_dir, opt.dataset, opt.frame_gap, opt.resolution, visualize=False)
        # break
    
    