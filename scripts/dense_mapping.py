import os, glob
import argparse
import open3d as o3d
import numpy as np
import cv2
import project_util
import fuse_detection
import render_result
import time

def integreate_tsdf_volume(scene_dir, dataroot, resolution, visualize):
    scene_name = os.path.basename(scene_dir)
    
    print('Processing {}, with {} resolution'.format(scene_name, resolution))
    assert os.path.exists(os.path.join(scene_dir,'intrinsic')), 'intrinsic file not found'
    
    if 'ScanNet' in dataroot:
        DEPTH_SCALE = 1000.0
        RGB_FOLDER = 'color'
        RGB_POSFIX = '.jpg'
        DATASET = 'scannet'
        # FRAME_GAP = 20
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True)    

    else:
        raise NotImplementedError

    # Init
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(depth_dim[1],depth_dim[0],K_depth[0,0],K_depth[1,1],K_depth[0,2],K_depth[1,2])
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / resolution,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    # 
    depth_frames = sorted(glob.glob(os.path.join(scene_dir,'depth','*.png')))
    print('find {} frames'.format(len(depth_frames)))
    
    for depth_dir in depth_frames:
        frame_name = os.path.basename(depth_dir).split('.')[0] 
        if DATASET=='scannet':
            frame_stamp = float(frame_name.split('-')[-1])
        else:
            raise NotImplementedError
        print('integrating frame {}'.format(frame_name))
        
        # Load RTB-D and pose
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
            rgb,depth,depth_scale=DEPTH_SCALE,depth_trunc=4.0,convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_wc))

    # Save volume
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(os.path.join(scene_dir,'mesh_o3d_{:.0f}.ply'.format(resolution)),mesh)
    print('{} finished'.format(scene_name))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--resolution', help='resolution of a block', default=256, type=float)
    parser.add_argument('--split', help='split', default='val')
    parser.add_argument('--split_file', help='split file name', default='val')
    opt = parser.parse_args()

    scans = fuse_detection.read_scans(os.path.join(opt.data_root,'splits','{}.txt'.format(opt.split_file)))
    print('Read {} scans to construct map'.format(len(scans)))
    
    for scan in scans:
        scan_dir = os.path.join(opt.data_root, opt.split, scan)
        integreate_tsdf_volume(scan_dir, opt.data_root, opt.resolution, visualize=False)
        # break
    
    