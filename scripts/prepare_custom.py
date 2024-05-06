import os, glob
import numpy as np
import open3d as o3d
import zipfile
import subprocess
import cv2
from scipy.spatial.transform import Rotation as R

from prepare_datasets import read_scans
from prepare_3rscan import read_scan_pairs

def write_intrinsic(scan_dir):
    '''
    Write intrsinci parameters to the scans collected using Realsense Camera
    '''
    intrinsic_folder = os.path.join(scan_dir,'intrinsic')
    if os.path.exists(intrinsic_folder)==False:
        os.mkdir(intrinsic_folder)
        
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    FX = 619.18
    FY = 618.17
    CX = 336.51
    CY = 246.95
    K = np.eye(4)
    K[0,0] = FX
    K[1,1] = FY
    K[0,2] = CX
    K[1,2] = CY
    
    with open(os.path.join(intrinsic_folder,'sensor_shapes.txt'),'w') as f:
        f.write('color_width:{}\n'.format(IMG_WIDTH))
        f.write('color_height:{}\n'.format(IMG_HEIGHT))
        f.write('depth_width:{}\n'.format(IMG_WIDTH))
        f.write('depth_height:{}\n'.format(IMG_HEIGHT))
        f.close()
        
    np.savetxt(os.path.join(intrinsic_folder,'intrinsic_depth.txt'),K,fmt='%.6f')
    print('intrinsic saved')
    
# todo: apply pose drift to the scans
def apply_pose_drift(scan_dir):
    pass

def move_dense_map(scan_folder,graph_folder):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(os.path.join(scan_folder,'mesh_o3d.ply'))
    o3d.io.write_point_cloud(os.path.join(graph_folder,'pcd_o3d.ply'),pcd)

def transform_coordinate(T_w_c:np.ndarray,T_c_tag:np.ndarray):
    R_w_tag = T_w_c[:3,:3] @ T_c_tag[:3,:3]
    t_w_tag = T_w_c[:3,:3] @ T_c_tag[:3,3] + T_w_c[:3,3]
    T_w_tag =  np.eye(4)
    T_w_tag[:3,:3] = R_w_tag
    T_w_tag[:3,3] = t_w_tag

    return T_w_tag

def transform_instance_boxes(T, instance_box_file, out_file):
    '''
    Read and transform the instance boxes accordingly. 
    Results are saved.
    '''
    instances_box = {}
    header = ''
    with open(instance_box_file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if'#' in line:
                header = line
                continue
            parts = line.split(';')
            idx = parts[0]
            center = np.array([float(x) for x in parts[1].split(',')])
            rotation = np.array([float(x) for x in parts[2].split(',')])
            extent = np.array([float(x) for x in parts[3].split(',')])
            
            rotation = rotation.reshape(3,3)
            if 'nan' not in line:
                center = T[:3,:3] @ center + T[:3,3]
                rotation = T[:3,:3] @ rotation
                instances_box[idx] = {'center':center,'rotation':rotation,'extent':extent}
        f.close()
    
    with open(out_file,'w') as f:
        f.write(header+'\n')
        for idx in instances_box.keys():
            center = instances_box[idx]['center']
            rotation = instances_box[idx]['rotation']
            extent = instances_box[idx]['extent']
            f.write('{};'.format(idx))
            f.write('{:.6f},{:.6f},{:.6f};'.format(center[0],center[1],center[2]))
            f.write('{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f};'.format(
                rotation[0,0],rotation[0,1],rotation[0,2],
                rotation[1,0],rotation[1,1],rotation[1,2],
                rotation[2,0],rotation[2,1],rotation[2,2]))
            f.write('{:.3f},{:.3f},{:.3f}\n'.format(extent[0],extent[1],extent[2]))
            
        f.close()

    return instances_box

def transform_scan_maps(dataroot, scene_name, T_ref_src, augment_drift=True, infolder='val_swap',outfolder='val'):
    '''
    Read the rescontruction in infolder. 
    Save the transformed reconstruction to the outfolder.
    '''
    print('Transform {} with augment drift {}'.format(scene_name,augment_drift))
    
    scene_dir = os.path.join(dataroot,infolder,scene_name)
    out_scene_dir = os.path.join(dataroot,outfolder,scene_name)
    if os.path.exists(out_scene_dir)==False:
        os.makedirs(out_scene_dir)
    
    T_drift = np.eye(4)
    if augment_drift:
        T_drift[:3,:3] = R.from_euler('z', np.random.uniform(-180,180,1), degrees=True).as_matrix()
        T_drift[:3,3] = np.random.uniform(-10.0,10.0,3)
    
    T_gt = transform_coordinate(T_ref_src,np.linalg.inv(T_drift))
    
    # Transform and write point cloud
    ply_files = glob.glob(scene_dir+'/*.ply')
    for plyfile in sorted(ply_files):
        pcd = o3d.io.read_point_cloud(plyfile)
        pcd.transform(T_drift)
        o3d.io.write_point_cloud(plyfile.replace(infolder,outfolder),pcd)
        center = pcd.get_center()
        # print('{}:({:.3f},{:.3f},{:.3f})'.format(os.path.basename(plyfile),center[0],center[1],center[2]))
    
    #
    transform_instance_boxes(T_drift,
                             os.path.join(scene_dir,'instance_box.txt'),
                             os.path.join(out_scene_dir,'instance_box.txt'))
    
    os.system('cp {} {}'.format(os.path.join(scene_dir,'instance_info.txt'),
                                os.path.join(out_scene_dir,'instance_info.txt')))
    
    np.savetxt(os.path.join(out_scene_dir,'transform.txt'),T_gt,fmt='%.6f')
    

if __name__ == '__main__':
    root_dir = '/data2/sgslam'
    split = 'scans'
    split_file = 'scans.txt'
    WEBRTC_IP = '143.89.46.75'
    
    #
    scans = read_scans(os.path.join(root_dir,'splits',split_file))
    scan_pairs = read_scan_pairs(os.path.join(root_dir,'splits','val.txt'))
    
    ref_scans = []
    for pair in scan_pairs:
        if pair[1] not in ref_scans:
            ref_scans.append(pair[1])

    # 
    if WEBRTC_IP is not None:
        os.environ.setdefault('WEBRTC_IP', WEBRTC_IP)
        os.environ.setdefault('WEBRTC_PORT', '8020')
        o3d.visualization.webrtc_server.enable_webrtc()


    # for scan in scans:
    #     print('---------- processing {} -------------'.format(scan))
    #     T_ref_src = np.loadtxt(os.path.join(root_dir,'scans',scan,'T_ref_src.txt'))
    #     if scan in ref_scans:
    #         augment_drift = False
    #     else:
    #         augment_drift = True
        
    #     transform_scan_maps(root_dir,scan,T_ref_src,augment_drift)
        
    #     break
        # move_dense_map(os.path.join(root_dir,'scans',scan),
        #                os.path.join(root_dir,'val',scan))
        # write_intrinsic(os.path.join(root_dir,split,scan))
        # break
    
    # exit(0)
    # Visualize the maps after augment with pose drift
    permute_indices = np.random.permutation(len(scan_pairs))
    scan_pairs = [scan_pairs[i] for i in permute_indices]
    for pair in scan_pairs:
        print('Visualize {} {}'.format(pair[0],pair[1]))
        src_pcd = o3d.io.read_point_cloud(os.path.join(root_dir,'val',pair[0],'instance_map.ply'))
        ref_pcd = o3d.io.read_point_cloud(os.path.join(root_dir,'val',pair[1],'instance_map.ply'))
        T_ref_src = np.loadtxt(os.path.join(root_dir,'val',pair[0],'transform.txt'))
        src_pcd.transform(T_ref_src)
        
        src_pcd.paint_uniform_color([0.0,1.0,0.7])
        ref_pcd.paint_uniform_color([0.7,1.0,0.0])
        
        out = [src_pcd,ref_pcd]
        # continue
        o3d.visualization.draw(out, non_blocking_and_return_uid=False)
