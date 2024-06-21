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
    
def read_da_file(dir):
    with open(dir, 'r') as f:
        lines = f.readlines()
        da = []
        for line in lines:
            eles = line.strip().split(' ')
            depth = os.path.basename(eles[0]).split('.')[0]
            da.append(depth.strip())
        f.close()
        return da

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

def transform_reconstructed_maps(dataroot, scene_name, T_ref_src, augment_drift=True, infolder='val_swap',outfolder='val'):
    '''
    Read the rescontruction in scans folder.
    Incorporate a random drift to the reconstruction. 
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

def transform_global_pcd(dataroot, scene_name, transform=True):
    '''
    post-process.
    A val folder is generated earlier, including the ground-truth transformation.
    Read the global pcd and align it with generated instance map in `val` folder.
    '''
    print('----------- Transform {} ----------'.format(scene_name))
    
    scans_scene = os.path.join(dataroot,'scans',scene_name)
    val_scene = os.path.join(dataroot,'val',scene_name)
    
    # load
    global_pcd = o3d.io.read_point_cloud(os.path.join(scans_scene,'mesh_o3d.ply'))
    
    if transform:
        T_ref_src = np.loadtxt(os.path.join(scans_scene,'T_ref_src.txt'))
        T_gt = np.loadtxt(os.path.join(val_scene,'transform.txt'))
        T_drift = transform_coordinate(np.linalg.inv(T_ref_src),T_gt)
        T_drift = np.linalg.inv(T_drift)
        global_pcd.transform(T_drift)
    
    # export
    o3d.io.write_point_cloud(os.path.join(val_scene,'global_pcd.ply'),global_pcd)

def transform_sequence(folder, T_ref_src):
    '''
    Incoporate random pose drift to the sequence.
    '''
    import shutil
    
    # back up pose folder
    if os.path.exists(os.path.join(folder,'pose_bk')):
        print('Skip scene ', folder)
        return None
    else:
        shutil.copytree(os.path.join(folder,'pose'),os.path.join(folder,'pose_bk'))    
    if os.path.exists(os.path.join(folder,'trajectory.log')):
        shutil.copyfile(os.path.join(folder,'trajectory.log'),os.path.join(folder,'trajectory_bk.log')) 

    # generate random pose drift
    T_drift = np.eye(4)
    T_drift[:3,:3] = R.from_euler('z', np.random.uniform(-180,180,1), degrees=True).as_matrix()
    T_drift[:3,3] = np.random.uniform(-10.0,10.0,3)
    
    # update pose
    pose_frames = glob.glob(os.path.join(folder,'pose','*.txt'))
    pose_frames = sorted(pose_frames)
    frame_pose_map = {}
    for i in range(len(pose_frames)):
        frame_name = os.path.basename(pose_frames[i]).split('.')[0]
        T_l_cam = np.loadtxt(pose_frames[i])
        T_lnew_cam = T_drift @ T_l_cam
        np.savetxt(pose_frames[i],T_lnew_cam,fmt='%.6f')
        frame_pose_map[frame_name] = T_lnew_cam
    
    # np.savetxt(os.path.join(folder,'T_drift.txt'),T_drift,fmt='%.6f')
    
    # update trajectory.log
    da_file = None
    if os.path.exists(os.path.join(folder,'data_association_bk.txt')):
        da_file = os.path.join(folder,'data_association_bk.txt')
    else:
        da_file = os.path.join(folder,'data_association.txt')
    
    depth_list = read_da_file(os.path.join(folder,da_file))
    i = 0

    with open(os.path.join(folder,'trajectory.log'),'w') as f:
        for frame_name in depth_list:
            if frame_name not in frame_pose_map.keys():
                print('[WARNNING] skip frame {} in trajectory.log'.format(frame_name))
                continue
            
            f.write('{} {} {}\n'.format(i,i,i+1))
            # write pose 4x4
            T = frame_pose_map[frame_name]
            for j in range(4):
                for k in range(4):
                    f.write('{:.6f} '.format(T[j,k]))
                f.write('\n')
            i += 1
        f.close()
        print('write trajectory.log')

    # update gt pose
    if T_ref_src is not None:
        T_gt = transform_coordinate(T_ref_src,np.linalg.inv(T_drift))
        return T_gt
    else: return None

def generate_vins_tag_poses(scene_folder):
    '''
        Compute T_vins_tag pose and save it to apriltag folder.
        Vins is the local coordinate the sequence is initialized.
    '''
    rgb_frames = glob.glob(scene_folder+'/apriltag/*.jpg')
    if len(rgb_frames)<1:
        print('No tag frames found in ', scene_folder)
        return None
    for rgb_frame in rgb_frames:
        frame_name = os.path.basename(rgb_frame).split('.')[0]
        T_c_tag = np.loadtxt(os.path.join(scene_folder, 'apriltag', frame_name + '_pose.txt'))
        T_c_tag[3,:3] = 0
        T_w_c = np.loadtxt(os.path.join(scene_folder, 'pose', frame_name + '.txt'))
        print(frame_name)

    T_w_tag = T_w_c @ T_c_tag
    np.savetxt(os.path.join(scene_folder, 'apriltag', 'T_vins_tag.txt'), T_w_tag, fmt='%.6f')
    print('write tag file to ', os.path.join(scene_folder, 'apriltag', '{}.txt'.format('T_vins_tag')))

if __name__ == '__main__':
    root_dir = '/data2/sgslam'
    split = 'scans'
    split_file = 'scans.txt'
    WEBRTC_IP = '143.89.46.75'
    
    #
    scans = read_scans(os.path.join(root_dir,'splits',split_file))
    scan_pairs = read_scan_pairs(os.path.join(root_dir,'splits','val_bk.txt'))
    gt_folder = os.path.join(root_dir,'gt')
    
    ref_scans = []
    for pair in scan_pairs:
        if pair[1] not in ref_scans:
            ref_scans.append(pair[1])

    # 
    if WEBRTC_IP is not None:
        os.environ.setdefault('WEBRTC_IP', WEBRTC_IP)
        os.environ.setdefault('WEBRTC_PORT', '8020')
        o3d.visualization.webrtc_server.enable_webrtc()

    for pair in scan_pairs:
        src_scene = pair[0]
        print('-------processing source scene: {}--------'.format(pair[0]))
        T_ref_src = np.loadtxt(os.path.join(root_dir,'scans',src_scene,'T_ref_src.txt'))
        generate_vins_tag_poses(os.path.join(root_dir,'scans',pair[0]))
        generate_vins_tag_poses(os.path.join(root_dir,'scans',pair[1]))
        
        continue
        T_ref_src_new = transform_sequence(os.path.join(root_dir,'scans',src_scene),T_ref_src)
        if T_ref_src_new is not None:
            np.savetxt(os.path.join(gt_folder,'{}-{}.txt'.format(pair[0],pair[1])),T_ref_src_new,fmt='%.6f')
        
        # break

    exit(0)
    for scan in scans:
        print('---------- processing {} -------------'.format(scan))

        T_ref_src = np.loadtxt(os.path.join(root_dir,'scans',scan,'T_ref_src.txt'))
        if scan in ref_scans:
            augment_drift = False
        else:
            augment_drift = True
        
        transform_reconstructed_maps(root_dir,scan,T_ref_src,augment_drift)
        
        break
        move_dense_map(os.path.join(root_dir,'scans',scan),
                       os.path.join(root_dir,'val',scan))
        write_intrinsic(os.path.join(root_dir,split,scan))
        break
    
    exit(0)
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
