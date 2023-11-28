import os, glob
import plyfile
import numpy as np
import numpy.linalg as LA
from numpy.linalg import inv
import math
import cv2
from enum import Enum
import argparse
import multiprocessing as mp

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
    return scans

def read_frames(dir):
    with open(dir,'r') as f:
        frames = {}
        lines = f.readlines()
        num_points = int(lines[1].strip().split(':')[-1])
        num_points_uv = 0
        for line in lines[2:]:
            line = line.strip().split(',')
            frames[line[0]] = int(line[1])
            num_points_uv +=int(line[1])
        f.close()
        return frames, num_points, num_points_uv

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4). Cam intrinsic
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    z = points[2, :]

    # Normalize x,y coordinate and keep z coordinate.
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points[2, :] = z

    return points


def read_intrinsic(dir,align_depth=False,verbose=False):
    """
    Args:
        dir (_type_): folder directory
    Returns:
        rgb_dim: [rows,cols]
    """
    K_rgb = np.loadtxt(os.path.join(dir,'intrinsic_color.txt'))
    K_depth = np.loadtxt(os.path.join(dir,'intrinsic_depth.txt'))
    rgb_dim = np.zeros((2),np.int32)    
    depth_dim = np.zeros((2),np.int32)
    
    with open(os.path.join(dir,'sensor_shapes.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.find('color_height') != -1:
                rgb_dim[0] = int(line.split(':')[-1])
            elif line.find('color_width') != -1:
                rgb_dim[1] = int(line.split(':')[-1])
            elif line.find('depth_height') != -1:
                depth_dim[0] = int(line.split(':')[-1])
            elif line.find('depth_width') != -1:
                depth_dim[1] = int(line.split(':')[-1])
        f.close()
        
    if verbose:
        print('Read color intrinsic: \n',K_rgb)
        print('Read depth intrinsic: \n',K_depth)
        print('Read color shape: \n',rgb_dim)
        print('Read depth shape: \n',depth_dim)
    if align_depth:
        K_rgb = adjust_intrinsic(K_rgb,rgb_dim,depth_dim)
        rgb_dim = depth_dim
    
    return K_rgb,K_depth,rgb_dim,depth_dim

def adjust_intrinsic(intrinsic, raw_image_dim, resized_image_dim):
    '''Adjust camera intrinsics.'''
    import math

    if np.sum(resized_image_dim - raw_image_dim)==0:
        return intrinsic
    resize_width = int(math.floor(resized_image_dim[1] * float(
                    raw_image_dim[0]) / float(raw_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(raw_image_dim[0])
    intrinsic[1, 1] *= float(resized_image_dim[1]) / float(raw_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(resized_image_dim[0] - 1) / float(raw_image_dim[0] - 1)
    intrinsic[1, 2] *= float(resized_image_dim[1] - 1) / float(raw_image_dim[1] - 1)
    return intrinsic    

def align_depth_to_rgb(depth_img,depth_intrinsic,K_rgb,rgb_shape, max_depth=3.0):
    align_depth = np.zeros(rgb_shape,np.float32)-100.0
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze() # Nx3
    # print('mean d: {}'.format(uv_depth[:,2].mean()))
    # print(uv_depth.shape)

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    
    # Project on rgb image
    rgb_uv_depth = view_points(points[:,:3].T,K_rgb,normalize=True).T # Nx3
    # print('mean d: {}'.format(rgb_uv_depth[:,2].mean()))
    count = 0
    
    for i in range(n):
        if rgb_uv_depth[i,0] < 1 or rgb_uv_depth[i,0] > rgb_shape[1]-1 or rgb_uv_depth[i,1] < 0 or rgb_uv_depth[i,1] > rgb_shape[0]-1 or rgb_uv_depth[i,2] < 0.1 or rgb_uv_depth[i,2] > max_depth:
            continue
        align_depth[int(rgb_uv_depth[i,1]),int(rgb_uv_depth[i,0])] = rgb_uv_depth[i,2]
        count +=1
    # print('{}/{} valid in aligned depth'.format(np.count_nonzero(align_depth>0),rgb_shape[0]*rgb_shape[1]))
    
    return align_depth


def project(points, normals,T_wc, K,im_shape,max_depth =3.0,min_depth=1.0,margin=20):
    """
    Args:
        points (np.ndarray): Nx3
        normals (np.ndarray): Nx3
    Output:
        points_uv_all (np.ndarray): Nx3, invalid points are set to -100
        mask (np.ndarray): Nx1, True for valid points
        theta: Nx1, cos theta
    """
    points_uv_all = np.ones((points.shape[0],3),np.float32)-100.0 # Nx3
    normals_uv_all = np.ones((normals.shape[0],3),np.float32)-100.0 # Nx3
    cos_theta_all = np.zeros((points.shape[0]),np.float32) # Nx1
    T_cw = inv(T_wc) # 4x4
    
    # Transform into camera coordinates
    points_homo = np.concatenate([points,np.ones((points.shape[0],1))],axis=1)  # Nx4
    normals_homo = np.concatenate([normals, np.ones((normals.shape[0],1))], axis=1) # 
    points_cam = T_cw.dot(points_homo.T)[:3,:].T  # 3xN, p
    normals_cam_ = T_cw.dot(normals_homo.T)[:3,:].T # 3xN, n
    normals_cam = np.tile(1/ LA.norm(normals_cam_.T,axis=0),(3,1)).T * normals_cam_

    # Cosine(theta) = p \dot n /(|p||n|)   
    # cos_theta = np.sum(points_cam * normals_cam, axis=1)
    # pn_mag = LA.norm(points_cam, axis=1) * LA.norm(normals_cam,axis=1)     
    P_c = points-T_wc[:3,3]
    cos_theta = np.sum(P_c * normals, axis=1)
    pn_mag = LA.norm(P_c, axis=1) * LA.norm(normals, axis=1) 
    cos_theta = cos_theta / pn_mag # [-1,1]
    cos_theta = abs(cos_theta) # [0.0,1.0]
    # cos_theta = np.zeros((points.shape[0]),np.float32)+0.99
    
    # Project into image
    points_uv = view_points(points_cam.T,K,normalize=True).T    # [u,v,d],Nx3
    
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = (points_cam[:,2] > min_depth) &(points_cam[:,2]< max_depth) & (
            points_uv[:,0] > margin) & (points_uv[:,0] < im_shape[1] - margin) & (
            points_uv[:,1] > margin) & (points_uv[:,1] < im_shape[0] - margin)
    # normal_mask = normals_cam[:,2] < 0.0
    # mask = np.logical_and(mask,normal_mask)
    points_uv_all[mask] = points_uv[mask][:,:3]
    normals_uv_all[mask] = normals_cam[mask][:,:3]
    cos_theta_all[mask] = cos_theta[mask]
    # print('mean depth: ',np.mean(points_uv_all[mask][:,2]))

    # print('{}/{} points in camera view'.format(np.count_nonzero(mask),points.shape[0]))
    return mask, points_uv_all, normals_uv_all, cos_theta

def filter_occlusion(points_uv,depth_map,max_dist_gap=0.05, kernel_size=8,
                    min_mask_pixels=0.5, max_variance = 0.2):
    """
    Args:
        points_uv: [N,3]
        depth_map: [H,W,D], float
    Output:
        filter_mask: [N], True for non-occluded points
    """
    # max_variance = 0.01
    
    M = points_uv.shape[0]
    # patch_pixels = (kernel_size*2+1)**2
    # min_mask_pixels = int(patch_pixels*mask_ratio)
    patch_side = kernel_size*2+1
    filter_mask = np.zeros(M,dtype=np.bool_)
    # uv = points_uv[:,:2].astype(np.int32)
    # d_ = depth_map[uv[:,1],uv[:,0]]
    # depth_diff = np.abs(points_uv[:,2] - d_)
    # filter_mask = np.logical_and(depth_diff<max_dist_gap, d_>0.1)
    
    # filter_mask = np.ones(M,dtype=np.bool_)
    
    count_valid = 0
    for i in range(M):
        cx = int(points_uv[i,0])
        cy = int(points_uv[i,1])
        d = points_uv[i,2]
        if d<0.2:continue

        patch = depth_map[cy-kernel_size:cy+kernel_size+1,cx-kernel_size:cx+kernel_size+1]
        mask = patch>0.0
        if np.count_nonzero(mask)/(patch_side*patch_side) <min_mask_pixels:continue
        var = np.var(patch[mask])
        mu = np.mean(patch[mask])
        # mu = depth_map[cy,cx]
        if abs(d-mu)<max_dist_gap and var<max_variance:
            filter_mask[i] = True
        count_valid +=1
                    
    # print('{}/{} non-occluded points'.format(np.count_nonzero(filter_mask),np.count_nonzero(points_uv)))
    return filter_mask


def filter_valid_scenes(arguments,valid_scenes):
    rootdir,scene_name,sample_number,enable_visualize = arguments
    filter_scenes = []
    for scene_name in valid_scenes:
        print(scene_name)
        scene_dir = os.path.join(rootdir,scene_name)
        intrinsic_dir = os.path.join(scene_dir,'intrinsic')
        K_rgb, K_depth,rgb_dim,depth_dim = read_intrinsic(intrinsic_dir,True)
        if rgb_dim[0] == 968 and rgb_dim[1] == 1296:
            filter_scenes.append(scene_name)
            
    print('{}/{} scenes are valid'.format(len(filter_scenes),len(valid_scenes)))
    return filter_scenes

def process_scene(arguments):
    '''
    All the input files and output files are in the scene directory.
    Input files:
        ply: scene_name_vh_clean_2.ply.
        intrinsic: rgb-d intrinsic.
        pose: randomly sample K frames from the sequence.
    Output files:
        preprocess folder: points_uv.npy, frames.txt
        points_uv.npy: (NN,5), [frame_id,pt_idx,u,v,d]. NN is the total number of projected points in all the frames.
        frames.txt: (K,2), [frame_id, projected_points_number]
    '''
    scene_dir,dataset,sample_number,enable_visualize = arguments
    # scene_dir = os.path.join(rootdir,scene_name)
    # Filter parameters
    max_dist_gap = 0.05
    depth_kernal_size = 8
    kernal_valid_ratio = 0.5
    kernal_max_var = 0.05
    min_theta = 0.2
    # Image frame paramters
    max_frames_number = 100
    min_points = 1000
    min_increment = 400
    min_frames_gap = 10
    scene_name = os.path.basename(scene_dir)

    class DepthSource(Enum):
        raw = 0
        aligned = 1 # reprojected to rgb
        rendered = 2 # render from mesh
    
    if dataset=='ScanNet':
        plydir = os.path.join(scene_dir,scene_name+'_vh_clean_2.ply')
        rgb_posix = '.jpg'
        depth_posix = '.png'
        render_depth_posix = None
        depth_source = DepthSource.raw
        output_dir = os.path.join(scene_dir,'preprocessed')
    elif dataset=='scanneto3d':
        plydir = os.path.join(scene_dir,'o3d_vx_0.05.ply')
        rgb_posix = '.jpg'
        depth_posix = '.png'
        render_depth_posix = None
        depth_source = DepthSource.raw
        output_dir = os.path.join(scene_dir,'o3d_preprocessed')
    elif dataset=='scannet005':
        plydir = os.path.join(scene_dir,'vh_clean_2_vx_0.05.ply')
        rgb_posix = '.jpg'
        depth_posix = '.png'
        render_depth_posix = None
        depth_source = DepthSource.raw
        output_dir = os.path.join(scene_dir,'preprocessed_005')
    elif dataset=='3rscan':
        plydir = os.path.join(scene_dir,'refined.ply')
        rgb_posix = '.color.jpg'
        depth_posix = '.depth.pgm'
        render_depth_posix = '.rendered.depth.png'
        depth_source=DepthSource.rendered
        output_dir = os.path.join(scene_dir,'o3d_preprocessed')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
        
    intrinsic_dir = os.path.join(scene_dir,'intrinsic')
    if(os.path.exists(output_dir)==False): os.mkdir(output_dir)
    
    print('-------- Processing {} ---------'.format(scene_name))
    
    f = plyfile.PlyData().read(plydir)
    point_states = np.array([list(x) for x in f.elements[0]])
    points = np.ascontiguousarray(point_states[:, :3],dtype=np.float32)
    colors = np.ascontiguousarray(point_states[:, 3:6],dtype=np.float32) #[0,255]
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals,dtype=np.float32)
    
    assert colors.max()<=255 and colors.min()>=0, 'color range is not [0,255]'
    assert normals.shape[0] == points.shape[0], 'normals and points have different number of points'
    print('read {} points from {}'.format(points.shape[0],plydir))
    
    ## 
    K_rgb, K_depth,rgb_dim,depth_dim = read_intrinsic(intrinsic_dir,False)
    if depth_source==DepthSource.raw:
        rgb_out_dim = depth_dim
        K_rgb_out = adjust_intrinsic(K_rgb,rgb_dim,rgb_out_dim)
    elif depth_source==DepthSource.aligned or depth_source==DepthSource.rendered:
        rgb_out_dim = rgb_dim
        K_rgb_out = K_rgb
    print(K_rgb_out)
    # print(rgb_out_dim)
    
    pose_files =  glob.glob(os.path.join(scene_dir,'pose','*.txt'))  
    assert len(pose_files)>=sample_number, '{} extract only {} frames'.format(scene_name,len(pose_files))
    # select_poses = np.random.choice(pose_files,20,replace=False)  
    select_poses = sorted(pose_files)
    unique_points_counter = {}
    # unique_points_counter[0] = 0
    summary_points_uv = None
    summary_indices = None
    frames = {} 
    frame_id = 0 # consistent with the frame names
    prev_frame_name = 'frame-000000'
    N = points.shape[0]
    
    if enable_visualize:
        colorbar = np.invert(np.arange(255).astype(np.uint8))
        colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET).squeeze() #255x3
    
    for i,pose_file in enumerate(select_poses):
        if len(frames)>max_frames_number: break
        frame_name = pose_file.split('/')[-1].split('.')[0]
        frame_gap  = int(frame_name.split('-')[-1]) - int(prev_frame_name.split('-')[-1])
        # print(frame_name)
        if frame_gap< min_frames_gap: continue
        rgb_dir = os.path.join(scene_dir,'color',frame_name+rgb_posix)
        
        print('->{} '.format(frame_name))
        pose = np.loadtxt(pose_file)

        rgbimg = cv2.imread(rgb_dir)
        if rgbimg.shape[0]!=rgb_out_dim[0] or rgbimg.shape[1]!=rgb_out_dim[1]:
            rgbimg = cv2.resize(rgbimg,(rgb_out_dim[1],rgb_out_dim[0]),interpolation=cv2.INTER_NEAREST)

        if depth_source==DepthSource.raw:
            depth_dir = os.path.join(scene_dir,'depth',frame_name+depth_posix)
            raw_depth = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
            depth = raw_depth.astype(np.float32)/1000.0
            assert raw_depth.shape[0]==depth_dim[0] and raw_depth.shape[1]==depth_dim[1], 'depth image dimension does not match'
        elif depth_source==DepthSource.aligned:
            depth = align_depth_to_rgb(raw_depth,K_depth,K_rgb,rgb_dim, 5.0)
        elif depth_source==DepthSource.rendered:
            depth = cv2.imread(os.path.join(scene_dir,'render',frame_name+render_depth_posix),-1)
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth = depth.astype(np.float32)/1000.0
        else: raise NotImplementedError    
        assert depth.shape[0] == rgbimg.shape[0]
        assert depth.shape[1] == rgbimg.shape[1]
        
        ## project
        mask, points_uv, normals_cam, theta = project(points, normals,pose, K_rgb_out, rgb_out_dim, 5.0, 0.5) # Nx3
        prev_frame_name = frame_name
        count_view_points = np.sum(mask)  
        if count_view_points < min_points:
            print('drop poor frame with {} valid points'.format(count_view_points))
            continue
        filter_mask = filter_occlusion(points_uv,depth,max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        theta_mask  = theta > min_theta
        
        #[frame_id,pt_idx,u,v,d,cos_theta]
        points_uv = np.concatenate((np.ones((N,1))*frame_id,np.arange(N).reshape(N,1),points_uv,theta.reshape(N,1)),axis=1) 
        mask = np.logical_and(mask,filter_mask)
        mask = np.logical_and(mask,theta_mask)
        
        indices = np.where(mask)[0]
        
        prev_points_count = np.unique(summary_indices).shape[0] if summary_indices is not None else 0
        if summary_indices is not None:
            count_increment = np.setdiff1d(indices,np.unique(summary_indices)).shape[0]
            if count_increment<min_increment:
                print('drop frame with low increment')
                continue
            
        print('{}/{} valid/viewed points'.format(np.count_nonzero(mask),count_view_points))

        ## Update the output data
        assert np.count_nonzero(mask) < points.shape[0], 'Filter does not work correct and all points are recorded.'
        summary_points_uv = points_uv[mask] if summary_points_uv is None else np.concatenate((summary_points_uv,points_uv[mask]),axis=0)
        summary_indices = indices if summary_indices is None else np.concatenate((summary_indices,indices),axis=0)
        frames[frame_name] = indices.shape[0]  #np.count_nonzero(mask)
        frame_id += 1
        unique_points_counter[i] = np.unique(summary_indices).shape[0]
        assert unique_points_counter[i]-prev_points_count>=min_increment, 'Fill rate does not increase {}:{}'.format(scene_name,frame_name)

        ######## Visualization #########
        if enable_visualize==False or np.count_nonzero(mask)<1: continue
        debug_img = rgbimg.copy()
        prj_rgb = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
        normal_map = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
        normal_map_cam = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
        
        valid_corlor = colors[mask].astype(np.float64)
        valid_point = points_uv[mask]
        valid_normal_cam = normals_cam[mask]
        valid_normal = normals[mask]
        valid_theta = theta[mask]
        vizdepth = ((depth / 5.0) *255.0).astype(np.uint8)
        vizdepth = cv2.applyColorMap(vizdepth, cv2.COLORMAP_JET)
        M = valid_point.shape[0]  
        
        # print('mean normal: {:.4f},{:.4f}'.format(valid_normal[:,2].min(),valid_normal[:,2].max()))
        # print('mean normal cam: {:.4f},{:.4f} '.format(valid_normal_cam[:,2].min(),valid_normal_cam[:,2].max()))
        # print('cos(theta): {:4f},{:.4f}'.format(valid_theta.min(),valid_theta.max()))
        
        for j in range(M):
            x = int(valid_point[j,2])
            y = int(valid_point[j,3])
            d = valid_point[j,2]
            nw = valid_normal[j]
            nc = valid_normal_cam[j]
            assert LA.norm(nw)>0.99 and LA.norm(nc)>0.99, 'normal is not unit vector'

            nw = (255.0*(nw)).astype(np.float64)    #[0,255]
            nc = (255.0*(nc)).astype(np.float64)
            
            pt_color = valid_corlor[j]
            viz_theta = np.float64(colorbar[math.floor(255*valid_theta[j])])

            cv2.circle(debug_img,(x,y),2,[pt_color[2],pt_color[1],pt_color[0]],-1)
            # cv2.circle(debug_img,(x,y),5,[int(marker_color[0]),int(marker_color[1]),int(marker_color[2])],-1)
            
            cv2.circle(prj_rgb,(x,y),5,[pt_color[2],pt_color[1],pt_color[0]],-1)
            cv2.circle(normal_map,(x,y),5,[nw[2],nw[1],nw[0]],-1)
            # cv2.circle(normal_map_cam,(x,y),5,[nc[2],nc[1],nc[0]],-1)
            cv2.circle(normal_map_cam,(x,y),5,[viz_theta[2],viz_theta[1],viz_theta[0]],-1)

            # debug_img = cv2.addWeighted(debug_img,0.9,viz_depth,0.9,0)
        
        cv2.putText(debug_img,'rgb+prj_pts',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
        cv2.putText(prj_rgb,'pts_rgb',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
        cv2.putText(normal_map,'pts_normal',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
        cv2.putText(normal_map_cam,'pts_conf',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
        # debug_img = cv2.rotate(debug_img, cv2.ROTATE_90_CLOCKWISE)
        # prj_rgb = cv2.rotate(prj_rgb, cv2.ROTATE_90_CLOCKWISE)
        debug_img = np.concatenate([debug_img,prj_rgb],axis=1)
        vizdepth = np.concatenate([normal_map,normal_map_cam],axis=1)
        debug_img = np.concatenate([debug_img,vizdepth],axis=0)
        cv2.imwrite(os.path.join(output_dir,frame_name+'.png'),debug_img)
        # break

    print('{} frames are processed for scene {}'.format(len(select_poses),scene_name))
    if summary_points_uv is None:
        print('No valid points are recorded')
        return
    # exit(0)
    
    ## Export data
    record_frame_numbers = np.max(summary_points_uv[:,0])
    assert record_frame_numbers == len(frames)-1, 'Frame number does not match'
    with open(os.path.join(output_dir,'frames.txt'), 'w') as fp:
        fp.write('# plydir: {}, min_frame_gap: {}, min_increment_point: {}, kernal_size:{}, max_distance: {}, kernal_valid:{}, kernal_variance:{}, min_theta: {} \n'.format(
            plydir.split('/')[-1],min_frames_gap,min_increment, depth_kernal_size, max_dist_gap, kernal_valid_ratio, kernal_max_var,min_theta))
        fp.write('points number: {}'.format(points.shape[0]))
        for frame_name, offset in frames.items():
            fp.write('\n')
            fp.write('{},{}'.format(frame_name,offset))
        fp.close()

    # (M,5), [frame_idx,pt_idx,u,v,d]``
    # print('all points number:{}'.format(summary_points_uv.shape[0]))
    np.save(os.path.join(output_dir,'points_uv.npy'),summary_points_uv)
    
    with open(os.path.join(output_dir,'log.txt'),'w') as log_f:
        num_matched = np.unique(summary_indices).shape[0]
        log_f.write('{}/{} or {:.1f}% points are associated \n'.format(num_matched,points.shape[0],100*num_matched/points.shape[0]))
        for k,v in unique_points_counter.items():
            log_f.write('{}:{:.1f}%, {}\n'.format(k,100*v/N,v))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, required=True, help='root directory of ScanNet')
    parser.add_argument('--dataset', type=str, default='ScanNet', help='ScanNet/scanneto3d/3rscan')
    parser.add_argument('--split', type=str, default='train', help='train/val/test')
    parser.add_argument('--sample_number', type=int, default=10, help='minimum number of frames to sample')
    parser.add_argument('--scene_id', type=str, required=False, help='scene id')
    parser.add_argument('--num_process', type=int, required=False, help='number of process')
    parser.add_argument('--visualize', help='enable visualization', action='store_true')

    opt = parser.parse_args()
    rootdir = opt.rootdir
    sample_number = opt.sample_number 
    enable_visualize = opt.visualize 
    if opt.dataset == 'ScanNet' or opt.dataset =='scanneto3d' or opt.dataset =='scannet005':
        dataroot = os.path.join(rootdir,opt.split)
    elif opt.dataset == '3rscan':
        dataroot = rootdir
    else:
        raise ValueError('Unknown dataset')
    print('Processing {}/{}'.format(opt.dataset,opt.split))
    
    split_filename = opt.split
    
    if opt.scene_id is not None: # a single scan
        valid_scenes = [opt.scene_id]
    else: # read scans from list
        valid_scenes = []
        scans = read_scans(os.path.join(rootdir,'splits',split_filename+'.txt'))
        for scene in scans:
            pose_files = glob.glob(os.path.join(dataroot,scene,'pose','*.txt'))
            if len(pose_files)>=20:
                valid_scenes.append(scene)
    
    num_process = opt.num_process if opt.num_process is not None else os.cpu_count()
    print('{} scenes are processing'.format(len(valid_scenes)))
    
    if num_process ==1:
        for scene_name in valid_scenes:
            process_scene((os.path.join(dataroot,scene_name),opt.dataset,sample_number,enable_visualize))
            # break
    else:
        p = mp.Pool(processes=num_process)
        p.map(process_scene,[(os.path.join(dataroot,scene_name),opt.dataset,sample_number,enable_visualize)for scene_name in valid_scenes])
        p.close()
        p.join()
            
        print('{} scenes are processed'.format(len(valid_scenes)))
    
    
