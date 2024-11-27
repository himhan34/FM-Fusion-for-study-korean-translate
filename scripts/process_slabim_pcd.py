import os, sys, glob
import cv2
import open3d as o3d
import numpy as np
import multiprocessing as mp

def read_sensor_shapes(dir):
    assert os.path.exists(dir), 'File not found: {}'.format(dir)
    with open(dir, 'r') as f:
        lines = f.readlines()
        height, width = 0, 0
        for line in lines:
            if 'color_width' in line:
                width = int(line.split(':')[1].strip())
            elif 'color_height' in line:
                height = int(line.split(':')[1].strip()) 
        return height, width

def read_intrinsics(intrinsic_folder):
    
    h,w = read_sensor_shapes(os.path.join(intrinsic_folder, 
                                          'sensor_shapes.txt'))
    K = np.loadtxt(os.path.join(intrinsic_folder, 'intrinsic_depth.txt'))

    return K, h, w

def project_xyz(xyz, K, h, w, depth_scale):
    z = xyz[2]
    u, v = np.dot(K, xyz)[:2] / z
    u, v = int(u), int(v)
    if u>=0 and u<w and v>=0 and v<h:
        # out_array = np.array([u, v, z * depth_scale], dtype=np.float32)
        # out_mat[v, u] = z * depth_scale
        return u, v, z * depth_scale
    else:
        return -1, -1, -1

def project_pcd2image(points:np.ndarray, 
                      K:np.ndarray, 
                      height:int, width:int,
                      DEPTH_SCALE:float=1000):
    depth_map = np.zeros((height, width), dtype=np.float32)
    N = points.shape[0]
    uvd_mat = np.zeros((N, 3), dtype=np.float32)
    
    for i in range(N):
        u,v,d = project_xyz(points[i], K, height, width, DEPTH_SCALE)
        uvd_mat[i] = np.array([u,v,d])
    
    for i in range(N):
        u,v,d = uvd_mat[i]
        if d>0: depth_map[int(v), int(u)] = d
        
    return depth_map

def project_pcd2image_cuda(points:np.ndarray,
                           K:np.ndarray,
                           height:int, width:int,
                           DEPTH_SCALE:float=1000):
    import torch
    depth_map = torch.zeros((height, width)).cuda()
    N = points.shape[0]
    
    points = torch.from_numpy(points).float().cuda() # (N,3)
    height = torch.tensor(height).cuda()
    width = torch.tensor(width).cuda()
    DEPTH_SCALE = torch.tensor(DEPTH_SCALE).cuda()
    K = torch.from_numpy(K).float().cuda() # (3,3)
    
    # extend K to (N,3,3)
    K = K.unsqueeze(0).expand(N, -1, -1)
    z = points[:,2] # (N,)
    points = points.unsqueeze(-1) # (N,3,1)

    uv = torch.matmul(K, points) # (N,3,3) x (N,3,1) -> (N,3,1)    
    uv = uv.squeeze() / z.unsqueeze(-1) # (N,3)
    u, v = uv[:, 0], uv[:, 1]
    u, v = u.int(), v.int()
    
    mask = (u>=0) & (u<width) & (v>=0) & (v<height) &(z>0)
    u, v, z = u[mask], v[mask], z[mask]
    depth_map[v, u] = z * DEPTH_SCALE
    
    depth_map = depth_map.cpu().numpy()
    return depth_map

if __name__=='__main__':
    ############ SET CONFIG ############
    DATA_ROOT = '/data2/slabim/scans'
    DEPTH_SCALE = 1000
    DEPTH_POSFIX = '.png'
    PROJECT_DEPTH = -1 # 0: run on CPU, 1: run on GPU, -1: not run
    MERGE_GLOBAL_MAP=True
    SAVE_VIZ = False
    
    ###################################
    scans = [
            # 'ab0202_00a',
            # 'ab0205_00a',
            # 'ab0206_00a',
            # 'ab0304_00a',
            'jk0001_00a',
            # 'jk0102_00a'
            ]
    
    if False: # debug
        # tmpdir = '/data2/sgslam/scans/uc0204_00a/depth/frame-000338.png'
        tmpdir = '/data2/slabim/scans/ab0202_00a/depth/frame-000000.png'
        depth = cv2.imread(tmpdir, cv2.IMREAD_UNCHANGED)
        print(depth.dtype, depth.shape)
        exit(0)
    
    for scan in scans:
        print('------------ Processing {} ------------'.format(scan))
        
        pcd_files = glob.glob(os.path.join(DATA_ROOT, scan, 'pcd', '*.pcd'))
        K, h, w = read_intrinsics(os.path.join(DATA_ROOT, scan, 'intrinsic'))
        os.makedirs(os.path.join(DATA_ROOT, scan, 'depth'), exist_ok=True)
        global_map = o3d.geometry.PointCloud()
        
        for pcd_file in sorted(pcd_files):
            frame_name = os.path.basename(pcd_file).split('.')[0]
            pcd = o3d.io.read_point_cloud(pcd_file) # in world coordinate
            print('Load {} points at {}'.format(len(pcd.points), os.path.basename(pcd_file)))
            
            if MERGE_GLOBAL_MAP:
                global_map += pcd
            
            if PROJECT_DEPTH<0: continue
            # Run depth projection
            T_wc = np.loadtxt(os.path.join(DATA_ROOT, scan, 'pose', frame_name+'.txt'))
            pcd.transform(np.linalg.inv(T_wc)) # in camera coordinate
            if PROJECT_DEPTH==1:
                depth = project_pcd2image_cuda(np.asarray(pcd.points), K, h, w)
            elif PROJECT_DEPTH==0:
                depth = project_pcd2image(np.asarray(pcd.points), K, h, w)
            
            depth = depth.astype(np.uint16)            
            cv2.imwrite(os.path.join(DATA_ROOT, scan, 'depth', frame_name+DEPTH_POSFIX), 
                        depth)
            print('Write {} depth image'.format(frame_name+DEPTH_POSFIX))
            
            if SAVE_VIZ:
                depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(DATA_ROOT, scan, 'viz', frame_name+'_depth'+DEPTH_POSFIX), 
                            depth_viz)
            
        if global_map.has_points():
            global_map = global_map.voxel_down_sample(voxel_size=0.05)
            o3d.io.write_point_cloud(os.path.join(DATA_ROOT, scan, 'mesh_o3d.ply'), 
                                     global_map)
            print('Write global map')
            
