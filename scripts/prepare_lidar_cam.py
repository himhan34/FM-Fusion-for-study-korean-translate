import os, glob, sys
import numpy as np 
import open3d as o3d
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from prepare_datasets import read_scans
from python.project_util import project


r3live_livox_params ={
    'img_width': 1280,
    'img_height': 1024,
    'fx': 863.4241,
    'fy': 863.4171,
    'cx': 640.6808,
    'cy': 518.3392
}

def convert_projection_matrix(params:dict):
    K = np.eye(4)
    K[0,0] = params['fx']
    K[1,1] = params['fy']
    K[0,2] = params['cx']
    K[1,2] = params['cy']
    
    img_shape = np.array([params['img_height'],params['img_width']])
    # img_shape = np.array([params['img_width'],params['img_height']])
    
    return K, img_shape

def write_intrinsic(scan_dir,K,img_shape):
    intrinsic_folder = os.path.join(scan_dir,'intrinsic')
    if os.path.exists(intrinsic_folder)==False:
        os.mkdir(intrinsic_folder)
        
    with open(os.path.join(intrinsic_folder,'sensor_shapes.txt'),'w') as f:
        f.write('color_width:{}\n'.format(img_shape[1]))
        f.write('color_height:{}\n'.format(img_shape[0]))
        f.write('depth_width:{}\n'.format(img_shape[1]))
        f.write('depth_height:{}\n'.format(img_shape[0]))
        f.close()
        
    np.savetxt(os.path.join(intrinsic_folder,'intrinsic_depth.txt'),K,fmt='%.6f')
    print('intrinsic saved')

    

def enrich_frame_pcd(global_pcd,frame_pcd,radius = 0.2):
    global_kd_tree = o3d.geometry.KDTreeFlann(global_pcd)
    enrich_points = o3d.geometry.PointCloud()
    N = len(frame_pcd.points)
    
    for i in range(N):
        [k, idx, _] = global_kd_tree.search_radius_vector_3d(frame_pcd.points[i], radius)
        if len(idx)>0:
            enrich_points = enrich_points + global_pcd.select_by_index(idx)
            
    print('Enriched {} points'.format(len(enrich_points.points)))
    out = frame_pcd + enrich_points
    
    return out
    
    

def render_point_to_depth(scandir,K,img_shape,visualize=None,depth_scale=1000.0):
    lidar_frames = glob.glob(scandir+'/point/*.pcd')
    
    global_pcd = o3d.io.read_point_cloud(os.path.join(scandir,'r3live_output','rgb_pt.pcd'))
    print('global point cloud has {} points'.format(len(global_pcd.points)))
    if visualize is not None:
        visualize_dir = os.path.join(scandir,'viz')
        if os.path.exists(visualize_dir)==False: os.makedirs(visualize_dir)
    FRAME_GAP = 1
    CHECK_DEPTH = False
    
    for i,lidar_frame in enumerate(sorted(lidar_frames)):
        if i%FRAME_GAP!=0: continue
        frame_pcd_w = o3d.io.read_point_cloud(lidar_frame)
        # frame_pcd_w = enrich_frame_pcd(global_pcd,frame_pcd_w)
        
        frame_name = os.path.basename(lidar_frame).split('.')[0]
        
        print('processing {}'.format(frame_name))
        camera_pose = np.loadtxt(os.path.join(scandir,'pose',frame_name+'.txt'))
        points_w = np.asarray(frame_pcd_w.points)
        # points_w = np.asarray(global_pcd.points)

        mask,points_uv,normals_uv,_ =project(points_w, np.zeros_like(points_w), camera_pose, K, img_shape, max_depth=50.0,min_depth=1.0,margin=10)
        u,v = points_uv[:,0].astype(np.int32), points_uv[:,1].astype(np.int32)
        d = points_uv[:,2]
        m = points_uv.shape[0]
        # print('max u:{} max v:{} max d:{}'.format(np.max(u),np.max(v),np.max(d)))
        # print('{} projected points'.format(m))

        render_depth = np.zeros(img_shape,dtype=np.uint16)
        render_depth[v,u] = (depth_scale * points_uv[:,2]).astype(np.uint16)
        cv2.imwrite(os.path.join(scandir,'depth',frame_name+'.png'),render_depth)
        
        if visualize:
            viz_depth = np.zeros(img_shape,dtype=np.uint8)
            # for u_,v_,d_ in zip(u,v,d):
            #     color_ = int(d_ * 10.0)
            #     cv2.circle(viz_depth,(u_,v_),3,(color_,color_,color_),-1)
            viz_depth[v,u] = (d * 10.0).astype(np.uint8)
            rgb = cv2.imread(os.path.join(scandir,'rgb',frame_name+'.png'))
            viz_depth = cv2.applyColorMap(viz_depth,cv2.COLORMAP_JET)
            
            # out = np.concatenate([rgb,viz_depth],axis=1)
            out = cv2.addWeighted(rgb,0.5,viz_depth,0.5,0)
            
            cv2.imwrite(os.path.join(visualize_dir,frame_name+'.png'),out)

        if CHECK_DEPTH:
            tmp_depth = cv2.imread(os.path.join(scandir,'depth',frame_name+'.png'),cv2.IMREAD_UNCHANGED)
            tmp_depth = tmp_depth.astype(np.float32) / depth_scale
            print('reload max depth:{}'.format(np.max(tmp_depth)))

        # print(camera_pose)
        # break
    


if __name__=='__main__':
    root_dir = '/data2/r3live'
    split = 'scans'
    split_file = 'active.txt'    
    VISUALIZE = True

    scans = read_scans(os.path.join(root_dir,'splits',split_file))
    K, img_shape = convert_projection_matrix(r3live_livox_params)

    for scan in scans:
        print('processing {}'.format(scan))
        write_intrinsic(os.path.join(root_dir,split,scan),K,img_shape)
        # render_point_to_depth(os.path.join(root_dir,split,scan),K, img_shape, VISUALIZE)
        
        break

