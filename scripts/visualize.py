import os, glob
import open3d as o3d
import numpy as np
from prepare_datasets import read_scans
from eval_loop import read_match_result, read_match_centroid_result, evaluate_fine



WEBRTC_IP = '143.89.46.75'
PORT = '8020'
os.environ.setdefault('WEBRTC_IP', WEBRTC_IP)
os.environ.setdefault('WEBRTC_PORT', PORT)
o3d.visualization.webrtc_server.enable_webrtc()
print('WebRTC server enabled at {}:{}'.format(os.environ['WEBRTC_IP'], os.environ['WEBRTC_PORT']))

def read_single_pose(t_dir):
    with open(t_dir, 'r') as f:
        lines=f.readlines()
        rows = []
        for line in lines[2:]:
            line = line.strip()
            line = line.split(' ')
            row = np.array(line).astype(float)
            rows.append(row)
        t = np.array(rows).reshape(4,4)
        # print(t)
        f.close()
        return t
    
def read_scans(dir):
    with open(dir, 'r') as f:
        scans = []
        lines = f.readlines()
        for line in lines:
            scans.append(line.strip())
        f.close()
        return scans

def read_trajectory(t_dir):
    with open(t_dir, 'r') as f:
        camera_poses = []
        lines=f.readlines()
        
        for t,line in enumerate(lines):
            elements = line.strip().split(' ')
            if len(elements) ==3:
                if t>0:
                    camera_poses.append(t_wc)
                t_wc = np.eye(4)
                row=0
            else:
                elements = [float(ele.strip()) for ele in elements]
                t_wc[row,:] = np.array(elements)
                row+=1
                
        print('load {} camera poses'.format(len(camera_poses)))

        f.close()
        return camera_poses

def camera_poses_to_line_set(camera_poses:list):
    n = len(camera_poses)
    points = []
    lines = []
    for i in range(n):
        points.append(camera_poses[i][:3,3])
        if i>0:
            lines.append([i-1,i])
    colors = [[1, 0, 0] for i in range(len(lines))]
    
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def camera_poses_to_centroids(camera_poses:list, color:np.ndarray=[1,0,0]):
    n = len(camera_poses)
    viz_centroids = []
    for i in range(n):
        centroid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        centroid_mesh.compute_vertex_normals()
        centroid_mesh.paint_uniform_color(color)
        centroid_mesh.translate(camera_poses[i][:3,3])
        viz_centroids.append(centroid_mesh)
        
    return viz_centroids


def correspondences_to_line_set(src_centroids:np.ndarray, ref_centroids:np.ndarray, tp_mask:np.ndarray):
    M = src_centroids.shape[0]
    points = []
    lines = []
    colors = []
    
    for i in range(M):
        points.append(src_centroids[i])
        points.append(ref_centroids[i])
        lines.append([2*i,2*i+1])
        if tp_mask[i]:
            colors.append([0,1,0])
        else:
            colors.append([1,0,0])
    
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set
    
    

# if __name__ == '__main__':
def viz_sequence():
    # 3RScan setting
    dataroot = '/data2/3rscan_raw'
    MAP_NAME = 'mesh_o3d.ply'
    split_file = 'val.txt'
    TRAJECTORY = False

    # General settings
    WEBRTC_IP = '143.89.46.75'
    
    if WEBRTC_IP is not None:
        os.environ.setdefault('WEBRTC_IP', WEBRTC_IP)
        os.environ.setdefault('WEBRTC_PORT', '8020')
        o3d.visualization.webrtc_server.enable_webrtc()
        print('WebRTC server enabled at {}:{}'.format(os.environ['WEBRTC_IP'], os.environ['WEBRTC_PORT']))
    
    scans = read_scans(os.path.join(dataroot,'splits',split_file))
    viz_geometries = []
    
    # offset_grid = OFFSET_DISTANCE * np.ones((OFFSET_LENGTH,OFFSET_LENGTH))

    for scan_id, scan in enumerate(scans):
        print(scan)
        scan_folder = os.path.join(dataroot, scan)

        mesh_map = o3d.io.read_point_cloud(os.path.join(scan_folder, MAP_NAME))
        
        # translation 
        translation = [0.0,0.0,5.0]
        # mesh_map.translate(translation)
        viz_geometries.append(mesh_map)
        
        if TRAJECTORY:
            camera_poses = read_trajectory(os.path.join(scan_folder, 'trajectory.log'))
            path = camera_poses_to_line_set(camera_poses)
            path.translate(translation)
            viz_geometries.append(path)
            
        if WEBRTC_IP is not None:
            o3d.visualization.draw(viz_geometries, non_blocking_and_return_uid=False)
        else:
            o3d.visualization.draw_geometries(viz_geometries, window_name=scan)
        # break
    
    
def lazy_viz():
    
    WEBRTC_IP = '143.89.46.75'
    PORT = '8080'
    
    if WEBRTC_IP is not None:
        os.environ.setdefault('WEBRTC_IP', WEBRTC_IP)
        os.environ.setdefault('WEBRTC_PORT', PORT)
        o3d.visualization.webrtc_server.enable_webrtc()
        print('WebRTC server enabled at {}:{}'.format(os.environ['WEBRTC_IP'], os.environ['WEBRTC_PORT']))
    
    map_dir = '/data2/sgslam/output/online_mapping/uc0110_00a/instance_map.ply'
    pcd = o3d.io.read_point_cloud(map_dir)
    print('load {} points'.format(len(pcd.points)))
    if WEBRTC_IP is not None:
        o3d.visualization.draw([pcd], non_blocking_and_return_uid=False)
    else:
        o3d.visualization.draw_geometries([pcd], window_name='map')

def render_two_maps(src_pcd, ref_pcd, T_ref_src):
    src_pcd.transform(T_ref_src)
    src_pcd.paint_uniform_color([0.707,0.707,0])
    ref_pcd.paint_uniform_color([0,0.707,0.707])

    if(WEBRTC_IP is not None):
        print('Visualizing using WebRTC')
        o3d.visualization.draw([src_pcd, ref_pcd], non_blocking_and_return_uid=False)

def render_map_correspondences(src_pcd, ref_pcd, src_centroids, ref_centroids, tp_mask):
    RADIUS = 0.2
    lineset = correspondences_to_line_set(src_centroids, ref_centroids, tp_mask)
    M = src_centroids.shape[0]
    viz_geometries = [src_pcd, ref_pcd, lineset]
    for i in range(M):
        mesh_src_centroid = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS)
        mesh_src_centroid.compute_vertex_normals()
        mesh_src_centroid.paint_uniform_color([1,0,0])
        mesh_src_centroid.translate(src_centroids[i])
        
        mesh_ref_centroid = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS)
        mesh_ref_centroid.compute_vertex_normals()
        mesh_ref_centroid.paint_uniform_color([0,0,1])
        mesh_ref_centroid.translate(ref_centroids[i])
        viz_geometries.append(mesh_src_centroid)
        viz_geometries.append(mesh_ref_centroid)
    
    if(WEBRTC_IP is not None):
        print('Visualizing using WebRTC')
        o3d.visualization.draw(viz_geometries, non_blocking_and_return_uid=False)
    else:
        o3d.visualization.draw_geometries(viz_geometries, window_name='correspondences')

def onloop_results():
    
    dataroot = '/data2/sgslam'
    output_root = '/data2/sgslam/output/online_coarse+'    
    agentA = 'uc0115_00b'
    agentB = 'uc0115_00a'
    frame_index = 100
    MASK_GT_MATCH = True
    INSTANCE_RADIUS = 1.0
    
    # load 
    pcd_src = o3d.io.read_point_cloud(os.path.join(output_root, agentA, 'instance_map.ply'))
    pcd_ref = o3d.io.read_point_cloud(os.path.join(output_root, agentB, 'instance_map.ply'))
    print('load {} points from agentA'.format(len(pcd_src.points)))
    print('load {} points from agentB'.format(len(pcd_ref.points)))    
    
    # Load one loop frame results
    loop_frames = glob.glob(os.path.join(output_root, agentA, agentB, '*.txt'))
    loop_frames = sorted(loop_frames)

    frame_index = max(0, min(frame_index, len(loop_frames)-1))
    print('showing the {}-th/{} loop frame {}'.format(frame_index,len(loop_frames), loop_frames[frame_index]))
    # pred_pose, _, _ = read_match_result(loop_frames[frame_index])
    
    pred_pose, src_centroids, ref_centroids = read_match_centroid_result(loop_frames[frame_index])
    print('{} instance correspondences'.format(len(src_centroids)))
    
    if MASK_GT_MATCH:
        gt_pose = np.loadtxt(os.path.join(dataroot, 'gt', '{}-{}.txt'.format(agentA, agentB)))
        _, tp_mask = evaluate_fine(src_centroids, ref_centroids, gt_pose, INSTANCE_RADIUS)
        # line_set = correspondences_to_line_set(src_centroids, ref_centroids, tp_mask)
        print('{}/{} true instance match'.format(tp_mask.sum, tp_mask.shape[0]))
        
        render_map_correspondences(pcd_src, pcd_ref, src_centroids, ref_centroids, tp_mask)
    else:
        print('Render with predicted pose\n', pred_pose)
        render_two_maps(pcd_src, pcd_ref, pred_pose)

def algin_agents_graph():
    ''' aligned map, and camera poses of two agents'''
    dataroot = '/data2/sgslam'
    output_root = '/data2/sgslam/output/online_coarse+'    
    agentA = 'uc0115_00b' # source
    agentB = 'uc0115_00a'
    POSE_RATIO = 0.1
    CAMERA_POSES = False
    RGB_MESH = False
    
    # load 
    pcd_src = o3d.io.read_point_cloud(os.path.join(output_root, agentA, 'instance_map.ply'))
    pcd_ref = o3d.io.read_point_cloud(os.path.join(output_root, agentB, 'instance_map.ply'))
    print('load {} points from agentA'.format(len(pcd_src.points)))
    print('load {} points from agentB'.format(len(pcd_ref.points)))    
    
    # Load one loop frame results
    loop_frames = glob.glob(os.path.join(output_root, agentA, agentB, '*.txt'))
    loop_frames = sorted(loop_frames)
    pred_pose, _, _ = read_match_centroid_result(loop_frames[-1])

    pcd_src.transform(pred_pose)
    pcd_src.paint_uniform_color([0.707,0.707,0])
    pcd_ref.paint_uniform_color([0,0.707,0.707])
    viz = [pcd_src, pcd_ref]
    
    #
    if CAMERA_POSES:
        from eval_trajectory import TrajectoryAnalysis
        src_scene_trajectory = TrajectoryAnalysis(os.path.join(dataroot,'scans',agentA))
        src_scene_trajectory.update_aligned_poses(output_root, agentB)
        src_scene_trajectory.update_icp_gt_poses(os.path.join(dataroot, 'gt', '{}-{}.txt'.format(agentA, agentB)))
        
        src_pred_poses = src_scene_trajectory.get_pred_poses_list(POSE_RATIO)
        src_gt_poses = src_scene_trajectory.get_gt_poses_listt(POSE_RATIO)
            
        viz += camera_poses_to_centroids(src_pred_poses, color=[1,0,0])  
        viz += camera_poses_to_centroids(src_gt_poses, color=[0,1,0])

    if RGB_MESH:
        mesh_src = o3d.io.read_triangle_mesh(os.path.join(dataroot, 'scans', agentA, 'mesh_o3d.ply'))
        mesh_ref = o3d.io.read_triangle_mesh(os.path.join(dataroot, 'scans', agentB, 'mesh_o3d.ply'))
        viz = [mesh_src, mesh_ref]

    if(WEBRTC_IP is not None):
        print('Visualizing using WebRTC')
        o3d.visualization.draw(viz, non_blocking_and_return_uid=False)
    

def gt_registration_results():
    
    dataroot = '/data2/sgslam'
    subfolder = 'output/gt_iou'
    src_scene = 'uc0110_00c'
    ref_scene = 'uc0110_00a'
    file_name = 'instance_map.ply' # 'mesh_o3d.ply'
    
    #
    pcd_src = o3d.io.read_point_cloud(os.path.join(dataroot, subfolder, src_scene, 'frame-000774.ply'))
    pcd_ref = o3d.io.read_point_cloud(os.path.join(dataroot, subfolder, ref_scene, 'frame-000665.ply'))
    
    #
    # T_drift = np.loadtxt(os.path.join(dataroot, 'scans', src_scene, 'T_drift.txt'))
    # T_ref_src = np.loadtxt(os.path.join(dataroot,'scans',src_scene,'T_ref_src.txt'))
    T_gt = np.loadtxt(os.path.join(dataroot, 'gt', '{}-{}.txt'.format(src_scene, ref_scene)))
    
    #
    T = T_gt
    # T = np.eye(4)
    render_two_maps(pcd_src, pcd_ref, T)

def single_map():
    dir = '/data2/sgslam/output/online_coarse/uc0115_00a/uc0115_00b/frame-000935_src.ply'
    pcd = o3d.io.read_point_cloud(dir)
    print('Load {} points'.format(len(pcd.points)))
    
    if(WEBRTC_IP is not None):
        print('Visualizing using WebRTC')
        o3d.visualization.draw([pcd], non_blocking_and_return_uid=False)
    
    
if __name__=='__main__':
    print('Visualization script')
    
    # viz_sequence()
    # lazy_viz()
    onloop_results()
    # gt_registration_results()
    
    # algin_agents_graph()
    
    # single_map()
    