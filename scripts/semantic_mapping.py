import os, glob
import argparse
import open3d as o3d
import numpy as np
import cv2
import project_util
import fuse_detection
import render_result
import time
import multiprocessing as mp

Detection = fuse_detection.Detection
InstanceMap = fuse_detection.ObjectMap
Project = project_util.project
FilterOcclusion = project_util.filter_occlusion
GenerateColors = render_result.generate_colors


def run_projection_func(args):
    idx, points, depth, T_wc, image_dims, intrinsic_matrix,min_mask = args
    max_dist_gap = 0.2
    depth_kernal_size = 5
    kernal_valid_ratio = 0.2
    kernal_max_var = 0.15
    max_points = 2000
    
    uv_map = np.zeros((image_dims[0],image_dims[1]),dtype=np.uint8)
    if points.shape[0]<1:
        return (idx, uv_map)
    elif points.shape[0]>max_points:
        sample_indices = np.random.choice(points.shape[0],max_points,replace=False)
    else:
        sample_indices = np.arange(points.shape[0])
        
    sample_points = points[sample_indices,:]
    normals = np.zeros((sample_points.shape[0],3))+0.33
    mask, points_uv, _, _ = Project(sample_points,normals,T_wc,intrinsic_matrix,image_dims,max_depth=5.0,min_depth=0.5)
    filter_mask = FilterOcclusion(points_uv,depth,
                                    max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
    mask = mask & filter_mask
    if mask.sum()> min_mask:
        points_uv = points_uv[mask,:].astype(np.int32)
        uv_map[points_uv[:,1],points_uv[:,0]] = 1
        
    return (idx, uv_map)

def update_projection(instance_map:InstanceMap, depth:np.ndarray, T_wc:np.ndarray, intrinsic:o3d.camera.PinholeCameraIntrinsic, min_mask=500):
    image_dims = [intrinsic.height,intrinsic.width]
    # depth filter related params
    max_dist_gap = 0.2
    depth_kernal_size = 5
    kernal_valid_ratio = 0.2
    kernal_max_var = 0.15
    
    count = 0
    points_map = {} # instance_id -> points
    
    for idx, instance in instance_map.instance_map.items():
        centroid_homo = np.concatenate([instance.centroid,np.array([1])],axis=0)
        centroid_in_camera = np.linalg.inv(T_wc).dot(centroid_homo.reshape(4,1))

        if centroid_in_camera[2]<0.5 or centroid_in_camera[2]>5.0 or np.abs(centroid_in_camera[0])>5.0 or np.abs(centroid_in_camera[1])>5.0:
            uv_map = np.zeros((image_dims[0],image_dims[1]),dtype=np.uint8)
        else:
            # points_map[idx] = instance.points
            _, uv_map = run_projection_func((idx,instance.points,depth,T_wc,image_dims,intrinsic.intrinsic_matrix,min_mask))
            count += 1
        instance.update_current_uv(uv_map)

        continue
        points = instance.points
        normals = np.zeros((points.shape[0],3))+0.33
        mask, points_uv, _, _ = Project(points,normals,T_wc,intrinsic.intrinsic_matrix,image_dims,max_depth=5.0,min_depth=0.5)
        filter_mask = FilterOcclusion(points_uv,depth,
                                        max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        mask = mask & filter_mask
        # if mask.sum() < min_mask: continue # skip instance with too few observation points
        if mask.sum()> min_mask:
            points_uv = points_uv[mask,:].astype(np.int32)
            uv_map[points_uv[:,1],points_uv[:,0]] = 1
                
    print('{}/{} instances projected'.format(count,len(instance_map.instance_map)))
    
    return None
    p = mp.Pool(processes=64)
    mask_map = p.map(run_projection_func,
                     [(idx,points,depth,T_wc,image_dims,intrinsic.intrinsic_matrix,min_mask) for idx,points in points_map.items()])
    
    for idx, uv_map in mask_map:
        instance_map.instance_map[str(idx)].update_current_uv(uv_map)
        if uv_map.sum()>0: count+=1
    
    print('projected')

def find_assignment(detections:list[Detection],instance_map:InstanceMap, 
                    min_iou=0.5, verbose=False):
    '''
    Output: 
        - mathches: (K), int8. Matched is 1, or -1 if not matched
    '''
    K = len(detections)
    M = instance_map.get_num_instances()
    
    iou = np.zeros((K,M))
    assignment = np.zeros((K,M),dtype=np.int32)
    matches = np.zeros((K),dtype=np.int32) - 1
    instance_indices = np.zeros((M),dtype=np.int32)
    if M<1: return matches
    
    # compute iou
    for k_,zk in enumerate(detections):
        j_=0
        for idx, instance in instance_map.instance_map.items():
            uv_j = instance.uv_map
            if uv_j.sum()>10:
                overlap = np.logical_and(zk.mask,uv_j)
                iou[k_,j_] = np.sum(overlap)/(np.sum(uv_j)) #+np.sum(uv_k)-np.sum(overlap))
            j_+=1
            instance_indices[j_-1] = int(idx)

    # update assignment 
    assignment[np.arange(K),np.argmax(iou,1)] = 1 # maximum match for each row
    
    instances_bin = assignment.sum(1) > 1
    if instances_bin.any(): # multiple detections assigned to one instance
        iou_col_max = iou.max(0)
        valid_col_max = np.abs(iou - np.tile(iou_col_max,(K,1))) < 1e-6 # non-maximum set to False
        assignment = assignment & valid_col_max
        
    valid_match = (iou > min_iou).astype(np.int32)
    assignment = assignment & valid_match # (K,M)
    
    #
    for k in range(K):
        if assignment[k,:].sum()>0:
            matches[k] = instance_indices[assignment[k,:].argmax()]
    # print(iou)
    # print(assignment)
    # print(matches)

    return matches

# def integrate_semantic_map(args):
def integrate_semantic_map(scene_dir:str, dataroot:str, out_folder:str, pred_folder_name:str, label_predictor:fuse_detection.LabelFusion, visualize:bool):
    scene_name = os.path.basename(scene_dir)
    pred_folder = os.path.join(scene_dir,pred_folder_name)
    
    print('Integrating {}'.format(scene_name))
    assert os.path.exists(os.path.join(scene_dir,'intrinsic')), 'intrinsic file not found'
    
    if 'ScanNet' in dataroot:
        gt_map_dir = os.path.join(scene_dir,'{}_{}'.format(scene_name,'vh_clean_2.ply'))
        DEPTH_SCALE = 1000.0
        MAP_POSFIX = 'mesh_o3d_256'
        RGB_FOLDER = 'color'
        RGB_POSFIX = '.jpg'
        DATASET = 'scannet'
        FRAME_GAP = 10
        VX_RESOLUTION = 256
        MIN_VIEW_POINTS = 200 # instance-wise
        MIN_MASK_DETECTION = 1000
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True)    

    else:
        raise NotImplementedError
    
    # Init
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(depth_dim[1],depth_dim[0],K_depth[0,0],K_depth[1,1],K_depth[0,2],K_depth[1,2])
    predict_frames =  glob.glob(os.path.join(scene_dir,pred_folder,'*_label.json'))  
    print('---- {}/{} find {} prediction frames'.format(DATASET,scene_name,len(predict_frames)))
    
    # Create TSDF volume
    instance_map = InstanceMap(None,None)
    instance_map.load_semantic_names(None)

    # Integration
    prev_frame_stamp = -100
    time_array = np.zeros((3),dtype=np.float32)
    count_frames = 0
    
    for i,pred_frame in enumerate(sorted(predict_frames)):   
        frame_name = os.path.basename(pred_frame).split('_')[0] 
        if DATASET=='scannet':
            frame_stamp = float(frame_name.split('-')[-1])
            depth_dir = os.path.join(scene_dir,'depth',frame_name+'.png')
        else:
            raise NotImplementedError
        
        # if frame_stamp>50: break
        if (frame_stamp - prev_frame_stamp) < FRAME_GAP:
            continue
        
        print('Processing frame {}'.format(frame_name))
        
        # Load RTB-D and pose
        rgbdir = os.path.join(scene_dir,RGB_FOLDER,frame_name+RGB_POSFIX)
        pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        if os.path.exists(pose_dir)==False:
            print('no pose file for frame {}. Stop the fusion.'.format(frame_name))
            break
        rgb_np = cv2.imread(rgbdir,cv2.IMREAD_UNCHANGED)
        depth_np = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        scaled_depth_np = depth_np.astype(np.float32)/DEPTH_SCALE
        rgb_np = cv2.cvtColor(rgb_np,cv2.COLOR_RGB2BGR)
        assert depth_np.shape[0]==depth_dim[0] and depth_np.shape[1]==depth_dim[1], 'depth image dimension does not match'
        assert depth_np.shape[0] == rgb_np.shape[0] and depth_np.shape[1] == rgb_np.shape[1]
        T_wc = np.loadtxt(pose_dir)
        # color = o3d.io.read_image(rgbdir)
        color = o3d.geometry.Image(rgb_np)

        # load prediction
        t_start = time.time()
        tags, detections = fuse_detection.load_pred(pred_folder,frame_name,label_predictor.openset_names)
        update_projection(instance_map,scaled_depth_np,T_wc,intrinsic,min_mask=MIN_VIEW_POINTS)
        print('projected')  
        t_1 = time.time()  
            
        # Association
        matches = find_assignment(detections,instance_map,min_iou=0.5,verbose=False)
        t_2 = time.time()

        # Integrate
        count_matched = 0
        for k, zk in enumerate(detections):
            prob_current, weight = label_predictor.estimate_single_prob(zk.labels)
            if weight < 1e-6 or np.sum(zk.mask)<MIN_MASK_DETECTION: continue            
            
            # prepare detection mask
            depth_zk = np.zeros(depth_np.shape,dtype=np.uint16)
            depth_zk[zk.mask] = depth_np[zk.mask]
            depth = o3d.geometry.Image(depth_zk)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,
                                                                      depth_scale=DEPTH_SCALE,depth_trunc=4.0,convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth,intrinsic,np.linalg.inv(T_wc),depth_scale=DEPTH_SCALE,depth_trunc=4.0) 
            pcd = pcd.voxel_down_sample(voxel_size=4.0/VX_RESOLUTION)
            if pcd.has_points()==False: continue
            
            if matches[k] < -0.1:# new instance            
                new_instance = fuse_detection.Instance(instance_map.get_num_instances(),J=prob_current.shape[0])
                new_instance.create_volume_debug(pcd,VX_RESOLUTION)
                # new_instance.create_volume(VX_RESOLUTION)
                # new_instance.volume.integrate(rgbd,intrinsic,np.linalg.inv(T_wc))
                new_instance.update_label_probility(prob_current,weight,zk.labels)
                instance_map.insert_instance(new_instance)
            else: # matched instance
                matched_instance = instance_map.instance_map[str(matches[k])]
                matched_instance.integrate_volume_debug(pcd)
                # matched_instance.volume.integrate(rgbd,intrinsic,np.linalg.inv(T_wc))
                matched_instance.update_label_probility(prob_current,weight,zk.labels)
                count_matched +=1
        t_3 = time.time()

        instance_map.update_volume_points()
        
        prev_frame_stamp = frame_stamp
        print('{} matched. {} existed instances'.format(count_matched,instance_map.get_num_instances()))
        print('projection {:.3f} s, da {:.3f} s, volumes {:.3f} s'.format(t_1-t_start,t_2-t_1,t_3-t_2))
        time_array += np.array([t_1-t_start,t_2-t_1,t_3-t_2])
        count_frames +=1
        # break
    
    instance_map.update_volume_points()
    # Export 
    print('finished scene')
    debug_folder = os.path.join(dataroot,'debug',out_folder,scene_name)
    eval_folder = os.path.join(dataroot,'eval',out_folder,scene_name)
    if os.path.exists(debug_folder)==False: os.makedirs(debug_folder)
    # if os.path.exists(eval_folder)==False: os.makedirs(eval_folder)
    # instance_map.load_dense_points(os.path.join(scene_dir,'{}.ply'.format(MAP_POSFIX)))
    _, voxel_points, instances, semantics = instance_map.extract_instance_voxel_map()
    time_array = time_array/count_frames
    instance_map.save_debug_results(debug_folder,vx_resolution=VX_RESOLUTION,time_record=time_array)
    
    # Save evaluation results
    # if DATASET=='scannet':
    #     label_pcd = o3d.io.read_point_cloud(gt_map_dir)
    #     label_points = np.asarray(label_pcd.points,dtype=np.float32)
    #     instance_map.save_scannet_results(eval_folder,label_points)

    # Save visualization
    composite_labels = 1000 * semantics + instances + 1
    semantic_colors, instance_colors = GenerateColors(composite_labels.astype(np.int64))
    viz_folder = os.path.join(dataroot,'output',out_folder)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_points)
    pcd.colors = o3d.utility.Vector3dVector(semantic_colors.astype(np.float32)/255.0)
    o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),pcd)
    pcd.colors = o3d.utility.Vector3dVector(instance_colors.astype(np.float32)/255.0)
    o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--prior_model', help='directory to likelihood model')
    parser.add_argument('--output_folder', help='folder name')
    parser.add_argument('--prediction_folder', help='prediction folder in each scan', default='prediction_no_augment')
    parser.add_argument('--split', help='split', default='val')
    parser.add_argument('--split_file', help='split file name', default='val')
    opt = parser.parse_args()

    FUSE_ALL_TOKENS = True
    label_predictor = fuse_detection.LabelFusion(opt.prior_model, fuse_all_tokens=FUSE_ALL_TOKENS)
    scans = fuse_detection.read_scans(os.path.join(opt.data_root,'splits','{}.txt'.format(opt.split_file)))
    print('Read {} scans to construct map'.format(len(scans)))

    debug_folder = os.path.join(opt.data_root,'debug',opt.output_folder)
    viz_folder = os.path.join(opt.data_root,'output',opt.output_folder)
    eval_folder = os.path.join(opt.data_root,'eval',opt.output_folder)
    if os.path.exists(debug_folder)==False: os.makedirs(debug_folder)
    if os.path.exists(viz_folder)==False: os.makedirs(viz_folder)
    if os.path.exists(eval_folder)==False: os.makedirs(eval_folder)
    
    # exit(0)
    # scans = ['scene0011_01']
    
    for scan in scans:
        args = os.path.join(opt.data_root,opt.split,scan), opt.data_root, opt.output_folder, opt.prediction_folder, label_predictor, False
        integrate_semantic_map(*args)
        # break
