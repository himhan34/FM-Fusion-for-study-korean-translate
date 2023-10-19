import os, glob
import argparse, json
import open3d as o3d
import open3d.core as o3c
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
COLOR20 = render_result.COLOR20

class O3d_View_Controller:
    def __init__(self,dir):
        with open(dir,'r') as f:
            view_data = json.load(f)
            assert 'trajectory' in view_data
            vp0 = view_data['trajectory'][0]
            self.fov = vp0['field_of_view']
            self.front = vp0['front']
            self.lookat = vp0['lookat']
            self.up = vp0['up']
            self.zoom = vp0['zoom']
    
    def update_visualizer(self, o3d_visualizer:o3d.visualization.Visualizer):
        o3d_view_control = o3d_visualizer.get_view_control()
        o3d_view_control.change_field_of_view(self.fov)
        o3d_view_control.set_front(self.front)
        o3d_view_control.set_lookat(self.lookat)
        o3d_view_control.set_up(self.up)
        o3d_view_control.set_zoom(self.zoom)
        # o3d_view_control.set_constant_z_far(15.0)
        o3d_view_control.set_constant_z_near(1.0)

def update_projection(instance_map:InstanceMap, voxel_centroids:np.ndarray, depth_map:np.ndarray, T_wc:np.ndarray, 
                      intrinsic:o3d.camera.PinholeCameraIntrinsic, min_view_points=100):
    # max_points = 8000
    MIN_OBSERVED_RATIO = 0.3
    
    image_dims = [intrinsic.height,intrinsic.width]
    count = 0
    depth_voxels_o3c = o3c.Tensor(voxel_centroids,dtype=o3c.int32)
    active_instances = []
    active_instances_size = []
    
    for idx, instance in instance_map.instance_map.items():
        centroid_homo = np.concatenate([instance.centroid,np.array([1])],axis=0)
        centroid_in_camera = np.linalg.inv(T_wc).dot(centroid_homo.reshape(4,1))
        uv_map = np.zeros((image_dims[0],image_dims[1]),dtype=np.uint8)

        if centroid_in_camera[2]<0.5 or centroid_in_camera[2]>5.0 or np.abs(centroid_in_camera[0])>5.0 or np.abs(centroid_in_camera[1])>5.0:
            continue
        
        # query viewd instance points
        points, valid_ratio = instance.query_voxel_centroids(depth_voxels_o3c)
        if points.size<min_view_points or valid_ratio<MIN_OBSERVED_RATIO:
            continue
        
        # if points.shape[0] > max_points:
        #     samples = np.random.choice(points.shape[0],max_points,replace=False)
        #     points = points[samples,:]
        
        normals = np.zeros((points.shape[0],3))+0.33
        mask, points_uv, _, _ = Project(points,normals,T_wc,intrinsic.intrinsic_matrix,image_dims,max_depth=5.0,min_depth=0.5)
        # filter_mask = FilterOcclusion(points_uv,depth_map,
        #                         max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        # mask = mask & filter_mask

        if mask.sum()> min_view_points:
            points_uv = points_uv[mask,:].astype(np.int32)
            uv_map[points_uv[:,1],points_uv[:,0]] = 1
            active_instances.append(idx)
            active_instances_size.append(instance.points.shape[0])

            instance.update_current_uv(uv_map)
            # count += 1
     
    print('{}/{} instances projected'.format(len(active_instances),len(instance_map.instance_map)))
    
    # sort active instances from small to large
    active_instances = [x for _,x in sorted(zip(active_instances_size,active_instances),reverse=False)]
    
    return active_instances

def save_projection(scene_folder, frame_name, rgb_img, detections:list, 
                    active_instances:list,instance_map:fuse_detection.ObjectMap, semantic_names:list, intrinsic:o3d.camera.PinholeCameraIntrinsic):
    image_dims = [intrinsic.height,intrinsic.width]
    output_folder = os.path.join(scene_folder,'tmp')
    projection_map = np.zeros((image_dims[0],image_dims[1],3),dtype=np.uint8)
    count = 0

    render_img = rgb_img.copy()
    render_img = cv2.cvtColor(render_img,cv2.COLOR_BGR2RGB)
    
    for zk in detections:
        if zk.assignment == 'new':
            zk_color = (0,0,255)
        elif zk.assignment == 'matched':
            zk_color = (0,255,0)
        else:
            zk_color = (255,0,0)
            # raise NotImplementedError
        
        cv2.rectangle(render_img,(int(zk.u0),int(zk.v0)),(int(zk.u1),int(zk.v1)),zk_color,1)
        cv2.putText(render_img, zk.get_label_str(), (int(zk.u0),int(zk.v0)+50) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, zk_color, 1)

    # Draw projected instance masks
    # for idx, instance in instance_map.instance_map.items():
    for idx in active_instances:
        instance = instance_map.instance_map[idx]
        if instance.uv_map is None:continue
        # if instance.uv_map is None or np.sum(instance.uv_map)<10: continue
        # label_id,_ = instance.estimate_label()
        uv_map = instance.uv_map
        # label_name = semantic_names[label_id]
        assert projection_map.shape[0]==uv_map.shape[0] and projection_map.shape[1]==uv_map.shape[1]
        instance_color = COLOR20[int(idx) % len(COLOR20)]
        instance_color = (int(instance_color[0]),int(instance_color[1]),int(instance_color[2]))
        uv_list = np.nonzero(uv_map)
        for uv in zip(uv_list[0],uv_list[1]):
            cv2.circle(projection_map,(uv[1],uv[0]),5,instance_color,-1)
    
    # Write instance labels
    # for idx, instance in instance_map.instance_map.items():
    for idx in active_instances:
        instance = instance_map.instance_map[idx]
        if instance.uv_map is None: continue
        # if instance.uv_map is None or np.sum(instance.uv_map)<10: continue

        label_id,_ = instance.estimate_label()
        label_name = semantic_names[label_id]

        uv_list = np.nonzero(instance.uv_map)
        uv_centroid = np.mean(uv_list,axis=1).astype(np.int32)
        label_name += ' {}'.format(instance.uv_map.sum())
        cv2.putText(projection_map, label_name, (uv_centroid[1],uv_centroid[0]) , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        count += 1
        
    cv2.putText(projection_map, '{}/{} proj.'.format(count,len(instance_map.instance_map)), (10,50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    
    # export 
    render_img = np.concatenate([render_img,projection_map],axis=1)
    cv2.imwrite(os.path.join(output_folder,'{}.png'.format(frame_name)),render_img)

def overlay_detection_mask(rgb:np.ndarray, detections:list):
    overlay = rgb.copy()
    
    for k, zk in enumerate(detections):
        zk_color = COLOR20[k % len(COLOR20)]
        
        # bbox
        # bbox_color = zk_color.list()
        bbox_color = (0,255,0)
        cv2.rectangle(overlay,(int(zk.u0),int(zk.v0)),(int(zk.u1),int(zk.v1)),bbox_color,1)
        
        # text
        text = zk.get_label_str()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1        
        pos = (int(zk.u0),int(zk.v0))
        x, y = pos
        text_color = (0,0,0)
        text_color_bg = (255, 255, 255)
        
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(overlay, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(overlay, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

        # mask
        color_mask = np.zeros((rgb.shape[0],rgb.shape[1],3),dtype=np.uint8)
        color_mask[zk.mask] = zk_color
        overlay = cv2.addWeighted(overlay,0.9,color_mask,0.5,0)

    # 
    white_gap = np.ones((64,rgb.shape[1],3),dtype=np.uint8)*255
    out = np.concatenate([rgb,white_gap,overlay,white_gap],axis=0)
    whilte_col = np.ones((out.shape[0],32,3),dtype=np.uint8)*255
    out = np.concatenate([whilte_col,out,whilte_col],axis=1)

    return out

def filter_depth(depth:np.ndarray, mask:np.ndarray, max_percentile=0.9, min_percentile=0.1):
    depth_zk = np.zeros(depth.shape,dtype=np.uint16)
    depth_zk[mask] = depth[mask]
    
    depth_list = depth_zk[mask].reshape(-1)
    depth_list = sorted(depth_list)
    max_depth = depth_list[int(len(depth_list)*max_percentile)-1]
    min_depth = depth_list[int(len(depth_list)*min_percentile)]
    filter_mask = np.logical_and(depth_zk>min_depth,depth_zk<max_depth)
    # depth = depth[filter_mask]  

    depth_zk[~filter_mask] = 0
    return depth_zk

def generate_voxel_from_points(rgb:o3d.geometry.Image, depth:o3d.geometry.Image,
                               resolution:int, depth_scale:float, T_wc:np.ndarray, intrinsic:o3d.camera.PinholeCameraIntrinsic):
    scale_volume =o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / resolution,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,depth,
                                                              depth_scale=depth_scale,depth_trunc=4.0,convert_rgb_to_intensity=False)
    scale_volume.integrate(rgbd,intrinsic,np.linalg.inv(T_wc))
    
    return scale_volume.extract_voxel_point_cloud()

def calculate_bbox(mask_map:np.ndarray):
    uv = np.argwhere(mask_map) # (N,2)
    u0 = np.min(uv[:,1])
    v0 = np.min(uv[:,0])
    u1 = np.max(uv[:,1])
    v1 = np.max(uv[:,0])
    area = (u1-u0)*(v1-v0)
    return area

def find_assignment(detections:list,instance_map:InstanceMap,active_instances:list, 
                    min_iou=0.5, verbose=False):
    '''
    Output: 
        - mathches: (K), int8. Matched is 1, or -1 if not matched
    '''
    K = len(detections)
    M = len(active_instances)
    
    iou = np.zeros((K,M))
    assignment = np.zeros((K,M),dtype=np.int32)
    matches = np.zeros((K),dtype=np.int32) - 1
    instance_indices = np.zeros((M),dtype=np.int32)
    if len(active_instances)<1 or K<1: return matches, []
    
    # compute iou
    flag = 0
    for k_,zk in enumerate(detections):
        j_=0
        zk_area = zk.get_bbox_area()
        for idx in active_instances:
            uv_j = instance_map.instance_map[idx].uv_map
            if uv_j.sum()>10:
                overlap = np.logical_and(zk.mask,uv_j)
                iou[k_,j_] = np.sum(overlap)/(np.sum(uv_j)) # +np.sum(zk.mask)-np.sum(overlap))
                bbox_ratio = calculate_bbox(uv_j)/zk_area
                if bbox_ratio<0.1:
                    iou[k_,j_] = 0.0
                    flag +=1 
                    # print('skip small instance observation!!!')
                
            j_+=1
            instance_indices[j_-1] = int(idx)

    if flag>0:
        print('{} small instances skipped'.format(flag))

    # update assignment 
    # assignment[np.arange(K),np.argmax(iou,1)] = 1 # maximum match for each row
    assignment[np.argmax(iou,0),np.arange(M)] = 1 # maximu match for each column
    valid_match = (iou > min_iou).astype(np.int32)
    assignment = assignment & valid_match # (K,M)
    
    # instances_bin = assignment.sum(0) > 1
    instances_bin = assignment.sum(1) > 1
    if instances_bin.any(): # multiple detections assigned to one instance
        row_wise_maximum = np.zeros((K,M),dtype=np.int32)
        row_wise_maximum[np.arange(K),np.argmax(iou,1)] = 1
        assignment = assignment & row_wise_maximum

    #
    for k in range(K):
        if assignment[k,:].sum()>0:
            matches[k] = instance_indices[assignment[k,:].argmax()]
    # print(iou)
    # print(assignment)
    # print(matches)
    
    # miss observed instances
    missed = []
    for idx in active_instances:
        if int(idx) not in matches:
            missed.append(idx)
    # if len(missed)>1:
    #     print('debug')

    return matches, missed

def extract_object_map(instance_map:fuse_detection.ObjectMap, scene_name:str, viz_folder:str):
    _, voxel_points, instances, semantics = instance_map.extract_instance_voxel_map()
    if voxel_points is None: return None, None
    composite_labels = 1000 * semantics + instances + 1
    semantic_colors, instance_colors = GenerateColors(composite_labels.astype(np.int64))
    
    # 
    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(voxel_points)
    semantic_pcd.colors = o3d.utility.Vector3dVector(semantic_colors.astype(np.float32)/255.0)
    
    instance_pcd = o3d.geometry.PointCloud()
    instance_pcd.points = o3d.utility.Vector3dVector(voxel_points)
    instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors.astype(np.float32)/255.0)
    
    return semantic_pcd, instance_pcd

def save_visualization(instance_map:fuse_detection.ObjectMap, scene_name:str, viz_folder:str):
    _, voxel_points, instances, semantics = instance_map.extract_instance_voxel_map()
    composite_labels = 1000 * semantics + instances + 1
    semantic_colors, instance_colors = GenerateColors(composite_labels.astype(np.int64))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_points)
    pcd.colors = o3d.utility.Vector3dVector(semantic_colors.astype(np.float32)/255.0)
    o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),pcd)
    pcd.colors = o3d.utility.Vector3dVector(instance_colors.astype(np.float32)/255.0)
    o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),pcd) 

def integrate_semantic_map(scene_dir:str, dataroot:str, out_folder:str, pred_folder_name:str, label_predictor:fuse_detection.LabelFusion, visualize:bool):
    scene_name = os.path.basename(scene_dir)
    pred_folder = os.path.join(scene_dir,pred_folder_name)
    viz_folder = os.path.join(dataroot,'output',out_folder)
    DENSE_MAPPING = True
    MERGE_INSTANCES = True

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
        MIN_VIEW_POINTS = 800 # instance-wise
        MIN_MASK_POINTS = 1000 # detection-wise pcd
        FILTER_DETECTION_IOU = 0.1
        ASSIGNMENT_IOU = 0.5
        NMS_IOU = 0.4
        NMS_SIMILARITY = 0.15
        INFLAT_RATIO = 2.0
        SMALL_INSTANCE_SIZE = 500
        SMALL_INSTANCE_NEG = 5
        REFINE_FRAME_GAP = 200
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True)    
    elif 'scenenn' in dataroot:
        DEPTH_SCALE = 1000.0
        RGB_FOLDER = 'image'
        RGB_POSFIX = '.png'
        DATASET = 'scenenn'
        FRAME_GAP = 5
        VX_RESOLUTION = 256
        MIN_VIEW_POINTS = 800 # instance-wise
        MIN_MASK_POINTS = 1000 # detection-wise pcd
        FILTER_DETECTION_IOU = 0.1
        ASSIGNMENT_IOU = 0.5
        NMS_IOU = 0.4
        NMS_SIMILARITY = 0.15
        INFLAT_RATIO = 2.0
        SMALL_INSTANCE_SIZE = 2000
        SMALL_INSTANCE_NEG = 8
        REFINE_FRAME_GAP = 300
        K_rgb,K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'),align_depth=True) 
    else:
        raise NotImplementedError
    
    # Init
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(depth_dim[1],depth_dim[0],K_depth[0,0],K_depth[1,1],K_depth[0,2],K_depth[1,2])
    # frame_sequence = glob.glob(os.path.join(scene_dir,pred_folder,'*_label.json'))  
    frame_sequence = glob.glob(os.path.join(scene_dir,RGB_FOLDER,'*{}'.format(RGB_POSFIX)))
    print('---- {}/{} find {} rgb frames'.format(DATASET,scene_name,len(frame_sequence)))
    
    # Create instance map manager
    instance_map = InstanceMap(None,None)
    instance_map.load_semantic_names(label_predictor.closet_names)
    recent_instances = []

    # Integration
    prev_frame_stamp = -100
    prev_merge_stamp = 0
    frame_time_array = np.zeros((5),dtype=np.float32) # [frames_count,t1,t2,t3,t4]
    object_time_array = np.zeros((6),dtype=np.float32) # [aggregated_instances_count, aggregate_active_count,t1,t2,t3,t4]
    
    # 
    if DENSE_MAPPING:
        global_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / VX_RESOLUTION,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    #
    if visualize:
        o3d_visulizer = o3d.visualization.Visualizer()
        o3d_visulizer.create_window()
        o3d_view_control = O3d_View_Controller(os.path.join(scene_dir,'o3d_view.json'))
        o3d_view_control.update_visualizer(o3d_visulizer)
        o3d_visulizer.poll_events()
        o3d_visulizer.update_renderer()
        prev_viz_pcd = None
        prev_viz_lineset = None
        positions = []
        lines = []
    
    count_frames = 0
    
    for i,frame_dir in enumerate(sorted(frame_sequence)):   
        frame_name = os.path.basename(frame_dir).split('_')[0][:-4] 
        if DATASET=='scannet':
            frame_stamp = float(frame_name[6:])
            depth_dir = os.path.join(scene_dir,'depth',frame_name+'.png')
            pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        elif DATASET=='scenenn':
            frame_stamp = float(frame_name[5:])
            depth_dir = os.path.join(scene_dir,'depth',frame_name.replace('image','depth')+'.png')
            pose_dir = os.path.join(scene_dir,'pose',frame_name.replace('image','frame-')+'.txt')

        else:
            raise NotImplementedError
        
        # if frame_stamp>600: break
        
        # Load RTB-D and pose
        rgbdir = os.path.join(scene_dir,RGB_FOLDER,frame_name+RGB_POSFIX)
        # pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        if os.path.exists(pose_dir)==False:
            print('no pose file for frame {}. Stop the fusion.'.format(frame_name))
            break
        T_wc = np.loadtxt(pose_dir)
        if T_wc.size<1:
            print('Skip empty pose')
            continue

        # Integrate global map
        if DENSE_MAPPING:
            print('frame {}: integrate global map'.format(frame_name))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(rgbdir),o3d.io.read_image(depth_dir),
                                                                    depth_scale=DEPTH_SCALE,depth_trunc=4.0,convert_rgb_to_intensity=False)
            global_volume.integrate(rgbd,intrinsic,np.linalg.inv(T_wc))

        # Is it prediction frame
        if (frame_stamp - prev_frame_stamp) < FRAME_GAP:
            continue
        tags, detections = fuse_detection.load_pred(pred_folder,frame_name,label_predictor.openset_names)
        if tags is None or detections is None: continue
        print('{}: prediction frame'.format(frame_name))
        rgb_np = cv2.imread(rgbdir,cv2.IMREAD_UNCHANGED)
        depth_np = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        scaled_depth_np = depth_np.astype(np.float32)/DEPTH_SCALE
        # rgb_np = cv2.cvtColor(rgb_np,cv2.COLOR_RGB2BGR)
        assert depth_np.shape[0]==depth_dim[0] and depth_np.shape[1]==depth_dim[1], 'depth image dimension does not match'
        assert depth_np.shape[0] == rgb_np.shape[0] and depth_np.shape[1] == rgb_np.shape[1]
                
        # Prepare detections
        t_start = time.time()
        detections = fuse_detection.filter_overlap_detections(detections= detections, min_iou=FILTER_DETECTION_IOU)
        detections = fuse_detection.filter_detection_depth(detections,scaled_depth_np)
        t_1 = time.time()  

        # query observed instances
        depth_img = o3d.geometry.Image(depth_np)
        depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img,intrinsic,np.linalg.inv(T_wc),depth_scale=DEPTH_SCALE,depth_trunc=4.0)
        depth_voxels_centroid = np.floor(np.asarray(depth_pcd.points)/(4.0/VX_RESOLUTION)).astype(np.int32)
        active_instances = update_projection(instance_map,depth_voxels_centroid, scaled_depth_np,T_wc,intrinsic,min_view_points=MIN_VIEW_POINTS)
        t_2 = time.time()


        # Association
        matches, missed = find_assignment(detections,instance_map,active_instances,min_iou=ASSIGNMENT_IOU,verbose=False)
        t_3 = time.time()

        # Integrate
        count_matched = 0
        projection_detections = []
        for k, zk in enumerate(detections):
            valid_zk = label_predictor.is_valid_labels(zk.labels)
            if valid_zk==False: continue
            # or np.sum(zk.mask)<MIN_MASK_POINTS: continue            
            
            # prepare detection mask
            depth_zk = filter_depth(depth_np, zk.mask, max_percentile=0.9, min_percentile=0.1)
            depth_zk = o3d.geometry.Image(depth_zk)
            pcd_zk = generate_voxel_from_points(o3d.geometry.Image(rgb_np),depth_zk,VX_RESOLUTION,DEPTH_SCALE,T_wc,intrinsic)            
            if len(pcd_zk.points)<MIN_MASK_POINTS: continue
                        
            if matches[k] < -0.1:# new instance            
                new_instance = fuse_detection.Instance(instance_map.max_instance_id,J=len(label_predictor.closet_names))
                new_instance.create_voxel_map(pcd_zk,VX_RESOLUTION)
                new_instance.prob_vector,flag = label_predictor.update_measurement(zk.labels,new_instance.prob_vector)
                if flag:
                    new_instance.save_label_measurements(zk.labels)
                    instance_map.insert_instance(new_instance)
                    zk.assignment = 'new'
                    recent_instances.append(str(new_instance.id))
            else: # matched instance
                matched_instance = instance_map.instance_map[str(matches[k])]
                matched_instance.prob_vector, flag = label_predictor.update_measurement(zk.labels,matched_instance.prob_vector)
                if flag:       
                    matched_instance.integrate_volume_debug(pcd_zk)
                    matched_instance.save_label_measurements(zk.labels)
                    count_matched +=1
                    zk.assignment = 'matched'
                    if str(matches[k]) not in recent_instances:
                        recent_instances.append(str(matches[k]))
                    
            projection_detections.append(zk)
        t_4 = time.time()
        
        #
        for idx in missed:
            instance_map.instance_map[idx].neg_observed += 1

        # save_projection(scene_dir, frame_name, rgb_np, projection_detections, active_instances, instance_map, label_predictor.closet_names, intrinsic) # for debug
        
        # Refine and update
        if MERGE_INSTANCES:
            ret = instance_map.merge_overlap_instances(active_instances=active_instances, 
                                                nms_iou=NMS_IOU, nms_similarity=NMS_SIMILARITY, inflat_ratio=INFLAT_RATIO)
        instance_map.update_volume_points()        
        if DENSE_MAPPING and (frame_stamp - prev_merge_stamp) >=REFINE_FRAME_GAP:
            global_pcd = global_volume.extract_point_cloud()
            refined_flag = instance_map.update_instance_with_global_map(global_pcd,recent_instances,50)
            recent_instances = []
            if refined_flag: 
                print('********** instance merged with global map ************')
                prev_merge_stamp = frame_stamp

        #
        instance_map.remove_rs_small_instances(max_negative_observe=SMALL_INSTANCE_NEG, min_points=SMALL_INSTANCE_SIZE)
        
        prev_frame_stamp = frame_stamp
        print('{} matched. {} existed instances'.format(count_matched,instance_map.get_num_instances()))
        print('projection {:.3f} s, da {:.3f} s, volumes {:.3f} s'.format(t_1-t_start,t_2-t_1,t_3-t_2))
        frame_time_array += np.array([1,t_1-t_start,t_2-t_1,t_3-t_2,t_4-t_3])
        object_time_array += np.array([instance_map.get_num_instances(),len(active_instances),t_1-t_start,t_2-t_1,t_3-t_2,t_4-t_3])
        count_frames +=1
        # break
        
        #
        if visualize:
            semantic_pcd, instance_pcd = extract_object_map(instance_map, scene_name, viz_folder)
            positions.append(T_wc[:3,3])
            lines.append([len(positions)-2,len(positions)-1])
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(positions),lines=o3d.utility.Vector2iVector(lines))
            
            if semantic_pcd is None: continue
            if prev_viz_pcd is not None:
                o3d_visulizer.remove_geometry(prev_viz_pcd)
                # o3d_visulizer.remove_geometry(prev_viz_lineset)
                # cv2.destroyWindow(prev_o3d_viz)
            o3d_visulizer.add_geometry(semantic_pcd)
            # o3d_visulizer.add_geometry(line_set)
            o3d_view_control.update_visualizer(o3d_visulizer)
            o3d_visulizer.poll_events()
            o3d_visulizer.update_renderer()

            rgb_masked = overlay_detection_mask(rgb_np, projection_detections)
            cv2.imshow('rgb',rgb_masked)
            cv2.waitKey(10) & 0XFF
            prev_viz_pcd=semantic_pcd
            prev_viz_lineset = line_set

    instance_map.update_volume_points()
    # Export 
    print('finished scene')
    debug_folder = os.path.join(dataroot,'debug',out_folder,scene_name)
    eval_folder = os.path.join(dataroot,'eval',out_folder,scene_name)
    if os.path.exists(debug_folder)==False: os.makedirs(debug_folder)
    # if os.path.exists(eval_folder)==False: os.makedirs(eval_folder)
    # instance_map.load_dense_points(os.path.join(scene_dir,'{}.ply'.format(MAP_POSFIX)))
    instance_map.save_debug_results(debug_folder,vx_resolution=VX_RESOLUTION,time_record=[frame_time_array,object_time_array])
    
    # Save visualization
    save_visualization(instance_map, scene_name, viz_folder)
    
    if visualize:
        o3d_visulizer.destroy_window()
        cv2.destroyAllWindows()
    
    if DENSE_MAPPING:
        global_mesh = global_volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join(scene_dir,'dense_map.ply'),global_mesh)
    
    # Save evaluation results
    # if DATASET=='scannet':
    #     label_pcd = o3d.io.read_point_cloud(gt_map_dir)
    #     label_points = np.asarray(label_pcd.points,dtype=np.float32)
    #     instance_map.save_scannet_results(eval_folder,label_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--prior_model', help='directory to likelihood model')
    parser.add_argument('--output_folder', help='folder name')
    parser.add_argument('--prediction_folder', help='prediction folder in each scan', default='prediction_no_augment')
    parser.add_argument('--split', help='split', default='val')
    parser.add_argument('--split_file', help='split file name', default='val')
    parser.add_argument('--scan', help='single scan name')
    parser.add_argument('--visualize', help='visualize', action='store_true')
    opt = parser.parse_args()

    FUSE_ALL_TOKENS = True
    label_predictor = fuse_detection.LabelFusion(opt.prior_model, fusion_method='bayesian')
    if opt.scan is not None:
        scans = [opt.scan]
        print('Read {}  to construct map'.format(scans))
    else:
        scans = fuse_detection.read_scans(os.path.join(opt.data_root,'splits','{}.txt'.format(opt.split_file)))
        print('Read {} scans to construct map'.format(len(scans)))

    debug_folder = os.path.join(opt.data_root,'debug',opt.output_folder)
    viz_folder = os.path.join(opt.data_root,'output',opt.output_folder)
    eval_folder = os.path.join(opt.data_root,'eval',opt.output_folder)
    if os.path.exists(debug_folder)==False: os.makedirs(debug_folder)
    if os.path.exists(viz_folder)==False: os.makedirs(viz_folder)
    if os.path.exists(eval_folder)==False: os.makedirs(eval_folder)
    
    # scans = ['scene0011_01']
    
    for scan in scans:
        args = os.path.join(opt.data_root,opt.split,scan), opt.data_root, opt.output_folder, opt.prediction_folder, label_predictor, opt.visualize
        integrate_semantic_map(*args)
        # break
