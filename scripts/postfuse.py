import os 
import fuse_detection
import numpy as np
import open3d as o3d
import open3d.core as o3c
import argparse

SEMANTIC_NAMES = fuse_detection.SEMANTIC_NAMES
SEMANTIC_IDX = fuse_detection.SEMANTIC_IDX
    
def write_scannet_gt(voxel_map,clean_map_dir,out_folder):
    pcd = o3d.io.read_point_cloud(clean_map_dir)
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)
    N = points.shape[0]
    voxel_length = 4.0 / voxel_map['resolution']
    
    query_voxels = np.floor(points/voxel_length).astype(np.int32)
    query_voxels = o3c.Tensor(query_voxels,dtype=o3c.int32,device=voxel_map['voxels'].device)
    buf_indices, masks = voxel_map['voxels'].find(query_voxels)
    buf_indices = buf_indices[masks].to(o3c.int64)
    
    print('{}/{} points find semantic labels'.format(len(buf_indices),N))
    voxel_semantic = voxel_map['voxels'].value_tensor(0)[buf_indices]
    voxel_instance = voxel_map['voxels'].value_tensor(1)[buf_indices]

def time_analysis(frame_time_table, object_time_table, output_folder):
    mean_frame_time = frame_time_table[:,1:].sum(axis=0)/frame_time_table[:,0].sum()
    mean_object_time = object_time_table[:,1:].sum(axis=0)/object_time_table[:,0].sum()
    print('mean frame time(msec): ', 1000*mean_frame_time)
    print('mean object time(msec): ',1000*mean_object_time)
    with open(os.path.join(output_folder,'time.txt'),'w') as f:
        f.write('mean frame time(msec): {}\n'.format(1000*mean_frame_time))
        f.write('mean object time(msec): {}\n'.format(1000*mean_object_time))
        f.close()

def process_scene(args):
    map_root, pred_root, scene_name, predictor,eval_folder, viz_folder = args
    map_folder = os.path.join(map_root,scene_name)
    pred_folder = os.path.join(pred_root,scene_name)
    
    mdevice = o3c.Device('cpu:0')
    if 'ScanNet' in map_root:
        MAP_POSFIX = 'mesh_o3d_256' 
        # MAP_POSFIX = 'scene0064_00_vh_clean_2' 
        SEGFILE_POSFIX = '0.010000.segs.json'
        map_dir = os.path.join(map_folder,'{}.ply'.format(MAP_POSFIX))
        segfile_dir = os.path.join(map_folder,'{}.{}'.format(MAP_POSFIX,SEGFILE_POSFIX))
        MIN_VOXEL_WEIGTH = 2
        VX_RESOLUTION = 256.0
        SMALL_INSTANCE = 200
        MIN_GEOMETRY = 100
        SEGMENT_IOU = 0.1
        NMS_IOU = 0.2
        NMS_SIMILARITY=0.1
        merge_semantic_classes = []
        # merge_semantic_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 'counter',
        #                 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

    elif 'tum' in map_root:
        MAP_POSFIX = 'mesh_o3d256_ds.ply'
        SEGFILE_POSFIX = 'mesh_o3d256_ds.0.010000.segs.json'
        map_dir = os.path.join(map_folder,'{}'.format(MAP_POSFIX))
        segfile_dir = os.path.join(map_folder,'{}'.format(SEGFILE_POSFIX))
        MIN_VIEW_COUNT = 4
        MIN_FOREGROUND = 0.4
        SMALL_INSTANCE = 500
        MIN_GEOMETRY = 200
        NMS_IOU = 0.1
        NMS_SIMILARITY=0.2
        merge_semantic_classes = ['computer monitor','earth','teddy','table','desk','chair']
    elif 'scenenn' in map_root:
        MAP_POSFIX = 'mesh_o3d256.ply'
        SEGFILE_POSFIX = 'mesh_o3d256.0.010000.segs.json'
        map_dir = os.path.join(map_folder,'{}'.format(MAP_POSFIX))
        segfile_dir = os.path.join(map_folder,'{}'.format(SEGFILE_POSFIX))
        MIN_VIEW_COUNT = 3
        MIN_FOREGROUND = 0.4
        SMALL_INSTANCE = 1000
        MIN_GEOMETRY = 200
        NMS_IOU = 0.1
        NMS_SIMILARITY=0.2
        merge_semantic_classes = ['chair','bookshelf','floor','cabinet','refridgerator','table','desk','window','curtain','shower curtain']

    pred_dir = os.path.join(pred_folder,'fusion_debug.txt')
    time_record = None
    object_time_record = None

    # return None
    print('----- Processing {} -----'.format(scene_name))
            
    # Load instance map
    instance_map = fuse_detection.ObjectMap(None,None)
    instance_map.load_semantic_names(predictor.closet_names)
    instance_map.load_dense_points(map_dir,4.0/VX_RESOLUTION)
    instance_map.load_segments(segfile_dir)
    # return None
    
    count_overcf = 0
    with open(pred_dir,'r') as f:
        for line in f.readlines():
            if 'time record' in line:
                time_record_string = line.split(':')[-1][2:-3]
                time_record = [float(ele.strip()) for ele in time_record_string.split(' ') if len(ele)>1]
                # time_record = np.array(time_record)
                print('time record: ',time_record)
            elif 'objects record' in line:
                object_time_record_string = line.split(':')[-1][2:-3]
                object_time_record = [float(ele.strip()) for ele in object_time_record_string.split(' ') if len(ele)>1]
                print('object time record: ',object_time_record)
            
            if line[0] == '#':continue
            secs = line.split(';')
            parts = secs[0].split(' ')
            mask_file = parts[0].strip()

            msg = mask_file + ': '
            id = int(mask_file.split('_')[0])
            
            new_instance = fuse_detection.Instance(id,device=mdevice)   
            new_instance.read_voxel_grid(os.path.join(pred_folder,'{}_hashmap.npz'.format(mask_file)),
                                         resolution=VX_RESOLUTION)

            openset_detections = secs[1].split(',')

            # update measurements
            for zk in openset_detections[:-1]:
                # print(det.strip())
                labels = {} # {y: score}
                label_score_pairs = zk.split('_')
                for pair in label_score_pairs[:-1]:
                    label, score = pair.split(':')
                    label = label.strip()
                    score = score.strip()
                    labels[label] = float(score)
                
                new_instance.prob_vector, flag = predictor.update_measurement(labels,new_instance.prob_vector)
                if flag: new_instance.prob_weight +=1

            if new_instance.prob_weight > 0:
                label, conf = new_instance.estimate_label()
                # print('id: {}, label: {}, conf: {:.3f}'.format(id,predictor.closet_names[label],conf))
                instance_map.insert_instance(new_instance)
            # else:
                # print('skip instance: {}'.format(id))
                # print(new_instance.prob_vector)
                    
        f.close()
        # print('{} overconfidence instances'.format(count_overcf))
    
    # return None
    print('Load {} instances done'.format(len(instance_map.instance_map)))
    import time 
    
    t_start = time.time()
    # Merge over-segmentation
    instance_map.fuse_instance_segments(merge_types=merge_semantic_classes,
                                        min_voxel_weight=MIN_VOXEL_WEIGTH,min_segments=MIN_GEOMETRY, segment_iou=SEGMENT_IOU)
    instance_map.merge_conflict_instances(nms_iou=NMS_IOU, nms_similarity=NMS_SIMILARITY)
    instance_map.remove_small_instances(min_points=SMALL_INSTANCE)
    t_end = time.time()
    # print('Merging time: {}s'.format(t_end-t_start))
    
    instance_map.update_instances_voxel_grid(device=mdevice)
    instance_map.update_volume_points()
    instance_map.update_semantic_queries()
    instance_map.verify_curtains()
    instance_map.merge_background()

    # Export
    if 'ScanNet' in map_root:
        print('Saving evaluation results')
        scannet_label_dir = os.path.join(map_folder,'{}_{}'.format(scene_name,'vh_clean_2.ply'))
        label_pcd = o3d.io.read_point_cloud(scannet_label_dir)
        label_points = np.asarray(label_pcd.points,dtype=np.float32)
        instance_map.save_scannet_results(os.path.join(eval_folder,scene_name),label_points)
    
    # return None

    # Save visualization
    import render_result
    unlabel_point_color = 'remove' # remove or black
    if viz_folder is not None:
        dense_points, dense_labels = instance_map.extract_object_map(external_points=None,extract_from_dense_points=False,min_weight=MIN_VOXEL_WEIGTH)
        semantic_colors, instance_colors = render_result.generate_colors(dense_labels.astype(np.int64))
        
        viz_pcd = o3d.geometry.PointCloud()
        if unlabel_point_color=='remove':
            valid = instance_colors.sum(axis=1)>0
            viz_pcd.points = o3d.utility.Vector3dVector(dense_points[valid])
            semantic_colors = semantic_colors[valid]
            instance_colors = instance_colors[valid]
        else: # black
            viz_pcd.points = o3d.utility.Vector3dVector(dense_points)
        
        viz_pcd.colors = o3d.utility.Vector3dVector(semantic_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),viz_pcd)
        viz_pcd.colors = o3d.utility.Vector3dVector(instance_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),viz_pcd)
    
    print('finished {}'.format(scene_name))
    return (time_record, object_time_record)
                
if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data2/ScanNet')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--split_file',type=str, default='val')
    parser.add_argument('--debug_folder', type=str, help='folder in the the debug directory', default='baseline')
    parser.add_argument('--prior_model',type=str, default='bayesian')
    parser.add_argument('--measurement_dir', type=str, default='noaugment')
    # parser.add_argument('--fuse_all_tokens', action='store_true', help='fuse all tokens')
    
    opt = parser.parse_args()
    
    # root_folder = '/data2/ScanNet'
    split = opt.split #'val'
    pred_root_dir = os.path.join(opt.dataroot,'debug',opt.debug_folder) # '/data2/ScanNet/debug/baseline' # bayesian_forward
    eval_folder = os.path.join(opt.dataroot,'eval','{}_refined'.format(opt.debug_folder)) #'/data2/ScanNet/eval/'+METHOD_NAME
    viz_folder = os.path.join(opt.dataroot,'output','{}_refined'.format(opt.debug_folder)) #'/data2/ScanNet/output/'+METHOD_NAME
    # semantic_eval_folder = os.path.join(opt.dataroot,'eval','{}_semantic'.format(opt.debug_folder)) #'/data2/ScanNet/eval/'+METHOD_NAME

    # if opt.prior_model=='bayesian':
    prior_model = os.path.join(opt.measurement_dir,opt.prior_model)

    scans = fuse_detection.read_scans(os.path.join(opt.dataroot,'splits','{}.txt'.format(opt.split_file)))
    # scans = ['scene0616_00']
    
    if viz_folder is not None:
        if os.path.exists(viz_folder) is False:
            os.makedirs(viz_folder)

    if os.path.exists(eval_folder) is False:
        os.makedirs(eval_folder)
    
    label_predictor = fuse_detection.LabelFusion(prior_model, fusion_method='bayesian')
    map_root_dir = os.path.join(opt.dataroot,split)
    # exit(0)
    
    valid_scans = []
    frame_time_table = [] # [frame_number,t0,t1,t2]
    object_time_table = [] # [aggregate_object_number,t0,t1,t2]
    for scan in scans:
        fuse_file = os.path.join(pred_root_dir,scan,'fusion_debug.txt')
        output_file = os.path.join(eval_folder,scan,'predinfo.txt')
        # if os.path.exists(output_file):continue
        if os.path.exists(fuse_file):
            valid_scans.append(scan)
            (time_record, object_time_record) = process_scene((map_root_dir,pred_root_dir,scan,label_predictor,eval_folder,viz_folder))
            
            if time_record is not None:
                frame_time_table.append(np.array(time_record))
            if object_time_table is not None:
                object_time_table.append(np.array(object_time_record))
            # break
        else:
            print('skip scan: {}'.format(scan))
            break
    
    frame_time_table = np.array(frame_time_table)
    object_time_table = np.array(object_time_table)
    time_analysis(frame_time_table,object_time_table,pred_root_dir)
    
    print('processed {} scans'.format(len(valid_scans)))
    exit(0)


    import multiprocessing as mp
    p = mp.Pool(processes=32)
    p.map(process_scene, [(map_root_dir, pred_root_dir,scan,label_predictor,eval_folder,viz_folder) for scan in valid_scans])
    p.close()
    p.join()
