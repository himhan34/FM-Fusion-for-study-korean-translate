import os 
import fuse_detection
import numpy as np
import open3d as o3d
import argparse

SEMANTIC_NAMES = fuse_detection.SEMANTIC_NAMES
SEMANTIC_IDX = fuse_detection.SEMANTIC_IDX

def process_scene(args):
    map_root, pred_root, scene_name, predictor,eval_folder, viz_folder = args
    map_folder = os.path.join(map_root,scene_name)
    pred_folder = os.path.join(pred_root,scene_name)

    if 'scannet' in map_root:
        MAP_POSFIX = 'vh_clean_2.ply'
        SEGFILE_POSFIX = 'vh_clean_2.0.010000.segs.json'
        map_dir = os.path.join(map_folder,'{}_{}'.format(scene_name,MAP_POSFIX))
        segfile_dir = os.path.join(map_folder,'{}_{}'.format(scene_name,SEGFILE_POSFIX))
        MIN_VIEW_COUNT = 3
        MIN_FOREGROUND = 0.4
        SMALL_INSTANCE = 200
        MIN_GEOMETRY = 100
        NMS_IOU = 0.1
        NMS_SIMILARITY=0.2
        merge_semantic_classes = SEMANTIC_NAMES
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
    

    print('----- Processing {} -----'.format(scene_name))
        
    # Load input map
    pcd = o3d.io.read_point_cloud(map_dir)
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)
    N = points.shape[0]
    # o3d.io.write_point_cloud(tmp_dir,pcd)
    
    # Load instance map
    instance_map = fuse_detection.ObjectMap(points,colors)
    instance_map.load_semantic_names(predictor.closet_names)
    instance_map.load_segments(segfile_dir)
    
    # exit(0)
    
    with open(pred_dir,'r') as f:
        for line in f.readlines():
            if line[0] == '#':continue
            secs = line.split(';')
            parts = secs[0].split(' ')
            
            mask_file = parts[0].strip()
            label_id = parts[1].strip()
            label_conf = parts[2].strip()
            pos_observed = float(parts[3].strip())
            neg_observed = float(parts[4].strip())
            # exit_conf = float(parts[3].strip())
            
            msg = mask_file + ': '
            # if exit_conf<EXIST_THRESHOD:
            #     msg += ' removed'
                # print(msg)
                # continue
            
            id = int(mask_file.split('_')[0])
            pos_pts = np.loadtxt(os.path.join(pred_folder,'{}_pos.txt'.format(mask_file))).astype(np.int32)
            negative_file = os.path.join(pred_folder,'{}_neg.txt'.format(mask_file))

            
            new_instance = fuse_detection.Instance(id)   
            new_instance.pos_observed = pos_observed
            new_instance.neg_observed = neg_observed
            # new_instance.points = np.where(mask==1)[0]
            new_instance.points = pos_pts
            if os.path.exists(negative_file):
                new_instance.negative = np.loadtxt(negative_file).astype(np.int32)
                # assert new_instance.negative.shape[0] >0, ' {} Negative points should not be empty'.format(scene_name)
                            
            openset_detections = secs[1].split(',')

            for det in openset_detections[:-1]:
                # print(det.strip())
                labels = {}
                label_score_pairs = det.split('_')
                for pair in label_score_pairs[:-1]:
                    label, score = pair.split(':')
                    label = label.strip()
                    score = score.strip()
                    labels[label] = float(score)
                new_instance.os_labels.append(labels)
                
                # os_name = parts[0].strip()
                # os_conf = [float(part) for part in parts[1:]] # list of scores
                # if os_name not in new_instance.os_labels:
                #     new_instance.os_labels[os_name] = os_conf
                # else:
                #     new_instance.os_labels[os_name] += os_conf
                # print('{}:{}'.format(os_name,os_conf))

            prob_vector, weight = predictor.estimate_batch_prob(new_instance.os_labels)
            new_instance.prob_weight = weight
            new_instance.prob_vector = prob_vector
            # best_label, conf = new_instance.estimate_label()
            
            instance_map.insert_instance(new_instance)
            
            continue
            # msg += str(weight)

            
            print(msg)       
        f.close()
    print('Load instance map done')
    # exit(0)
    import time 
    
    t_start = time.time()
    # Merge over-segmentation
    instance_map.update_result_points(min_fg=MIN_FOREGROUND,min_viewed=MIN_VIEW_COUNT)
    instance_map.fuse_instance_segments(merge_types=merge_semantic_classes,min_segments=MIN_GEOMETRY)
    instance_map.merge_conflict_instances(nms_iou=NMS_IOU, nms_similarity=NMS_SIMILARITY)
    instance_map.remove_small_instances(min_points=SMALL_INSTANCE)
    t_end = time.time()
    print('Merging time: {}s'.format(t_end-t_start))
    
    # Export
    if 'scannet' in map_root:
        instance_map.save_scannet_results(eval_folder,scene_name,merged_results=True)
    
    # return None

    # Save visualization
    import render_result
    unlabel_point_color = 'remove' # raw, remove or black
    if viz_folder is not None:
        instance_labels = instance_map.extract_object_map(merged_results=True)
        semantic_colors, instance_colors = render_result.generate_colors(instance_labels.astype(np.int64))
        
        viz_pcd = o3d.geometry.PointCloud()
        if unlabel_point_color=='remove':
            valid = instance_colors.sum(axis=1)>0

            # print('{}/{} valid'.format(valid.sum(),N))
            # print(instance_colors.shape)
            viz_pcd.points = o3d.utility.Vector3dVector(points[valid])
            semantic_colors = semantic_colors[valid]
            instance_colors = instance_colors[valid]
        elif unlabel_point_color=='raw':
            viz_pcd.points = o3d.utility.Vector3dVector(points)
            black_points = instance_colors.sum(axis=1)<1e-6
            semantic_colors[black_points] = 255*colors[black_points]
            instance_colors[black_points] = 255*colors[black_points]
            print('color {} points in rgb'.format(black_points.sum()))
        else: # black
            viz_pcd.points = o3d.utility.Vector3dVector(points)
        
        viz_pcd.colors = o3d.utility.Vector3dVector(semantic_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),viz_pcd)
        viz_pcd.colors = o3d.utility.Vector3dVector(instance_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),viz_pcd)
                
if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data2/ScanNet')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--split_file',type=str, default='val')
    parser.add_argument('--debug_folder', type=str, help='folder in the the debug directory', default='baseline')
    parser.add_argument('--prior_model',type=str, default='bayesian_forward')
    parser.add_argument('--measurement_dir', type=str, default='noaugment')
    parser.add_argument('--fuse_all_tokens', action='store_true', help='fuse all tokens')
    
    opt = parser.parse_args()
    
    # root_folder = '/data2/ScanNet'
    split = opt.split #'val'
    pred_root_dir = os.path.join(opt.dataroot,'debug',opt.debug_folder) # '/data2/ScanNet/debug/baseline' # bayesian_forward
    # METHOD_NAME = 'bayesian_forward'
    eval_folder = os.path.join(opt.dataroot,'eval','{}_offline'.format(opt.prior_model)) #'/data2/ScanNet/eval/'+METHOD_NAME
    output_folder = os.path.join(opt.dataroot,'output','{}_offline'.format(opt.prior_model)) #'/data2/ScanNet/output/'+METHOD_NAME
    # FUSE_ALL_TOKENS = True

    if opt.prior_model=='bayesian':
        prior_model = os.path.join(opt.measurement_dir,'bayesian')
    elif opt.prior_model == 'hardcode':
        prior_model = os.path.join(opt.measurement_dir,'hardcode')
    else:
        raise NotImplementedError

    scans = fuse_detection.read_scans(os.path.join(opt.dataroot,'splits','{}.txt'.format(opt.split_file)))
    scans = ['255']
    
    if output_folder is not None:
        if os.path.exists(output_folder) is False:
            os.makedirs(output_folder)

    if os.path.exists(eval_folder) is False:
        os.makedirs(eval_folder)
    
    label_predictor = fuse_detection.LabelFusion(prior_model, fuse_all_tokens=opt.fuse_all_tokens)
    map_root_dir = os.path.join(opt.dataroot,split)
    # exit(0)
    
    valid_scans = []
    for scan in scans:
        fuse_file = os.path.join(pred_root_dir,scan,'fusion_debug.txt')
        if os.path.exists(fuse_file):
            valid_scans.append(scan)
            # process_scene((map_root_dir,pred_root_dir,scan,label_predictor,eval_folder,output_folder))
            # break
    
    print('processing {} scans'.format(len(valid_scans)))
    # exit(0)


    import multiprocessing as mp
    p = mp.Pool(processes=32)
    p.map(process_scene, [(map_root_dir, pred_root_dir,scan,label_predictor,eval_folder,output_folder) for scan in valid_scans])
    p.close()
    p.join()
