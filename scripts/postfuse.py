import os 
import fuse_detection
import numpy as np
import open3d as o3d

SEMANTIC_NAMES = fuse_detection.SEMANTIC_NAMES
SEMANTIC_IDX = fuse_detection.SEMANTIC_IDX

def process_scene(args):
    map_root, pred_root, scene_name, predictor,eval_folder, viz_folder = args
    map_folder = os.path.join(map_root,scene_name)
    pred_folder = os.path.join(pred_root,scene_name)
    map_dir = os.path.join(map_folder,'{}_vh_clean_2.ply'.format(scene_name))
    pred_dir = os.path.join(pred_folder,'fusion_debug.txt')

    EXIST_THRESHOD = 0.2
    print('----- Processing {} -----'.format(scene_name))
        
    # Load input map
    # tmp_dir = os.path.join('/data2/ScanNet/output/input','{}.ply'.format(scene_name))
    pcd = o3d.io.read_point_cloud(map_dir)
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)
    N = points.shape[0]
    # o3d.io.write_point_cloud(tmp_dir,pcd)
    
    # Load instance map
    instance_map = fuse_detection.ObjectMap(points,colors)
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
            # print(openset_detections)
            # os_labels = {}
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
            best_label, conf = new_instance.estimate_label()
            top_labels, confs = new_instance.estimate_top_labels(3)
            
            instance_map.insert_instance(new_instance)
            # msg += str(weight)
            # msg += '{},{:.3f},{:.1f}'.format(SEMANTIC_NAMES[best_label],conf,new_instance.prob_weight)
            for k in range(len(top_labels)):
                msg += '{}({:.3f}),'.format(SEMANTIC_NAMES[top_labels[k]],confs[k])
            
            print(msg)       
        f.close()
    print('Load instance map done')
    # exit(0)
    
    # Merge over-segmentation
    # instance_map.extract_all_centroids()
    # instance_map.merge_instances()
    # instance_map.filter_conflict_objects()
    
    # Export
    MIN_POSITIVE_COUNT = 2
    
    instance_map.save_scannet_results(eval_folder,scene_name,min_pos=MIN_POSITIVE_COUNT)

    # Save visualization
    import render_result
    if viz_folder is not None:
        instance_labels = instance_map.extract_object_map(min_pos=MIN_POSITIVE_COUNT)
        semantic_colors, instance_colors = render_result.generate_colors(instance_labels.astype(np.int64))
        pcd.colors = o3d.utility.Vector3dVector(semantic_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),pcd)
        pcd.colors = o3d.utility.Vector3dVector(instance_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),pcd)
                
if __name__ =='__main__':
    USE_BASELINE = True
    root_folder = '/data2/ScanNet'
    split = 'val'
    pred_root_dir = '/data2/ScanNet/debug/bayesian'
    METHOD_NAME = 'baseline'
    eval_folder = '/data2/ScanNet/eval/'+METHOD_NAME
    output_folder = '/data2/ScanNet/output/'+METHOD_NAME
    # output_folder = None
    prior_model = '/home/cliuci/PointGroup/benchmark/temporal_output/prompt_likelihood.npy'

    scans = fuse_detection.read_scans(os.path.join(root_folder,'splits','val_clean.txt'))
    # scans = ['scene0633_01']
    
    if output_folder is not None:
        if os.path.exists(output_folder) is False:
            os.makedirs(output_folder)

    if os.path.exists(eval_folder) is False:
        os.makedirs(eval_folder)
    
    label_predictor = fuse_detection.LabelFusion(prior_model, use_baseline=USE_BASELINE, propogate_method='mean')
    map_root_dir = os.path.join(root_folder,split)
    # exit(0)
    
    valid_scans = []
    for scan in scans:
        fuse_file = os.path.join(pred_root_dir,scan,'fusion_debug.txt')
        if os.path.exists(fuse_file):
            valid_scans.append(scan)
            # process_scene((map_root_dir, pred_root_dir,scan,label_predictor, eval_folder,output_folder))
            
    # exit(0)
    
    print('processing {} scans'.format(len(valid_scans)))

    import multiprocessing as mp
    p = mp.Pool(processes=32)
    p.map(process_scene, [(map_root_dir, pred_root_dir,scan,label_predictor,eval_folder,output_folder) for scan in valid_scans])
    p.close()
    p.join()
