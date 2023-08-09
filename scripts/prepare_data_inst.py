'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import os
import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse
import math
import cv2, open3d as o3d
import scannet_util
import pandas as pd

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

label_map_nyu40 = scannet_util.read_label_mapping('/data2/ScanNet/scannetv2-labels.combined.tsv', 
                                                  label_from='id', label_to='nyu40id')

def create_image_features(dataset,scene_dir,points_color,output_dir,rgb_posix,depth_posix,visualize=False,folder_name='preprocessed'):
    '''
    Input files:
        points_uv.npy: [frame_id, point_id, u, v, d, conf]
        frames.txt: {frame_name: num_points_uv}
        # intrinsic.txt: [K_rgb, K_depth,rgb_dim,depth_dim]
    '''
    import project_util
    save_aligned_features = False
    
    # scene_dir = os.path.join(split_dir,scene_name)
    scene_name = os.path.basename(scene_dir)
    preprocess_dir = os.path.join(scene_dir,folder_name)
    frames, num_points, num_points_uv = project_util.read_frames(os.path.join(preprocess_dir,'frames.txt'))
    summary_points_uv = np.load(os.path.join(preprocess_dir,'points_uv.npy')) # [frame_id, point_id, u, v, d, conf]
    assert num_points==points_color.shape[0], '{} has points number mismatch'.format(scene_name)
    assert summary_points_uv.shape[0] == num_points_uv
    assert summary_points_uv.shape[1] == 6
    assert len(frames) == summary_points_uv[:,0].max()+1, '{} has frames inconsistent! {} !={}'.format(scene_name, 
                                                                                                len(frames),summary_points_uv[:,0].max()+1)
    # print('color range [{},{}]'.format(points_color.min(),points_color.max()))

    # K_raw_rgb, K_depth,raw_rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,'intrinsic'))
    if dataset=='scannetv2' or dataset =='scanneto3d' or dataset=='scannet005':
        rgb_dim = np.array([480,640])
    elif dataset=='rscan':
        rgb_dim = np.array([540,960])
    else: raise NotImplementedError
    # assert np.sum(rgb_dim-depth_dim) ==0, 'rgb and depth dimension mismatch'
    # K_rgb = project_util.adjust_intrinsic(K_raw_rgb,raw_rgb_dim,rgb_dim)
    assert summary_points_uv[:,2].max()<=rgb_dim[1] and summary_points_uv[:,2].min()>=0, 'u range error'
    assert summary_points_uv[:,3].max()<=rgb_dim[0] and summary_points_uv[:,3].min()>=0, 'v range error'

    # Export
    torch.save((summary_points_uv,frames, num_points), os.path.join(output_dir,'points_uv.pth'))
    if save_aligned_features:
        features = torch.load(os.path.join(preprocess_dir,'features.pth'))
        torch.save(features, os.path.join(output_dir,'features.pth'))
    print('points uv saved at {}'.format(output_dir))
    color_folder = os.path.join(output_dir,'color')
    label_folder = os.path.join(output_dir,'label')
    
    if os.path.exists(color_folder)==False: os.mkdir(color_folder)
    if os.path.exists(label_folder)==False: os.mkdir(label_folder)
    
    # 
    points_color = (points_color +1.0) * 127.5
    offset = 0
    frame_id = 0
    if visualize:
        colorbar = np.invert(np.arange(255).astype(np.uint8))
        colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET).squeeze() #255x3
    
    for frame_name, points_uv_count in frames.items():
        rgb_dir = os.path.join(scene_dir,'color',frame_name+rgb_posix)
        label_dir = os.path.join(scene_dir,'label',frame_name.split('-')[-1].lstrip('0')+'.png')
        depth_dir = os.path.join(scene_dir,'depth',frame_name+depth_posix)
        
        # print('->{} '.format(frame_name))
        rgbimg = cv2.imread(rgb_dir,cv2.IMREAD_UNCHANGED)
        # if rgbimg is None:
        #     print('[WARNING] rgb image {} at {} not found'.format(frame_name,scene_name))
        #     continue
        assert rgbimg is not None, 'rgb image {} at {} not found'.format(frame_name,scene_name)
        if rgbimg.shape[0]!=rgb_dim[0] or rgbimg.shape[1]!=rgb_dim[1]:
            rgbimg = cv2.resize(rgbimg,(rgb_dim[1],rgb_dim[0]),interpolation=cv2.INTER_LINEAR)
        labelimg = cv2.imread(label_dir,cv2.IMREAD_UNCHANGED)
        assert labelimg is not None, 'label image {} at {} not found'.format(frame_name,scene_name)
        labelimg = scannet_util.map_label_image(labelimg,label_map_nyu40)
        if labelimg.shape[0]!=rgb_dim[0] or labelimg.shape[1]!=rgb_dim[1]:
            labelimg = cv2.resize(labelimg,(rgb_dim[1],rgb_dim[0]),interpolation=cv2.INTER_NEAREST)
        assert labelimg.max()<=40 and labelimg.min()>=0, 'label range error'
        cv2.imwrite(os.path.join(color_folder,str(frame_id).zfill(4)+'.jpg'),rgbimg)
        cv2.imwrite(os.path.join(label_folder,str(frame_id).zfill(4)+'.png'),labelimg)
        
        points_uv_segment = summary_points_uv[summary_points_uv[:,0]==frame_id] #summary_points_uv[offset:offset+points_uv_count,:]
        assert points_uv_segment.shape[0]>0, '{} contains frame {} with no points'.format(scene_name,frame_name)
        frame_id +=1
        
        ######## Visualization #########
        if visualize==False:continue  
                
        # re-organize points_uv [frame_id,pt_id,u,v,d,conf]
        points_uv = np.zeros((num_points,4),np.float32)-1.0
        points_uv[points_uv_segment[:,1].astype(np.int32)] = points_uv_segment[:,2:6]
        mask = points_uv[:,2]>0.0
        assert mask.sum()>1 , '{} in {} has no valid points'.format(frame_name, scene_name)

        debug_img = rgbimg.copy()
        prj_rgb = np.zeros((rgb_dim[0],rgb_dim[1],3),np.uint8)
        conf_map = np.zeros((rgb_dim[0],rgb_dim[1],3),np.uint8)
        
        valid_point = np.floor(points_uv[mask,:2]).astype(np.int32)
        valid_corlor = points_color[mask].astype(np.float64)
        valid_conf = points_uv[mask,3].astype(np.float64)
        
        M = np.count_nonzero(mask)
        # if M>0: print('Mean depth {}'.format(valid_point[:,2].mean()))
        
        for j in range(M):
            u = valid_point[j,0]
            v = valid_point[j,1]
            # d = valid_point[j,2]
            viz_conf = np.float64(colorbar[math.floor(valid_conf[j]*255)])

            cv2.circle(debug_img,(u,v),2,[valid_corlor[j][2],valid_corlor[j][1],valid_corlor[j][0]],-1)
            cv2.circle(prj_rgb,(u,v),5,[valid_corlor[j][2],valid_corlor[j][1],valid_corlor[j][0]],-1)
            cv2.circle(conf_map,(u,v),5,[viz_conf[2],viz_conf[1],viz_conf[0]],-1)
        debug_img = np.concatenate([debug_img,prj_rgb,conf_map],axis=1)
        viz_dir = os.path.join(output_dir,'viz')
        if os.path.exists(viz_dir)==False: os.mkdir(viz_dir)
        cv2.imwrite(os.path.join(viz_dir,frame_name+'.png'),debug_img)
        
        offset +=points_uv_count

def process_3rscan_scene(args):
    '''
    Input files:
        - refined.ply, point cloud converted from .obj
        - annotation.npy, (N,5), [x,y,z,semantic_label,instance_label]
    '''
    
    out_posfix = 'inst_nostuff.pth'
    scene_dir, output_dir, img_feat, visualization= args
    scene_name = os.path.basename(scene_dir)
    
    scene_output = os.path.join(output_dir,scene_name)
    # output_pth_dir = os.path.join(output_dir, scene_name+'.pth')
    ply_filename = os.path.join(scene_dir, 'refined.ply')
    annotation = os.path.join(scene_dir, 'annotation.npy')
    max_dist_threshold = 0.05 
    if os.path.exists(scene_output)==False: os.mkdir(scene_output)

    print('processing {}'.format(scene_name))
    pcd = o3d.io.read_point_cloud(ply_filename)
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) *2.0 -1.0
    
    N = coords.shape[0]
    semantic_labels = (-100) * np.ones(N)
    instance_labels = (-100) * np.ones(N)

    # otherfurniture, bookshelf
    ignored_types = [40,39,22,10]
    
    if os.path.exists(annotation):
        mat = np.load(annotation)
        annotated_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mat[:,:3]))
        annotated_tree = o3d.geometry.KDTreeFlann(annotated_pcd)
        print(annotated_pcd)
        # np.asarray(annotated_pcd.points) = o3d.utility.Vector3dVector(mat[:,:3])
        # print('{},{}'.format(np.count_nonzero(mat[:,3]>0),np.count_nonzero(mat[:,4]>0)))
        for i in range(N):
            [k,idx,_] = annotated_tree.search_radius_vector_3d(pcd.points[i],max_dist_threshold)
            if k>0:
                semantic_labels[i] = int(mat[idx[0],3])
                instance_labels[i] = int(mat[idx[0],4])
                if semantic_labels[i] in ignored_types:
                    semantic_labels[i] = -100
                    instance_labels[i] = -100
                    
        semantic_labels[semantic_labels<0] = 0
        instance_labels[semantic_labels<0] = 0
        print('{}/{} semantic pts, {}/{} instance pts'.format(np.count_nonzero(semantic_labels>0),N,
                                                        np.count_nonzero(instance_labels>0),N))
        # semantic_points = colorize_pointcloud_label(coords,semantic_labels)        
        # pcd.colors = o3d.utility.Vector3dVector(semantic_points[:,3:] / 255.0)
        # o3d.visualization.draw_geometries([pcd],'annotation')
    else:
        print('semantic and instance labels are all set to zero')

    
    coords = coords - coords.mean(0)
    semantic_labels = remapper[semantic_labels.astype(np.int32)]
    assert colors.min()>=-1 and colors.max()<=1, 'color range should be [-1,1]'
    assert semantic_labels.max()<=19, 'semantic label range should be [0,20)'
    
    torch.save((np.float32(coords),np.float32(colors),semantic_labels,instance_labels),os.path.join(scene_output,out_posfix))
    print('Saved to {}'.format(scene_output))

    if img_feat:
        create_image_features('rscan',scene_dir, colors, scene_output, '.color.jpg','.depth.pgm', visualization)
    print('processed {}'.format(scene_name))
    # return True

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')

def read_scannet_segjson(dir):
    '''
        segid_to_pointid: dict, key: segid, value: list of pointid
    '''
    segid_to_pointid = {}

    with open(dir) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    return segid_to_pointid

def read_scannet_aggjson(dir, ignore_labels = ['wall','floor','remove','unknown','ceiling']):
    '''
    Input:
        - dir: str, path to agg json file
    Output:
        - instance_segids: list of list, each list contains segids of one instance
        - labels: list of string, each string is the label of one instance
    '''
    instance_segids = []
    labels = []
    # IGNORE_LABELS = ['wall','floor','remove','unknown','ceiling']
    scene_name = dir.split('/')[-2]
    
    print('--reading {}-'.format(dir))
    with open(dir) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            # nyuname = name_remapper[x['label']]
            if x['label'] in ignore_labels:
                continue
            # if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[x['label']] != 'floor':
            instance_segids.append(x['segments'])
            labels.append(x['label'])
            # assert (x['label'] in scannet_util.g_raw2scannetv2.keys())
    if(scene_name == 'scene0217_00' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check), '{} instance_segids is not unique. {}'.format(scene_name, dir)
    
    return instance_segids, labels

def create_instance_labels(sem_labels, segid_to_pointid, instance_segids,instance_labels_name, min_consist_rate = 0.9):  
    '''
    Output:
        - instance_labels: np.array, shape: (N,), invalid is set to -100
    '''  
    
    K = len(instance_segids)
    # instance_map = np.zeros([sem_labels.shape[0],K])
    instance_labels = -100 * np.ones(sem_labels.shape[0],dtype=np.int32)
    
    for i in range(K):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        # instance_map[pointids,i] += 1
    
    # instance_labels = np.argmax(instance_map,axis=1)
        
    # check
    for i in range(K):
        pointids = np.where(instance_labels == i)[0]
        sem_candidates = np.unique(sem_labels[pointids])
        if len(sem_candidates) ==1: 
            continue
            print('instance type {}-{} has only one semantic label {}'.format(
                instance_labels_name[i],sem_candidates[0],sem_candidates[0]))
            
        else:
            hist,bin_edges = np.histogram(sem_labels[pointids],
                                          bins=np.concatenate((np.array([-100]),np.arange(20)),axis=0))
            category = int(np.argmax(hist))
            label = int(bin_edges[category])
            category_rate = hist[category] / len(pointids)
            if category_rate<min_consist_rate:
                instance_labels[pointids] = -100 # invalid     
                print('filter instance {}-{} due to low category rate {}'.format(
                    instance_labels_name[i],label,category_rate))
            # print('instance type {}_{}'.format(instance_labels_name[i],label))
        
            # assert len(np.unique(sem_labels[pointids])) == 1, '{} has multiple semantic labels'.format(scene_name)
    return instance_labels

# todo: process it.
def process_matterport_scene(args):
    '''
    Input:
        - region_ply_dir: str, path to the region ply file
    '''
    
    # from matterport3d import matterport3d_util
    # MATTERPORT_ALLOWED_NYU_CLASSES = matterport3d_util.MATTERPORT_ALLOWED_NYU_CLASSES
    # MATTERPORT_CLASS_REMAP = matterport3d_util.MATTERPORT_CLASS_REMAP
    
    region_ply_dir, output_root, img_feat, visualize = args
    seg_dir = os.path.dirname(region_ply_dir)
    region_name = os.path.basename(region_ply_dir).split('.')[0]
    scene_name = region_ply_dir.split('/')[-3]
    
    output_folder = os.path.join(output_root, '{}_{}'.format(scene_name, region_name))
    if os.path.exists(output_folder)==False:os.mkdir(output_folder)
    
    plydir = os.path.join(seg_dir,'{}.ply'.format(region_name))
    fseg = os.path.join(seg_dir,'{}.fsegs.json'.format(region_name))
    semseg = os.path.join(seg_dir,'{}.semseg.json'.format(region_name))
    
    pcd_data = plyfile.PlyData().read(plydir)
    v = np.array([list(x) for x in pcd_data.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    coords = coords - coords.mean(0)
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    category_id = pcd_data['face']['category_id']
    segment_id = pcd_data['face']['segment_id']
    category_id[category_id==-1] = 0
    mapped_labels = mp_label_mapper[category_id] # nyu40 id
    valid_faces = np.logical_and(mapped_labels>=0,mapped_labels<40)
    mapped_labels[np.logical_not(valid_faces)] = 0 # filter invalid
    
    # mapped_labels[np.logical_not(np.isin(mapped_labels, MATTERPORT_ALLOWED_NYU_CLASSES))] = 0 # filter invalid
    
    remapped_face_labels = remapper[mapped_labels].astype(int) # [0,19], -100 is invalid
    valid_faces = remapped_face_labels>=0
    # print('{}/{} valid faces'.format(np.sum(valid_faces),len(valid_faces)))
    # remapped_labels = MATTERPORT_CLASS_REMAP[mapped_labels].astype(int) # [1,20], 0 is invalid
    # print(np.unique(remapped_face_labels))
    
    triangles = pcd_data['face']['vertex_indices']
    sem_labels = np.zeros(coords.shape[0], dtype=np.int32)-100
    sem_labels_hist = np.zeros((coords.shape[0], 20), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        if remapped_face_labels[row_id] >= 0:
            for i in range(3):
                sem_labels_hist[triangles[row_id][i],remapped_face_labels[row_id]] += 1

    valid_mask = np.sum(sem_labels_hist, axis=1) > 0
    # print('{} valid'.format(np.sum(valid_mask)))
    sem_labels[valid_mask] = np.argmax(sem_labels_hist, axis=1).astype(np.int32)[valid_mask]
    # print(np.unique(sem_labels))
    assert sem_labels.max() <20, 'semantic label range wrong'
    assert sem_labels.shape[0] == coords.shape[0], 'xyz and semantic labels do not match'
    
    # create instance labels
    segid_to_faceid = read_scannet_segjson(fseg)
    segid_to_pointid = {}
    for segid, faceids in segid_to_faceid.items():
        segid_to_pointid[segid] = []
        for faceid in faceids:
            for i in range(3): segid_to_pointid[segid].append(triangles[faceid][i])
    
    max_point_id = 0
    for setgid, seg_points in segid_to_pointid.items():
        max_point_id = max(max_point_id, max(seg_points))
    assert max_point_id<coords.shape[0], 'segid_to_pointid has invalid point id'
    
    instance_segids, instance_labels_name = read_scannet_aggjson(semseg)
    # print(instance_labels_name)
    
    instance_labels = create_instance_labels(sem_labels, segid_to_pointid, instance_segids,
                                             instance_labels_name, min_consist_rate=0.9)
    
    out = (coords, colors, sem_labels, instance_labels.astype(np.int32))
    torch.save(out, os.path.join(output_folder, 'inst_nostuff.pth'))
    print('{}_{}: {}/{} valid label, max label: {}, {} instances'.format(
        scene_name,region_name, valid_mask.sum(), coords.shape[0], sem_labels.max(), len(instance_segids)))

def process_scannet_scene(args):
    '''
    Saved file:
        sem_labels: (N,), [0, 19], invalid points are labeled as -100
    '''
    out_posfix = 'inst_nostuff.pth'
    scene_folder, output_root, img_feat, visualize = args
    scene_name = os.path.basename(scene_folder)
    fn = os.path.join(scene_folder,scene_name+'_vh_clean_2.ply')
    
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    # scene_name = fn.split('/')[-1][:12]
    print('processing '+scene_name+'...')

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    
    # pcd = o3d.io.read_point_cloud(fn)
    # coords = np.asarray(pcd.points,dtype=np.float32)
    # coords = coords - coords.mean(0)
    # colors = (np.asarray(pcd.colors,dtype=np.float32)* 2.0) - 1.0 
    assert colors.max()<=1 and colors.min()>=-1, 'color range should be [-1,1]'
    
    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])] # [0,19]
    assert coords.shape[0] == sem_labels.shape[0]

    segid_to_pointid = read_scannet_segjson(fn3)

    instance_segids, labels = read_scannet_aggjson(fn4)

    instance_labels = create_instance_labels(sem_labels, segid_to_pointid, instance_segids)

    output_folder = os.path.join(output_root, scene_name)
    if os.path.exists(output_folder)==False:os.mkdir(output_folder)
    torch.save((coords, colors, sem_labels, instance_labels), os.path.join(output_folder,'inst_nostuff.pth'))#fn[:-15]+'_inst_nostuff.pth')
    # print('Saving 3d data to ' + output_folder)
    
    if img_feat:
        create_image_features('scannetv2',scene_folder,colors,output_folder,'.jpg','.png',visualize,'preprocessed')

def process_custom_scannet_scene(args):
    # ply_name = 'o3d_vx_0.05.ply'
    # process_name = 'o3d_preprocessed'
    scene_folder, ply_name,process_name, output_root, img_feat, visualize = args
    scene_name = os.path.basename(scene_folder)
    print('processing '+scene_name+'...')

    plydir = os.path.join(scene_folder,ply_name)
    fn2 = os.path.join(scene_folder, scene_name+'_vh_clean_2.labels.ply')
    fn3 = os.path.join(scene_folder, scene_name+'_vh_clean_2.0.010000.segs.json')
    fn4 = os.path.join(scene_folder, scene_name+'.aggregation.json')

    # input pointcloud
    f = plyfile.PlyData().read(plydir)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3]) # - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    assert colors.max()<=1 and colors.min()>=-1, 'color range should be [-1,1]'
    
    # read annotation
    f2 = plyfile.PlyData().read(fn2)
    points_label = np.array([list(x) for x in f2.elements[0]])
    segid_to_pointid = read_scannet_segjson(fn3)
    instance_segids, labels = read_scannet_aggjson(fn4)
    
    q_xyz_label = np.ascontiguousarray(points_label[:, :3])
    q_sem_labels = remapper[np.array(f2.elements[0]['label'])]
    # print('{} valid semantic, max {}'.format((q_sem_labels>=0).sum(),q_sem_labels.max()))
    q_instance_labels = create_instance_labels(q_sem_labels,segid_to_pointid,instance_segids)
    q_instance_labels = q_instance_labels.astype(np.float64)
    assert q_xyz_label.shape[0] == q_sem_labels.shape[0], 'annotated point number not consistent!'
    # print('{}({})/{} valid semantic(instance) annotated points'.format((q_sem_labels>=0).sum(),(q_instance_labels>=0).sum(),q_xyz_label.shape[0]))
    # print('{} instances'.format(np.unique(q_instance_labels).shape[0]-1))
    
    # 
    N = coords.shape[0]
    pcd_labels = o3d.geometry.PointCloud()
    pcd_labels.points = o3d.utility.Vector3dVector(q_xyz_label)
    annotated_tree = o3d.geometry.KDTreeFlann(pcd_labels)
    sem_labels = np.zeros((N,),dtype=np.float64)-100.0
    instance_labels = np.zeros((N,),dtype=np.float64)-100.0
    max_dist = 0.1
    
    counter = 0
    idx = []
    label_indices = []
    for i in range(N):
        [k, idx_, _]= annotated_tree.search_radius_vector_3d(coords[i], max_dist)
        if k>0:
            counter +=1
            idx.append(i)
            label_indices.append(idx_[0])

    idx = np.array(idx)
    label_indices = np.array(label_indices)
    sem_labels[idx] = q_sem_labels[label_indices]
    instance_labels[idx] = q_instance_labels[label_indices]
    
    print('[{}] {} points find annotation, {}({})/{} points are semantic (instance) annotated'.format(
        scene_name,counter,np.count_nonzero(sem_labels>=0),(instance_labels>=0).sum(),N))
    
    # Export
    coords = coords - coords.mean(0) # center
    output_folder = os.path.join(output_root, scene_name)
    if os.path.exists(output_folder)==False:os.mkdir(output_folder)
    torch.save((coords.astype(np.float32), colors.astype(np.float32), sem_labels, instance_labels), os.path.join(output_folder,'inst_nostuff.pth'))

    if img_feat:
        create_image_features('scannet005',scene_folder,colors,output_folder,'.jpg','.png',visualize,process_name)

def process_manager(args):
    dataset, scene_folder, output_dir, img_feat, visualization = args  
    
    if dataset == 'scannetv2':
        process_scannet_scene(args[1:])
    elif dataset == 'scanneto3d':
        ply_name = 'o3d_vx_0.05.ply'
        process_name = 'o3d_preprocessed'
        args = scene_folder, ply_name,process_name, output_dir, img_feat, visualization
        process_custom_scannet_scene(args)
    elif dataset == 'scannet005':
        ply_name = 'vh_clean_2_vx_0.05.ply'
        process_name = 'preprocessed_005'
        args = scene_folder, ply_name,process_name, output_dir, img_feat, visualization
        process_custom_scannet_scene(args)
    elif dataset == 'rscan':
        process_3rscan_scene(args[1:])
    elif dataset =='matterport3d': #todo
        region_ply_file = scene_folder
        args = region_ply_file, output_dir, img_feat, visualization
        process_matterport_scene(args)
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--dataset', help='dataset name {rscan, scannetv2, scanneto3d}', default='scannetv2')
    parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
    parser.add_argument('--img_feat',help='Generate image features',action='store_true')
    parser.add_argument('--visualize',help='Visualize the projected points',action='store_true')
    parser.add_argument('--output_folder',help='Output folder that store the processed data')
    parser.add_argument('--scene_id', type=str, required=False, help='scene id')
    parser.add_argument('--num_process', type=int, required=False, help='number of process')
    # parser.add_argument('--dataset', help='dataset name {scannet, rscan}', default='scannetv2')
    opt = parser.parse_args()

    split = opt.data_split
    split_dir = os.path.join(opt.data_root, split)
    
    # if opt.data_root.find('ScanNet')!=-1:
    #     opt.dataset = 'scannetv2'
    # elif opt.data_root.find('rscan')!=-1:
    #     opt.dataset = 'rscan'
    # else:
    #     raise NotImplementedError

    
    output_dir = os.path.join(opt.output_folder, opt.dataset, split)
    output_region_list = os.path.join(opt.output_folder, opt.dataset, 'splits','{}.txt'.format(split))
    if os.path.exists(output_dir)==False:os.mkdir(output_dir) 

    print('{} data with split: {}'.format(opt.dataset,split))
    # val_list = []
    files = []
    scene_folders = []
    split_file_list = 'scan_{}.txt'.format(split)
    
    if opt.scene_id is not None:
        # val_list = [opt.scene_id]
        scene_folders = [os.path.join(split_dir,opt.scene_id)]
    else:
        with open(os.path.join(opt.output_folder,opt.dataset,'splits',split_file_list)) as f:
            for line in f.readlines():
                scene_name = line.strip()
                if scene_name =='scene0591_01':
                    print('scene0591_01 is not processed')
                    continue
                if opt.dataset == 'scannetv2' or opt.dataset=='scanneto3d' or opt.dataset=='scannet005':
                    scene_dir = os.path.join(split_dir,scene_name)
                    assert os.path.exists(os.path.join(scene_dir,scene_name+'_vh_clean_2.ply')),'{} miss vh_clean_2.ply'.format(scene_name)
                    assert os.path.exists(os.path.join(scene_dir,scene_name+'_vh_clean_2.0.010000.segs.json')),'{} miss vh_clean_2.0.010000.segs.json'.format(scene_name)
                    assert os.path.exists(os.path.join(scene_dir,scene_name+'.aggregation.json')),'{} miss aggregation.json'.format(scene_name)
                    # files.append(os.path.join(scene_dir,scene_name+'_vh_clean_2.ply'))
                elif opt.dataset =='matterport3d':
                    scene_dir = os.path.join(opt.data_root,scene_name)
                elif opt.dataset == 'rscan':
                    scene_dir = os.path.join(opt.data_root,scene_name)
                    assert os.path.exists(os.path.join(scene_dir,'refined.ply')),'{} miss input ply'.format(scene_name)
                    assert os.path.exists(os.path.join(scene_dir,'annotation.npy')),'{} miss labels'.format(scene_name)                    
                else:
                    raise NotImplementedError
                scene_folders.append(scene_dir)
                # break
            f.close()
        if opt.dataset =='matterport3d':
            from matterport3d import matterport3d_util
            tsv_file = os.path.join(opt.data_root,'category_mapping.tsv')
            
            category_mapping = pd.read_csv(tsv_file,sep='\t',header=0)
            mp_label_mapper = np.insert(category_mapping[['nyu40id']].to_numpy()
                            .astype(int).flatten(), 0, 0, axis=0) # matterport label to nyu40 id
            mp_name_mapper = matterport3d_util.get_raw2nyu_label_map(category_mapping)
            
            region_files = []
            for scene in scene_folders:
                region_files += glob.glob(os.path.join(scene,'region_segmentations','*.ply'))
            scene_folders = region_files
            
            with open(output_region_list,'w') as f:
                region_names = []
                for region_file in sorted(region_files):
                    scene_name = region_file.split('/')[-3]
                    region_name = region_file.split('/')[-1].split('.')[0]
                    region_names.append('{}_{}'.format(scene_name,region_name))
                    f.write('{}_{}\n'.format(scene_name,region_name))
                f.close()
            
    print('Ready to extract {} scenes'.format(len(scene_folders)))
    # scene_folders =[os.path.join(split_dir,'scene0591_01')]
    assert opt.dataset == 'matterport3d'
    # exit(0)
    
    num_process = opt.num_process if opt.num_process is not None else os.cpu_count()
    
    if num_process>1:
        p = mp.Pool(processes=num_process)
        if opt.data_split == 'test':
            # p.map(f_test, files)
            raise NotImplementedError
        else:
            p.map(process_manager, [(opt.dataset,scene,output_dir, opt.img_feat, opt.visualize) for scene in scene_folders])
        p.close()
        p.join()
    else:
        for scene in scene_folders:
            print('processing {}'.format(scene))
            # scene_name = scene.split('/')[-1]
            # if os.path.exists(os.path.join(output_dir,scene_name)):continue
            args = (opt.dataset, scene, output_dir, opt.img_feat, opt.visualize)
            process_manager(args)
            break
    



