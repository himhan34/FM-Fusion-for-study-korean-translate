import os, glob,sys
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2, plyfile
import json
# from numpy import linalg as LA
import inspect

import render_result, fuse_detection
# sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
# sys.path.append('/home/cliuci/PointGroup/dataset')
import project_util, prepare_data_inst

SEMANTIC_NAMES = render_result.SEMANTIC_NAMES
SEMANTIC_IDX = render_result.SEMANTIC_IDXS
HARD_ASSOCIATION_DIR = '/home/cliuci/code_ws/OpensetFusion/measurement_model/categories.json'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from dataset.scannetv2 import util

class Measurement:
    def __init__(self,id,det_label_name,det_score,iou,viewed_points_count):
        self.id = id
        self.view_points_number = viewed_points_count
        self.iou = iou
        self.pred_label_name = det_label_name 
        self.pred_score = det_score

class Instance:
    def __init__(self,idx,label_name,label_id,points, N):
        self.id = idx
        self.gt_label = label_name
        self.gt_label_id = label_id # -1 for invalid
        self.points = points # point indices
        self.N = N # total number of points
        self.measurements = []
    
    def get_points(self):
        '''
            return a boolean array of points, (N,), np.bool_
        '''
        points_array = np.zeros(self.N,dtype=np.bool_)
        points_array[self.points] = True
        return points_array

class Prompt:
    def __init__(self,id,name):
        self.id = id
        self.openset_name = name

class DetInstancePair:
    def __init__(self,scan,frame_name,det,instance_id,iou):
        self.det = det
        self.instance_id = instance_id
        self.iou = iou

# todo: remove this
def extract_background_gt(fn2, segid_to_pointid,min_points=20000):
    '''
    Input:
        - fn2: str, path to the label file
    Output:
        - background_points: (N,3), np.float32
    '''
    
    f2 = plyfile.PlyData().read(fn2)
    labels = prepare_data_inst.remapper[np.array(f2.elements[0]['label'])] # [0,19]
    bg_instances = []
    
    for segid, points in segid_to_pointid.items():
        points_np = np.array(points)
        seg_labels = labels[points_np].astype(np.int32)
        mask = np.logical_or(seg_labels ==0, seg_labels==1) # mask background
        if np.all(mask) and len(points)>min_points:
            bg_instances.append(Instance(segid,SEMANTIC_NAMES[seg_labels[0]],-1,points_np,labels.shape[0]))
    print('add {} bg instances'.format(len(bg_instances)))
    return bg_instances

def read_scannet_gt(scene_folder, coords, min_points=500):
    scene_name = os.path.basename(scene_folder)
    fn = os.path.join(scene_folder,scene_name+'_vh_clean_2.ply')
    
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    
    # f2 = plyfile.PlyData().read(fn2)
    # labels = prepare_data_inst.remapper[np.array(f2.elements[0]['label'])] # [0,19]
    segid_to_pointid = prepare_data_inst.read_scannet_segjson(fn3)
    instance_segids, instance_labels = prepare_data_inst.read_scannet_aggjson(fn4, 
                                                                              ignore_labels=['ceiling', 'remove', 'unknown'])
    
    M = len(instance_segids)
    Instances = []
    count_bg = 0
    
    for i in range(M):
        segs = instance_segids[i]
        points = []
        for seg in segs:
            points +=segid_to_pointid[seg]
        points = np.array(points)
        label_name = instance_labels[i]
        if points.shape[0]>min_points:
            last_ele_id = len(Instances)
            if label_name =='floor' or label_name=='wall': count_bg +=1
            Instances.append(Instance(last_ele_id,label_name,-1,points,coords.shape[0]))

    print('---- {} find {}/{} valid instances, including {} bg instances'.format(scene_name,len(Instances),M, count_bg))
    # print('instance labels: {}'.format(labels))
    return Instances
    
def cal_overlap(instance_map,detections):
    '''
    Input:
        - instance_map: (H,W), np.bool_
        - detections: list of Detection
    Output:
        - ious: (K,), np.float32
    '''
    ious = np.zeros(len(detections))
    for k,zk in enumerate(detections):
        overlap = instance_map & zk.mask
        ious[k] = overlap.sum()/instance_map.sum()
    return ious                

def create_prompts(tags):
    '''
    Output: 
        [tag_id,tag_name], a map of the prompts
    '''
    prompts ={}
    prompt_list = []
    tag_elements = tags.split(',')
    # id_offset = len(prompts_map)
    # current_prompts_id = []
    for i, tag in enumerate(tag_elements):
        prompts[i] = tag.strip() #Prompt(i,tag)
        prompt_list.append(tag.strip())
        # current_prompts_id.append(i)
    return prompts, prompt_list

def create_measurements(detections):
    '''
    Output:
        measurement is {'k':k,'label_name':det.label_name,'score':det.conf}
        [measurement_id,measurement], a map of the measurement info
    '''
    measurments = {}
    # id_offset = len(measurement_map)
    # current_measurements_id = []
    for k,det in enumerate(detections):
        # zk = Measurement(k, det.label_name, det.conf, 0, 0)
        # assert det.conf>0.2, 'detection score {} too low'.format(det.conf)
        measurments[str(k)] = det.labels 
            #{'k':k,'labels':det.labels}
                            # 'label_name':det.label_name,'score':det.conf})
        # current_measurements_id.append(k)
    
    return measurments

def read_hard_association(dir):
    with open(dir,'r') as f:
        data = json.load(f)
        label_mapper = {} # close-set to open-set: {closeset_label: [openset_labels]}
        for lc,lo_list in data['objects'].items():
            # print('{}: {}'.format(lc,lo_list['main']))
            label_mapper[lc] = lo_list['main']
        f.close()
        return label_mapper

def process_scene(args):
    scene_dir, output_dir,prediction_name = args
    scene_name = os.path.basename(scene_dir)
    viz_dir = os.path.join(scene_dir,'tmp2')
    outfile = os.path.join(output_dir,'{}.json'.format(scene_name))
    # if os.path.exists(outfile):
    #     print('skip {}'.format(scene_name))
    #     return

    if os.path.exists(viz_dir)==False:
        os.mkdir(viz_dir)
    
    MAP_POSIX = 'vh_clean_2.ply'
    INTRINSIC_FOLDER ='intrinsic'
    MIN_VIEW_POINTS = 3000
    FRAME_GAP = 20
    MIN_IOU = 0.8
    
    # depth filter related params
    visualize = False
    max_dist_gap = 0.2
    depth_kernal_size = 5
    kernal_valid_ratio = 0.2
    kernal_max_var = 0.15
    
    # Init
    K_rgb, K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,INTRINSIC_FOLDER))
    rgb_out_dim = depth_dim
    K_rgb_out = project_util.adjust_intrinsic(K_rgb,rgb_dim,rgb_out_dim)
    predict_folder = os.path.join(scene_dir,prediction_name)
    predict_frames =  glob.glob(os.path.join(predict_folder,'*_label.json'))  
    print('---- {} find {} prediction frames'.format(scene_name,len(predict_frames)))
    if len(predict_frames)==0: return 0
    
    # Load map
    map_dir = os.path.join(scene_dir,'{}_{}'.format(scene_name,MAP_POSIX))
    pcd = o3d.io.read_point_cloud(map_dir)
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)*255.0
    normals = np.zeros(points.shape,dtype=np.float32)
    
    N = points.shape[0]
        
    # 
    Instances = read_scannet_gt(scene_dir,points,500)
    association_map = {}
    # exit(0)
    # Run
    for index,pred_frame in enumerate(sorted(predict_frames)):   
        frame_name = os.path.basename(pred_frame).split('_')[0] 
        frameidx = int(frame_name.split('-')[-1])
        if frameidx % FRAME_GAP != 0: continue
                
        rgbdir = os.path.join(scene_dir,'color',frame_name+'.jpg')
        pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        depth_dir = os.path.join(scene_dir,'depth',frame_name+'.png')

        # load rgbd, pose
        rgbimg = cv2.imread(rgbdir)
        raw_depth = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        depth = raw_depth.astype(np.float32)/1000.0
        assert raw_depth.shape[0]==depth_dim[0] and raw_depth.shape[1]==depth_dim[1], 'depth image dimension does not match'
        assert depth.shape[0] == rgbimg.shape[0] and depth.shape[1] == rgbimg.shape[1]
        pose = np.loadtxt(pose_dir)

        # projection
        view_mask, points_uv, _, _ = project_util.project(
            points, normals,pose, K_rgb_out, rgb_out_dim, 5.0, 0.5) # Nx3
        filter_mask = project_util.filter_occlusion(points_uv,depth,max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        view_mask = np.logical_and(view_mask,filter_mask)
        points_uv[~view_mask] = np.array([-100,-100,-100]) # (N,3)
        points_uv = points_uv.astype(np.int32)
        
        # load prediction
        tags, detections = fuse_detection.load_pred(predict_folder, frame_name, True)
        if len(detections)<1: continue
        
        # create current vertices
        frame_association = dict()
        _, frame_association['prompts'] = create_prompts(tags)
        frame_association['detections'] = create_measurements(detections)

        # assignment
        proj_rgb = np.zeros(rgbimg.shape,dtype=np.uint8)
        assignment_det_gt =  np.zeros((len(detections),len(Instances)),dtype=np.int32)
        matches_gt = []
        matches_promp = []
        viewed_instances = []
        
        # viewed_instances = 0
        instances_centroid = np.zeros((len(Instances),2),dtype=np.int32) # (N,2),[u,v]
        for j,instance in enumerate(Instances): # assign detection and gt instances
            instance_mask = instance.get_points() # (N,), bool
            uv = points_uv[instance_mask&view_mask,:2]
            uv_rgb = colors[instance_mask&view_mask,:3].astype(np.uint8)
            if np.sum(uv)<MIN_VIEW_POINTS: continue
            # viewed_instances +=1
            uv_map = np.zeros((rgbimg.shape[0],rgbimg.shape[1]),dtype=np.bool_)
            uv_map[uv[:,1],uv[:,0]] = True
            ious = cal_overlap(uv_map,detections)
            k_ = np.argmax(ious)
            if ious[k_] > MIN_IOU: 
                assignment_det_gt[k_,j] = 1
                match_={'det':int(k_),'gt':int(j),'iou':ious[k_]}
                # match_ = DetInstancePair(scene_name,frame_name,k_,j,ious[k_])
                # associations_det_inst.append(match_)
                matches_gt.append(match_)
            viewed_instances.append(j)
            centroid = np.mean(uv,axis=0)

            # viz
            for i in np.arange(uv.shape[0]):
                cv2.circle(proj_rgb,(uv[i,0],uv[i,1]),1,(int(uv_rgb[i,2]),int(uv_rgb[i,1]),int(uv_rgb[i,0])),-1)
            cv2.putText(proj_rgb,instance.gt_label,(int(centroid[0]+10),int(centroid[1]+10)),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1,cv2.LINE_AA)
            if assignment_det_gt[k_,j]==1: # record_measurement
                centroid_color = (0,255,0)
                zk_ = detections[k_]
                bbox_centroid = zk_.cal_centroid()
                cv2.line(proj_rgb,(int(centroid[0]),int(centroid[1])),(int(bbox_centroid[0]),int(bbox_centroid[1])),(0,255,255),1)
                # measure_k = Measurement(0,zk_.label_name,zk_.conf,ious[k_],np.sum(uv))
                # instance.measurements.append(measure_k)
            else:
                centroid_color = (0,0,255)
            cv2.circle(proj_rgb,(int(centroid[0]),int(centroid[1])),3,centroid_color,-1)

        # for i, prompt_name in frame_association['prompts'].items(): # assign prompt
        #     for k, zk in enumerate(frame_association['detections']): 
        #         if prompt_name in zk['label_name']:
        #             matches_promp.append({'det':int(k),'prompt':int(i)})
        
        frame_association['matches_gt'] = matches_gt
        frame_association['viewed_instances'] = viewed_instances
        # frame_association['matches_promp'] = matches_promp
        association_map[frame_name] = frame_association
        
        # print('{}/{} instances are matched'.format(np.sum(assignment_det_gt),len(viewed_instances)))
        
        if visualize:
            for k_,zk in enumerate(detections): # mark detection
                # print('{} has {} masks'.format(zk.label_name,np.sum(zk.mask)))
                if assignment_det_gt[k_,:].max()>0:
                    bbox_color = (0,255,0)
                else: bbox_color = (0,0,255)
                
                token_msg = ''
                for os_label, score in zk.labels.items():
                    token_msg += '{}({:.2f}) '.format(os_label,score)

                cv2.rectangle(proj_rgb,pt1=(int(zk.u0),int(zk.v0)),pt2=(int(zk.u1),int(zk.v1)),color=bbox_color,thickness=1)
                cv2.putText(proj_rgb, token_msg, (int(zk.u0+10),int(zk.v0+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            out = proj_rgb
            cv2.imwrite(os.path.join(viz_dir,'{}_measure.png'.format(frame_name)),out)
        # if index>2:
        #     break

    # Export data
    json_data = {"min_iou":MIN_IOU,"frame_gap":FRAME_GAP}
    
    instance_data = []# dict()
    for j, instance in enumerate(Instances):
        lj = {
            'id': int(j),
            'label_name': instance.gt_label,
            'points_num': int(instance.get_points().sum())
            }
        instance_data.append(lj)
    json_data['instances'] = instance_data
    json_data['associations'] = association_map

    with open(outfile,'w') as f:
        json.dump(json_data,f)
        f.close()
        print('result write to {}'.format(outfile))

def process_scene_new(args):
    import imageio.v2 as imageio
    label_map_file = '/data2/ScanNet/scannetv2-labels.combined.tsv'

    scene_dir, output_dir,prediction_name = args
    scene_name = os.path.basename(scene_dir)
    viz_dir = os.path.join(scene_dir,'tmp2')
    outfile = os.path.join(output_dir,'{}.json'.format(scene_name))
    predict_folder = os.path.join(scene_dir,prediction_name)

    MIN_VIEW_POINTS = 3000
    FRAME_GAP = 20
    MIN_IOU = 0.8
    PRED_IMG_WIDTH = 640
    PRED_IMG_HEIGHT = 480

    predict_frames =  glob.glob(os.path.join(predict_folder,'*_label.json'))  
    print('---- {} find {} prediction frames'.format(scene_name,len(predict_frames)))
    if len(predict_frames)==0: return 0
    association_map = {} # {frame_name: frame_association}. It only considers the instances been detected.
    ram_results = {} # {frame_name: ram_result}. All the observed instances should be considered.

    for index,pred_frame in enumerate(sorted(predict_frames)):   
        frame_name = os.path.basename(pred_frame).split('_')[0] 
        frameidx = int(frame_name.split('-')[-1])
        if frameidx % FRAME_GAP != 0: continue

        # load prediction
        tags, detections = fuse_detection.load_pred(predict_folder, frame_name)
        if len(detections)<1: continue
        
        # load gt
        instance_file = os.path.join(scene_dir,'instance','{}.png'.format(frameidx))
        label_file = os.path.join(scene_dir,'label','{}.png'.format(frameidx))
        assert os.path.exists(instance_file) and os.path.exists(label_file), 'instance or label file does not exist'
        
        instance_image = np.array(imageio.imread(instance_file))
        label_image = np.array(imageio.imread(label_file))
        label_map = util.read_label_mapping(label_map_file, label_from='id', label_to='nyu40id')
        output_label_image = util.map_label_image(label_image, label_map) # nyu40id
        output_instance_image = util.make_instance_image(output_label_image, instance_image) # np.uint16

        instant_indices = np.unique(output_instance_image)
        
        # print('find {} instances'.format(len(instant_indices)))
        
        if output_label_image.shape[0] != PRED_IMG_HEIGHT or output_label_image.shape[1] != PRED_IMG_WIDTH:
            output_label_image = cv2.resize(output_label_image,(PRED_IMG_WIDTH,PRED_IMG_HEIGHT),interpolation=cv2.INTER_LINEAR)
        if output_instance_image.shape[0] != PRED_IMG_HEIGHT or output_instance_image.shape[1] != PRED_IMG_WIDTH:
            output_instance_image = cv2.resize(output_instance_image,(PRED_IMG_WIDTH,PRED_IMG_HEIGHT),interpolation=cv2.INTER_LINEAR)    
        
        # aligned_lable_image = util.get_labels_from_instance(output_instance_image)
        
        # assignment
        instances = []
        
        # Update data
        frame_ram = {'observed':[], 'prompts':create_prompts(tags)[1][0]} 
        frame_association = dict()
        _, frame_association['prompts'] = create_prompts(tags)
        frame_association['detections'] = create_measurements(detections)

        # update instances with matches
        for i,inst_id in enumerate(instant_indices):
            if inst_id==0: continue
            gt_mask = output_instance_image==inst_id
            if gt_mask.sum()<MIN_VIEW_POINTS: continue
            gt_label40_id = output_label_image[gt_mask][0]
            if gt_label40_id not in SEMANTIC_IDX: continue
            gt_label20_id = np.where(SEMANTIC_IDX==gt_label40_id)[0][0]
            # print('find instance {} with label {}'.format(inst_id,SEMANTIC_NAMES[gt_label40_id]))
            frame_ram['observed'].append(SEMANTIC_NAMES[gt_label20_id])
            
            ious = cal_overlap(gt_mask,detections)
            k_ = np.argmax(ious)
            if ious[k_] > MIN_IOU: 
                matched_instance={'gt':int(inst_id),'label':SEMANTIC_NAMES[gt_label20_id],
                           'det':int(k_),'iou':ious[k_],'viewed':int(gt_mask.sum())}
                instances.append(matched_instance)
                # print('gt instance {} is matched with detection {}'.format(SEMANTIC_NAMES[gt_label20_id],detections[k_].labels))
            else:
                matched_instance = {'gt':int(inst_id),'label':SEMANTIC_NAMES[gt_label20_id],
                           'det':-1,'iou':0,'viewed':int(gt_mask.sum())}
            
            # break

        frame_association['instances'] = instances
        association_map[frame_name] = frame_association
        ram_results[frame_name] = frame_ram
        
        print('find {}/{} matched instances'.format(len(instances),len(instant_indices)))
        # break

    # Export data
    json_data = {"min_iou":MIN_IOU,"frame_gap":FRAME_GAP}
    json_data['associations'] = association_map
    json_data['ram'] = ram_results

    with open(outfile,'w') as f:
        json.dump(json_data,f)
        f.close()
        print('result write to {}'.format(outfile))

if __name__=='__main__':
    dataroot = '/data2/ScanNet'
    split='train'
    # PREDICT = 'prediction_no_augment'
    PREDICT = 'prediction_forward'
    result_folder = os.path.join(dataroot,'measurements','bayesian')
    if os.path.exists(result_folder)==False:
        os.makedirs(result_folder)
    scans = render_result.read_scans(os.path.join(dataroot,'splits','train_micro.txt'))
    valid_scans =  [scan for scan in scans if os.path.exists(os.path.join(dataroot,split,scan,PREDICT))]
    print('{}/{} scans are valid'.format(len(valid_scans),len(scans)))
    # exit(0)
    
    # scans = ['scene0407_01']
    
    # for scan in scans:
    #     process_scene_new((os.path.join(dataroot,split,scan),result_folder,PREDICT))
    #     break
    # exit(0)
    
    import multiprocessing as mp
    p = mp.Pool(processes=16)
    p.map(process_scene_new, [(os.path.join(dataroot,split,scan), result_folder, PREDICT) for scan in valid_scans])
    p.close()
    p.join()
    

