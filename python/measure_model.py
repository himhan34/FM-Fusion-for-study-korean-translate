import os, glob,sys
import numpy as np
import open3d as o3d
# import open3d.core as o3c
import cv2
import json
import inspect

import render_result, fuse_detection, prepare_data_inst

SEMANTIC_NAMES = render_result.SEMANTIC_NAMES
SEMANTIC_IDX = render_result.SEMANTIC_IDXS

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
    for i, tag in enumerate(tag_elements):
        prompts[i] = tag.strip() #Prompt(i,tag)
        prompt_list.append(tag.strip())
    return prompts, prompt_list

def create_measurements(detections):
    '''
    Output:
        measurement is {'k':k,'label_name':det.label_name,'score':det.conf}
        [measurement_id,measurement], a map of the measurement info
    '''
    measurments = {}
    for k,det in enumerate(detections):
        measurments[str(k)] = det.labels 
    
    return measurments

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
    ''' For each scan, generate a json file and save it to the result folder.
        Each json file contains, 
        - Global instances info (centroid, label, etc.)
        - Instances been observed at each frame.
        - Frame-wise prompts and detections.
        - Frame-wise association between detections and global instances.
        THe json file is used for calculating the label likelihood matrix. 
    '''
    
    ####### SET DATA DIRECTORIES HERE ########
    dataroot = '/data2/ScanNet'
    split='train'
    PREDICT = 'prediction_augment'
    result_folder = os.path.join(dataroot,'measurements',PREDICT)
    if os.path.exists(result_folder)==False:
        os.makedirs(result_folder)
    ##########################################
        
    # find scans
    scans = render_result.read_scans(os.path.join(dataroot,'splits',split+'.txt'))
    valid_scans =  [scan for scan in scans if os.path.exists(os.path.join(dataroot,split,scan,PREDICT))]
    print('{}/{} scans are valid'.format(len(valid_scans),len(scans)))
    
    # For debug
    # process_scene_new((os.path.join(dataroot,split,valid_scans[0]), result_folder, PREDICT))
    # exit(0)
    
    # Run in mp
    import multiprocessing as mp
    p = mp.Pool(processes=64)
    p.map(process_scene_new, [(os.path.join(dataroot,split,scan), result_folder, PREDICT) for scan in valid_scans])
    p.close()
    p.join()
    

