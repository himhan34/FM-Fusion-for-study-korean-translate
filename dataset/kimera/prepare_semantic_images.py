import os, glob, sys
import csv,json
import numpy as np
import cv2

sys.path.append('/home/cliuci/code_ws/OpensetFusion/python')
import fuse_detection

def read_mapper_file(dir):
    csv_file = open(dir, 'r')
    mapper = {} # {label_name: [r,g,b,a]}
    for line in csv_file.readlines()[1:]:
        eles = line.strip().split(',')
        rgba = [int(e) for e in eles[1:-1]]
        name = eles[0]
        mapper[name] = np.array(rgba)
        # print('{}:{}'.format(name, rgba))
    return mapper

def read_hard_association(dir):
    openset_to_nyu20 = {} # {openset_name: [nyu20_name]}
    empirical_association = json.load(open(dir,'r'))
    objects = empirical_association['objects']
    for gt_name, gt_info in objects.items():
        openset_names = gt_info['main']
        for openset in openset_names:
            openset_to_nyu20[openset] = gt_name
            print('{}:{}'.format(openset,gt_name))
    
    print('{} valid openset names'.format(len(openset_to_nyu20)))
    return openset_to_nyu20


def generate_gsam_sequence(scene_dir:str,output_scene_folder:str,association_maper, class_to_color):
    IMAGE_SHAPE = (480, 640) # (height, width)
    FRAME_GAP = 10
    DATASET = 'scenenn'
    pred_folder = os.path.join(scene_dir, 'prediction_no_augment')
    output_folder = os.path.join(output_scene_folder, 'pred_gsam_color')
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder)
    
    prediction_frames = glob.glob(os.path.join(pred_folder, '*_label.json'))

    for i, pred_file in enumerate(sorted(prediction_frames)):
        frame_name = os.path.basename(pred_file).split('_')[0]
        
        if DATASET=='scannet':
            frame_stamp = int(frame_name.split('-')[-1])
            depth_frame_name = frame_name
        elif DATASET=='scenenn':
            frame_stamp = int(frame_name[-5:])
            depth_frame_name = frame_name.replace('frame','depth')
        else :
            raise NotImplementedError
        if frame_stamp%FRAME_GAP!=0: continue


        print('write detection result {}:{}'.format(os.path.basename(scene_dir),frame_name))

        # read depth
        if DATASET=='scannet':
            depth_dir = os.path.join(scene_dir, 'depth', depth_frame_name+'.png')
            if os.path.exists(depth_dir)==False:
                print('reached the end of the sequence')
                break
            depth_img = cv2.imread(depth_dir, cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img.astype(np.float32)/1000.0

        # read and filter predictions
        semantic_image = np.zeros((IMAGE_SHAPE[0],IMAGE_SHAPE[1],3), dtype=np.uint8)
        _, detections = fuse_detection.load_pred(pred_folder, frame_name, list(association_maper.keys()))
        detections = fuse_detection.filter_overlap_detections(detections= detections, min_iou=0.1)
        if DATASET=='scannet':
            detections = fuse_detection.filter_detection_depth(detections,depth_img)
        if detections is None: 
            print('empty tags!')
            cv2.imwrite(os.path.join(output_folder, '{}.png'.format(frame_name)), semantic_image)
            continue

        #
        for zk in detections:
            openset_label = list(zk.labels.keys())[0]
            if openset_label not in association_maper: continue
            semantic_class = association_maper[openset_label]
            assert semantic_class in class_to_color, 'invalid semantic class:{}, openset label: {}'.format(semantic_class, openset_label)
            semantic_color = class_to_color[semantic_class][:3]
            semantic_image[zk.mask] = semantic_color

        # export
        cv2.cvtColor(semantic_image, cv2.COLOR_RGB2BGR, semantic_image)
        cv2.imwrite(os.path.join(output_folder, '{}.png'.format(frame_name)), semantic_image)
        # break

def generate_maskrcnn_seqeunce(scene_dir:str,output_scene_folder:str, label_to_color):
    IMAGE_SHAPE = (480, 640) # (height, width)
    FRAME_GAP = 10
    
    pred_folder = os.path.join(scene_dir, 'pred_maskrcnn_refined')
    output_folder = os.path.join(output_scene_folder, 'pred_maskrcnn_color_rf')
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder)
        
    prediction_frames = glob.glob(os.path.join(pred_folder, '*_label.json'))
    valid_labels = list(label_to_color.keys())
    
    for i, pred_file in enumerate(sorted(prediction_frames)):
        frame_name = os.path.basename(pred_file).split('_')[0]
        frame_stamp = int(frame_name.split('-')[-1])
        semantic_image = np.zeros((IMAGE_SHAPE[0],IMAGE_SHAPE[1],3), dtype=np.uint8)
        if frame_stamp%FRAME_GAP!=0: 
            # cv2.imwrite(os.path.join(output_folder, '{}.png'.format(frame_name)), semantic_image)
            print('skipping write semantic image {}'.format(frame_name))
            continue
        
        # semantic_image[:,:,3] = 255 # alpha channel
        print('write detection result {}:{}'.format(os.path.basename(scene_dir),frame_name))
        _, detections = fuse_detection.load_pred(pred_folder, frame_name, 0.9, valid_labels)

        for zk in detections:
            semantic_class = list(zk.labels.keys())[0]
            if semantic_class not in valid_labels:
                continue
            semantic_color = label_to_color[semantic_class][:3]
            # m = zk.mask
            # print('m.shape:{}'.format(m.shape))
            semantic_image[zk.mask] = semantic_color
        
        cv2.cvtColor(semantic_image, cv2.COLOR_RGB2BGR, semantic_image)
        cv2.imwrite(os.path.join(output_folder, '{}.png'.format(frame_name)), semantic_image)
        # break
    
if __name__=='__main__':
    # data_root = '/data2/scenenn'
    data_root = '/data2/ScanNet'
    split = 'val'
    split_file = 'val'

    # Load label to color mapper    
    association_file = 'measurement_model/categories.json'
    maskrcnn_mapper_dir = 'dataset/kimera/maskrcnn_mapping.csv'
    gsam_mapper_dir = 'dataset/kimera/gsam_mapping.csv'
    maskrcnn_mapper = read_mapper_file(maskrcnn_mapper_dir)
    gsam_color_mapper = read_mapper_file(gsam_mapper_dir)
    openset_to_nyu20 = read_hard_association(association_file)
    
    scans = fuse_detection.read_scans(os.path.join(data_root,'splits',split_file+'.txt'))
    print('find {} scans'.format(len(scans)))
    output_folder = os.path.join(data_root,split)
    # output_folder = os.path.join(data_root,'target_scans')
    # scans = ['scene0011_01']
    
    for scan in scans:
        out_scene_folder = os.path.join(output_folder,scan)
        if os.path.exists(out_scene_folder)==False:
            os.makedirs(out_scene_folder)
        # generate_gsam_sequence(os.path.join(data_root,split,scan),out_scene_folder,openset_to_nyu20,gsam_color_mapper)
        generate_maskrcnn_seqeunce(os.path.join(data_root,split,scan),out_scene_folder,maskrcnn_mapper)
        # break
