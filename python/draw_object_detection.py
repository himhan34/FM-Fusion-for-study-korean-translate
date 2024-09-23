import os, glob 
import json
import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

def show_box_tokens(box,ax,token_pairs):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    # ax.text(x0, y0, label)
    labels = ''
    for token, score in token_pairs.items():
        labels += '{}({:.3f}) '.format(token, score)
    ax.text(x0, y0, labels)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def draw_object_detection(rgb:np.ndarray, box:np.ndarray, label, color=(0,255,0), thickness=2):
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    cv2.rectangle(rgb, (x0, y0), (x1, y1), color, thickness)
    cv2.putText(rgb, label, (x0+5, y0 +16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return rgb

def read_scan_list(dir):
    with open(dir) as f:
        scan_list = f.readlines()
        scans = [scan.strip() for scan in scan_list]
        
        f.close()
        return scans

def load_pred(label_file,valid_openset_names=None):
    with open(label_file, 'r') as f:
        json_data = json.load(f)
        tags = json_data['tags'] if 'tags' in json_data else ''
        raw_tags = json_data['raw_tags'] if 'raw_tags' in json_data else ''
        masks = json_data['mask']
        boxes = [] # [x1,y1,x2,y2]
        semantics = [] # [{label:conf}]

        for ele in masks:
            if 'box' in ele:
                # if label[-1]==',':label=label[:-1]
                instance_id = ele['value']-1    
                detection_id = ele['value']
                bbox = ele['box']  
                labels = ele['labels'] # {label:conf}

                # box_area_normal = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(img_width*img_height)
                # if box_area_normal > MAX_BOX_RATIO: continue

                if valid_openset_names is not None:
                    valid = False
                    for label in labels:
                        if label in valid_openset_names:
                            valid = True
                            break
                    if valid==False: continue
                box_tensor = torch.Tensor([[bbox[0],bbox[1]],[bbox[2],bbox[3]]]) # [x1,y1,x2,y2]
                boxes.append(box_tensor.unsqueeze(0)) # [x1,y1,x2,y2]
                semantics.append(labels)
                # z_ = Detection(bbox[0],bbox[1],bbox[2],bbox[3],labels)
                # z_.add_mask(mask==detection_id)
                # detections.append(z_)
                
            else: # background
                continue  
        assert len(boxes) == len(masks)-1, 'boxes dimension not aligned'
        f.close()
        if len(boxes)>0: 
            boxes = torch.cat(boxes, dim=0) # (num_prompts, 2, 2)
            LINE_LENGTH = 60
            if len(raw_tags)>LINE_LENGTH:
                # seperate tags by line. Each line should be less than 50 characters
                number_lines = int(len(raw_tags)/LINE_LENGTH)+1
                rephrase_raw_tags = ''
                for i in range(number_lines):
                    rephrase_raw_tags += raw_tags[i*LINE_LENGTH:(i+1)*LINE_LENGTH]+'\n'
            else:
                rephrase_raw_tags = raw_tags
            joint_tags = 'raw tags: {} valid tags: {}'.format(rephrase_raw_tags, tags)
            return boxes, semantics, joint_tags
        else:
            return torch.zeros((1,2,2)), [], ''
            return None,None,None

if __name__=='__main__':
    ########## SET PARAMETERS ##########
    dataroot = '/data2/ScanNet'
    scans_folder = os.path.join(dataroot,'val')
    split_file = 'tmp.txt'
    prediction_folder = 'prediction_no_augment'
    visualize_mask = False
    ####################################
    
    rotated = False
    if '3rscan' in dataroot:
        rgb_folder = 'sequence' # color
        rgb_posfix = '.color.jpg'  #'.jpg'
        scans_folder = dataroot
        rotated = True
    elif 'sgslam' in dataroot:
        rgb_folder = 'rgb'
        rgb_posfix = '.png'
    elif 'bim' in dataroot:
        rgb_folder = 'color'
        rgb_posfix = '.jpg'
        scans_folder = os.path.join(dataroot, 'test')
    else:
        rgb_folder = 'color'
        rgb_posfix = '.jpg'
    
    ### original prediction are based on rotated rgb ###    
    scans = read_scan_list(os.path.join(dataroot, 'splits', split_file))
    scene_pred_folders = [os.path.join(scans_folder, scan, prediction_folder) for scan in scans]
    print('find {} pred folders'.format(len(scene_pred_folders)))

    for pred_folder in scene_pred_folders:
        scene = pred_folder.split('/')[-2]
        scene_viz_folder = os.path.join(scans_folder, scene, 'pred_viz')
        if os.path.exists(scene_viz_folder)==False:
            os.makedirs(scene_viz_folder)

        if 'lg' in scene: continue
        print('--------- processing {}----------'.format(scene))
        pred_files = glob.glob(os.path.join(pred_folder, '*.json'))
        count = 0
        for frame_pred in sorted(pred_files):
            print(frame_pred)
            frame_name = frame_pred.split('/')[-1][:12]
            frame_id = int(frame_name[6:])
            if count % 50==0:
                print('{} / {}'.format(frame_name, len(pred_files)))
                
            boxes, semantics, tags = load_pred(frame_pred)
            color_file = os.path.join(scans_folder, scene, rgb_folder, frame_name+rgb_posfix)
            rgb = cv2.imread(color_file)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if rotated: rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)            

            for box, semantic_label_dict in zip(boxes, semantics):
                label = list(semantic_label_dict.keys())[0]
                score = semantic_label_dict[label]       
                label_score_str = '{}({:.2f})'.format(label, score)             
                rgb = draw_object_detection(rgb, box.numpy().astype(np.int32).reshape(-1),
                                label_score_str,
                                color=(0,255,0), thickness=2)

            cv2.imwrite(os.path.join(scene_viz_folder, frame_name+'.jpg'), rgb)
            count +=1
            
        print('{} saved {} frames'.format(scene, count))
