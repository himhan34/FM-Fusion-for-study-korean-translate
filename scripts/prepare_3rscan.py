import os, glob
import cv2, json
import numpy as np
import open3d as o3d
import torch
import shutil
from prepare_datasets import read_scans

def read_scan_pairs(dir):
    with open(dir) as f:
        scan_pairs = []
        for line in f.readlines():
            scan_pairs.append(line.strip().split(' '))
        f.close()
        return scan_pairs

def write_scan_pairs(scan_pairs,dir):
    with open(dir,'w') as f:
        for pair in scan_pairs:
            f.write('{} {}\n'.format(pair[0],pair[1]))
        f.close()

def read_scan_names_dict(dir):
    with open(dir) as f:
        scan_names = {}
        for line in f.readlines():
            parts = line.strip().split(':')
            scan_names[parts[0]] = parts[1]
        f.close()
    return scan_names

def load_rescans_map(json_dir):
    with open(json_dir) as f:
        scenedata = json.load(f)
        scan2ref = {} # [scan, reference_scan]
        scan2ref_transform = {}
        ref_rescans = {}
        N = 0
        for scene in scenedata:
            reference = scene['reference']
            scans = scene['scans']
            scan2ref[reference] = reference
            scan2ref_transform[reference] = np.eye(4)
            rescans = []
            for scan in scans:
                scan2ref[scan['reference']] = reference
                if 'transform' in scan:
                    scan2ref_transform[scan['reference']] = np.transpose(np.array(scan['transform']).reshape(4,4))
                else:
                    scan2ref_transform[scan['reference']] = np.nan
                rescans.append(scan['reference'])
            ref_rescans[reference] = rescans
            N += len(scans)
    return scan2ref, scan2ref_transform, ref_rescans

def process_rio_scan(root_dir,scan_name,depth_source='render'):
    sequence_dir = os.path.join(root_dir,scan_name,'sequence')
    render_folder = os.path.join(root_dir,scan_name,'render')
    MAX_FRAMES = 99999999
    prefix = 'frame-'
    color_suffix='.color.jpg'
    depth_suffix='.rendered.depth.png'
    pose_suffix='.pose.txt'
    if(os.path.exists(sequence_dir+'/_info.txt')==False):return False
    print('processing '+scan_name+' ...')
    
    association_dir = os.path.join(root_dir,scan_name,'data_association.txt')
    info_file_dir = os.path.join(sequence_dir,'_info.txt')
    intrinsic_folder = os.path.join(root_dir,scan_name,'intrinsic')
    f_association = open(association_dir,'w')
    f_trajectory = open(os.path.join(root_dir,scan_name,'trajectory.log'),'w')

    # Write data association and trajectory
    count =0
    for i in range(MAX_FRAMES):
        framename = prefix+str(i).zfill(6)
        color_frame = os.path.join(sequence_dir,framename+color_suffix)
        if depth_source=='render':
            depth_frame = os.path.join(render_folder,framename+depth_suffix)
            depth_folder = 'render/'
        else:
            depth_frame = os.path.join(sequence_dir,framename+depth_suffix)
            depth_folder = 'sequence/'
        
        pose_frame = os.path.join(sequence_dir,framename+pose_suffix)
        if(os.path.exists(color_frame)==False or os.path.exists(pose_frame)==False or os.path.exists(depth_frame)==False): break

        f_association.write(depth_folder+framename+depth_suffix+' sequence/'+ framename+color_suffix+'\n')
        count= i
        # print(color_frame)
        # print(depth_frame)
        # print(pose_frame)
        f_trajectory.write(str(i)+' '+str(i)+' '+str(i+1)+'\n')
        with open(pose_frame) as f:
            for line in f.readlines():
                f_trajectory.write(line)
    
    f_association.close()
    f_trajectory.close()
    
    # Read intrinsic
    img_width, img_height = 0, 0 
    intrinsic = None
    with open(info_file_dir) as info_file:
        lines = info_file.readlines()
        for line in lines:
            parts = line.split('=')
            if 'colorWidth' in parts[0]: img_width = int(parts[1])
            if 'colorHeight' in parts[0]: img_height = int(parts[1])
            if 'ColorIntrinsic' in parts[0]: 
                intrinsic = parts[1].split(' ')[:-1]
                intrinsic = [float(ele) for ele in intrinsic if ele != '']
                intrinsic = np.array(intrinsic).reshape(4,4)    
            # print(parts[0])
    info_file.close()
    print('img shape: {},{}'.format(img_width,img_height))
    print('intrinsic: \n{}'.format(intrinsic))
    
    # Write intrinsic
    os.makedirs(intrinsic_folder,exist_ok=True)
    with open(os.path.join(intrinsic_folder,'intrinsic_depth.txt'),'w') as f:
        np.savetxt(f,intrinsic,fmt='%.6f')
    f.close()
    with open(os.path.join(intrinsic_folder,'sensor_shapes.txt'),'w') as f:
        f.write('color_width:{}\n'.format(img_width))  
        f.write('color_height:{}\n'.format(img_height))
        f.write('depth_width:{}\n'.format(img_width))
        f.write('depth_height:{}\n'.format(img_height))
    f.close()
    
    print('{} frames are valid and saved'.format(count+1))
    return True

def generate_rotate_rgb(root_dir, scan_name):
    sequence_folder = os.path.join(root_dir,scan_name,'sequence')
    color_folder = os.path.join(root_dir,scan_name,'color')
    color_images = glob.glob(sequence_folder +'/*color.jpg')
    rotate_color_images = glob.glob(color_folder +'/*.jpg')
    
    if os.path.exists(color_folder)==False: os.mkdir(color_folder)
    if len(color_images)<3:
        print('WARNING: {} has less than 3 color images'.format(scan_name))
        return False
    if len(rotate_color_images)>3:
        print('skip {}'.format(scan_name))
        return True
    # else:
    #     return False
    # return True
    for color_img_dir in sorted(color_images):
        assert 'render' not in color_img_dir
        frame_name = os.path.basename(color_img_dir).split('.')[0]
        assert 'color' not in frame_name
        frame_id = int(frame_name.split('-')[-1])
        if frame_id%50==0: print('  processed frame {}'.format(frame_id))
        if frame_id % 2 == 0:
            color_img = cv2.imread(color_img_dir)
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(color_folder,frame_name+'.jpg'),color_img)
            # print(frame_id)
            # break
    return True

def clear_scene_folder(root_dir,folder_name='render'):
    color_folders = glob.glob(root_dir+'/*/'+folder_name)
    for color_folder in color_folders:
        # shutil.rmtree(color_folder)
        print('remove {}'.format(color_folder))

def check_pred_and_depth(root_dir, scans):
    
    valid_scans = []
    
    for scan in scans:
        sequence_folder = os.path.join(root_dir, scan,'sequence')
        render_folder = os.path.join(root_dir, scan,'render')
        pred_folder = os.path.join(root_dir, scan,'prediction_no_augment_align')
        
        color_images = glob.glob(sequence_folder +'/*color.jpg')
        render_images = glob.glob(render_folder +'/*rendered.depth.png')
        pred_frames = glob.glob(pred_folder+'/*label.json')
        
        print('{} color, {} render, {} pred'.format(len(color_images),len(render_images),len(pred_frames)))
        if len(color_images)>10 and len(render_images)>10 and len(pred_frames)>10:
            # print('valid scan: {}'.format(scan))
            valid_scans.append(scan)
        
        # print(color_images)
        # break
    
    print('find {}/{} valid scans'.format(len(valid_scans),len(scans)))
    
    return valid_scans
    with open(os.path.join(root_dir,'splits','valid_scans.txt'),'w') as f:
        for scan in valid_scans:
            f.write(scan+'\n')
    f.close()
    
    return valid_scans

def check_dense_mesh(dataroot,scans):
    
    valid_scans = []
    for scan in scans:
        mesh_dir = os.path.join(dataroot,scan,'mesh_o3d.ply')
        mesh = o3d.io.read_triangle_mesh(mesh_dir)
        if mesh.is_empty():
            continue
        else:
            print('valid scan: {}'.format(scan))
            valid_scans.append(scan)
    print('{}/{} valid scans'.format(len(valid_scans),len(scans)))

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
            joint_tags = 'raw tags: {}valid tags: {}'.format(rephrase_raw_tags, tags)
            return boxes, semantics, raw_tags, tags
        else:
            return None,None,None,None


def back_rotate_prediction(dataroot,scans):
    image_width = 540
    image_height = 960
    valid_scans = []
    
    for scan in scans:
        print('-------- processing {}----------'.format(scan))
        pred_folder = os.path.join(dataroot,scan,'prediction_no_augment')
        align_pred_folder = os.path.join(dataroot,scan,'prediction_no_augment_align')
        if os.path.exists(align_pred_folder)==False: os.mkdir(align_pred_folder)
        
        frames_label_files = glob.glob(pred_folder+'/*label.json')
        count = 0
        for frame_label_file in sorted(frames_label_files):
            frame_name = os.path.basename(frame_label_file)[:12]
            frame_id = int(frame_name[6:])
            # print(frame_name)
            
            # rotate box
            aligned_boxes = []
            boxes, semantics, raw_tags, tags = load_pred(frame_label_file)
            if boxes is None: continue
            for box, semantic_score in zip(boxes,semantics):
                box_vec = np.array(box).reshape(-1) # [x1,y1,x2,y2]
                # rotate box coordinates anti-clockwise
                aligned_box_array = [box_vec[1],image_width-box_vec[2],box_vec[3],image_width-box_vec[0]]
                # box_w = box_vec[2]-box_vec[0]
                # box_h = box_vec[3]-box_vec[1]
                # aligned_box_array = [image_height-box_vec[1],box_vec[0],image_height-box_vec[3],box_vec[2]]
                aligned_box_array = [float(ele) for ele in aligned_box_array]
                aligned_boxes.append(aligned_box_array)

            # export
            with open(os.path.join(align_pred_folder,frame_name+'_label.json'),'w') as f:
                # json.dump({'raw_tags':raw_tags,'tags':tags,},f)
                mask_list = [{'value':0, 'label':'background'}]
                i = 1
                for aligned_box_array, semantic_score in zip(aligned_boxes,semantics):
                    # print(aligned_box_array)
                    # aligned_box_string = [str(ele) for ele in aligned_box_array]
                    mask_list.append({'value':i,'labels':semantic_score,'box':aligned_box_array})
                    i+=1
                json.dump({'raw_tags':raw_tags,
                           'tags':tags,
                            'mask':mask_list},f)
                f.close()

            # read and rotate mask image
            mask_img = cv2.imread(os.path.join(pred_folder,frame_name+'_mask.png'),cv2.IMREAD_UNCHANGED)
            mask_img = cv2.rotate(mask_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(align_pred_folder,frame_name+'_mask.png'),mask_img)
            count += 1

            # break
        # break
        print('{} back rotate {} frames'.format(scan,count))
        if count>0: valid_scans.append(scan)
    print('finished {}/{} valid scans'.format(len(valid_scans),len(scans)))

def map_scene_names(dataroot,rescans):
    scan2ref, scan2ref_transform, _ = load_rescans_map(os.path.join(dataroot,'3RScan.json'))

    # organize graph data
    ref_rescan_map = {}
    for scan in rescans:
        ref_scan =scan2ref[scan]
        transform = scan2ref_transform[scan]
        if np.any(np.isnan(transform)):
            print('warning: {} has no transform'.format(scan))
        
        if ref_scan not in ref_rescan_map:
            ref_rescan_map[ref_scan] = [scan]
        else:
            ref_rescan_map[ref_scan].append(scan)
    # print(ref_rescan_map)

    #
    index = 100
    scene_name_map = []
    sub_idxs = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    for ref,rescans in ref_rescan_map.items():
        # print(ref, index)
        # sub_idx = 0
        scene_name_map.append(
            {'name':'scene{}_00{}'.format(str(index).zfill(4),sub_idxs[0]),
             'serial':ref,
             'transform':scan2ref_transform[ref]})

        for i, scan in enumerate(rescans):
            if scan==ref:continue
            # sub_idx+=1
            scene_name_map.append(
                {'name':'scene{}_00{}'.format(str(index).zfill(4),sub_idxs[i+1]),
                 'serial':scan,
                 'transform':scan2ref_transform[scan]})
        
        index +=1
    
    #
    for scene_name_dict in scene_name_map:
        print('{}:{}'.format(scene_name_dict['name'],scene_name_dict['serial']))
        # print(scene_name_dict['transform'])
    print('{} scenes'.format(len(scene_name_map)))
    return scene_name_map

def move_scene_graph(dataroot,graphroot,scene_name_map):
    result_folder = os.path.join(dataroot,'output')
    split = 'val'
    if os.path.exists(os.path.join(graphroot,split))==False: 
        os.mkdir(os.path.join(graphroot,split))
    valid_scans = []
    
    for scene_name_dict in scene_name_map:
        serial_name = scene_name_dict['serial']
        scene_name = scene_name_dict['name']
        infolder = os.path.join(result_folder,scene_name_dict['serial'])
        outfolder = os.path.join(graphroot,'val',scene_name)
        if os.path.exists(os.path.join(infolder,'instance_info.txt')):
            # shutil.copytree(infolder,outfolder)
            np.savetxt(os.path.join(outfolder,'transform.txt'),scene_name_dict['transform'],fmt='%.6f')
            valid_scans.append(scene_name_dict)
        # break
    
    # 
    with open(os.path.join(graphroot,'splits','scans.txt'),'w') as f:
        for scene_name_dict in valid_scans:
            f.write('{}:{}\n'.format(scene_name_dict['name'],scene_name_dict['serial']))
        f.close()
    
    #
    ref_scans = []
    rescans = []
    ref_rescan_pairs = []
    for scene_name_dict in valid_scans:
        if scene_name_dict['name'][-1]=='a':
            ref_scans.append(scene_name_dict['name'])
        else:
            rescans.append(scene_name_dict['name'])
    
    #
    for ref_scan in ref_scans:
        scene_name = ref_scan[:-1]
        for rescan in rescans:
            if rescan.startswith(scene_name) and ref_scan!=rescan:
                ref_rescan_pairs.append((rescan, ref_scan))
    #
    with open(os.path.join(graphroot,'splits','val.txt'),'w') as f:
        for ref_scan, rescan in ref_rescan_pairs:
            f.write('{} {}\n'.format(ref_scan,rescan))
        f.close()

    
def verify_depth():
    tmp_folder = '/home/cliuci/tmp'
    sequence_old = os.path.join(tmp_folder,'sequence_old')
    sequence_new = os.path.join(tmp_folder,'sequence_new')
    viz_folder = os.path.join(tmp_folder,'viz')
    
    render_frames = glob.glob(sequence_old+'/*rendered.depth.png')
    for render_frame in sorted(render_frames)[:10]:
        frame_name = os.path.basename(render_frame)
        print(frame_name)
        
        d_old = cv2.imread(render_frame,cv2.IMREAD_UNCHANGED)
        d_new = cv2.imread(os.path.join(sequence_new,frame_name),cv2.IMREAD_UNCHANGED)
        d_new = cv2.rotate(d_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out = np.concatenate([d_old,d_new],axis=1)
        cv2.imwrite(os.path.join(viz_folder,frame_name),out)
        # break

def root2sync(dataroot, syncroot, filetypes=['']):
    ''' move files in 3rscan root to a sync root.'''
    
    segfiles = glob.glob(dataroot+'/*/semseg.v2.json')
    allscans = []
    for segfile in segfiles:
        scan_name = os.path.basename(os.path.dirname(segfile))
        allscans.append(scan_name)

    for scan in allscans:
        inscan_dir = os.path.join(dataroot,scan)
        outscan_dir = os.path.join(syncroot,scan)
        if os.path.exists(outscan_dir)==False: os.makedirs(outscan_dir)

        for filetype in filetypes:    
            print(os.path.join(inscan_dir,filetype))
            if os.path.exists(os.path.join(inscan_dir,filetype)):
                # print('true')
                shutil.copyfile(os.path.join(inscan_dir,filetype),os.path.join(outscan_dir,filetype))
        # break

def move_rendered_depth(sync_folder, dataroot, scans):
    valid_scans = []
    
    for scan in scans:
        print('--------- {} ---------'.format(scan))
        infolder = os.path.join(sync_folder,scan,'render')
        outfolder = os.path.join(dataroot,scan,'render')
        if os.path.exists(infolder)==False: continue
        if os.path.exists(outfolder)==False:
            os.makedirs(outfolder)
        frames = glob.glob(infolder+'/*depth.png')
        for frame in frames:
            frame_name = os.path.basename(frame)
            depth = cv2.imread(frame,cv2.IMREAD_UNCHANGED)
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            cv2.imwrite(os.path.join(outfolder,frame_name),depth)
        print('{} move {} render depth'.format(scan,len(frames)))
    
        valid_scans.append(scan)
    print('moved {} scans render depth'.format(len(valid_scans)))
    

def clear_render_color(dataroot):
    
    allscans = []
    scan_segfiles = glob.glob(dataroot+'/*/semseg.v2.json')
    
    for segfile in scan_segfiles:
        scan_name = os.path.basename(os.path.dirname(segfile))
        allscans.append(scan_name)

    print('total {} scans'.format(len(allscans)))
    
    for scan in scans:
        sequence_folder = os.path.join(dataroot,scan,'sequence')
        render_images = glob.glob(sequence_folder +'/*rendered.color.jpg')
        if len(render_images)>0:
            print('find {} render colors in {}'.format(len(render_images),scan))

def write_scan_serials(scene_names, scene_name_dict, dir):
    serial_names = []
    for scene_pair in scene_names:
        src_serial_name = scene_name_dict[scene_pair[0]]
        ref_serial_name = scene_name_dict[scene_pair[1]]
        print('{}:{}'.format(scene_pair[0],src_serial_name))
        print('{}:{}'.format(scene_pair[1],ref_serial_name))
        if src_serial_name not in serial_names:
            serial_names.append(src_serial_name)
        if ref_serial_name not in serial_names:
            serial_names.append(ref_serial_name)
    print('total {} serial names'.format(len(serial_names)))
    
    with open(dir,'w') as f:
        for serial_name in serial_names:
            f.write(serial_name+'\n')
        f.close()

def move_dense_map(scene_names, scene_name_dict, dataroot, graphroot):
    
    serial_names = {}
    for scene_pair in scene_names:
        src_serial_name = scene_name_dict[scene_pair[0]]
        ref_serial_name = scene_name_dict[scene_pair[1]]
    
        if src_serial_name not in serial_names:
            serial_names[src_serial_name] = scene_pair[0]
        if ref_serial_name not in serial_names:
            serial_names[ref_serial_name] = scene_pair[1]
    
    for serial_name, scene_name in serial_names.items():
        # print('{}:{}'.format(scene_name,serial_name))
        scan_folder = os.path.join(dataroot,serial_name)
        graph_folder = os.path.join(graphroot,'val',scene_name)
        pcd = o3d.io.read_point_cloud(os.path.join(scan_folder,'mesh_o3d.ply'))
        print('read {} points from {}'.format(len(pcd.points),serial_name))
        o3d.io.write_point_cloud(os.path.join(graph_folder,'pcd_o3d.ply'),pcd)
        
    

if __name__=='__main__':
    sync_folder = '/data2/3rscan_sync'
    dataroot = '/data2/3rscan_raw'
    graphroot = '/data2/RioGraph'
    split='val'
    split_file = 'scans_ver2_.txt' #'scans_ver2_.txt'
    
    # clear_scene_folder(dataroot,'render')
    # sync_files(dataroot,'/data2/3rscan_sync',['labels.instances.annotated.v2.ply'])
    scans = read_scans(os.path.join(dataroot, 'splits', split_file))
    # scans = ['4acaebcc-6c10-2a2a-858b-29c7e4fb410d']
    # move_rendered_depth(sync_folder,dataroot,scans)
    # clear_render_color(dataroot)
    val_full_pairs = read_scan_pairs(os.path.join(graphroot,'splits','val_full.txt'))
    val_scan_pairs = []
    for pair in val_full_pairs:
        print('{}:{}'.format(pair[0],pair[1]))
        if pair not in val_scan_pairs:
            val_scan_pairs.append(pair)
            print('valid')
    print('Find {} valid pairs'.format(len(val_scan_pairs)))
    write_scan_pairs(val_scan_pairs,os.path.join(graphroot,'splits','val.txt'))
    exit(0)
    
    scene_name_dict = read_scan_names_dict(os.path.join(graphroot,'splits','scans.txt'))
    # write_scan_serials(val_scan_pairs,scene_name_dict,os.path.join(dataroot,'splits','serials80.txt'))
    # move_dense_map(val_scan_pairs,scene_name_dict, dataroot, graphroot)
    
    # exit(0)
    # 1
    # check_pred_and_depth(dataroot, scans)
    # 2
    # check_dense_mesh(dataroot, scans)
    # 3
    # back_rotate_prediction(dataroot, scans)
    # 4
    # scene_name_dict = map_scene_names(dataroot,scans)
    # move_scene_graph(dataroot,graphroot,scene_name_dict)
    
    exit(0)
    valid_scans = []
    for scan in scans:
        print('----------processing {}--------'.format(scan))
        # process_rio_scan(dataroot, scan)
        # break
        check = generate_rotate_rgb(dataroot, scan)
        if check == False: 
            print('Error: {}'.format(scan))
        # else:
        #     valid_scans.append(scan)
        #     break
        # break
    print('{}/{} scans'.format(len(valid_scans),len(scans)))