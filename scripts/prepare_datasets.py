import os, glob
import numpy as np
import zipfile
import subprocess
import cv2

def read_scans(dir):
    scans = []
    with open(dir) as f:
        for line in f.readlines():
            scans.append(line.strip())
            # print(line[:36])
        f.close()
    return scans

def process_scenenn(root_dir, scan_name):
    timestamp_dir = os.path.join(root_dir,scan_name,'timestamp.txt')  
    association_dir = os.path.join(root_dir,scan_name,'data_association.txt')  
    # timestamp = np.loadtxt(timestamp_dir,dtype=np.int32,delimiter=' ')
    POSE_FOLDER = 'pose_raw'
    if os.path.exists(os.path.join(root_dir,scan_name,POSE_FOLDER))==False:
        os.mkdir(os.path.join(root_dir,scan_name,POSE_FOLDER))

    
    with open(os.path.join(root_dir,scan_name,'trajectory.log'),'r') as f:
        transform_table = {}
        frame_id = None
        transform_array = []
        

        for line in f.readlines():
            eles = line.split(' ')
            if len(eles) ==3:
                if len(transform_array)==16:
                    transform_array = np.array(transform_array).reshape(4,4)
                    transform_table[frame_id] = transform_array
                
                frame_id = int(eles[0])
                transform_array = []
            else:
                for ele in eles:
                    transform_array.append(float(ele.strip()))
            # timestamp = np.append(timestamp,np.array([int(line.split(' ')[0])]),axis=0)

        print('load {} transform poses'.format(len(transform_table)))

        # transform 90 degree along x axis
        # t_g_l = np.array([[1,0,0,0],
        #                   [0,0,-1,0],
        #                   [0,1,0,0],
        #                   [0,0,0,1]])
        t_g_l = np.eye(4)
        
        for i in range(len(transform_table)):
            pose_dir = os.path.join(root_dir,scan_name,POSE_FOLDER,'frame-'+str(i).zfill(5)+'.txt')
            t_g_c = np.dot(t_g_l,transform_table[i])
            np.savetxt(pose_dir,t_g_c,fmt='%.6f')
        
    return None

    f_da = open(association_dir,'w')
    
    for row in timestamp:
        # print(row[0])
        index = str(row[0])
        f_da.write('depth/depth'+index.zfill(5)+'.png image/image'+index.zfill(5)+'.png\n')
    f_da.close()
    print('Done')
    
def read_frames(root_dir, scan_name):
    print('-------- Reading frames ScanNet {} --------- '.format(scan_name))
    scan_root = os.path.join(root_dir,scan_name)
    pose_folder = os.path.join(scan_root,'pose')
    
    print(pose_folder)
    pose_list = glob.glob(os.path.join(pose_folder,'*.txt'))
    frame_list = [pose_file.split('/')[-1][:-4] for pose_file in pose_list]
    frame_list = sorted(frame_list)
    print(frame_list[-10:])
    
    max_frame_id = int(frame_list[-1].split('-')[-1])
    
    return max_frame_id
    

def process_scannet(root_dir,scan_name,posfix,start_frame,end_frame):
    print('-------- Processing ScanNet {} --------- '.format(scan_name))
    scan_root = os.path.join(root_dir,scan_name)
    rgb_folder = os.path.join(scan_root,'color')
    depth_folder = os.path.join(scan_root,'depth')
    pose_folder = os.path.join(scan_root,'pose')
    
    print(pose_folder)
    pose_list = glob.glob(os.path.join(pose_folder,'*.txt'))
    frame_list = [pose_file.split('/')[-1][:-4] for pose_file in pose_list]
    frame_list = sorted(frame_list)
    
    N = len(frame_list)
    # start_frame = int(N*start_ratio)
    # end_frame = int(N*end_ratio)
    # frame_list = frame_list[start_frame:end_frame]
    print('trim frame {} to {}'.format(start_frame,end_frame))

    counter = 0
    association_f = open(os.path.join(scan_root,'data_association_{}.txt'.format(posfix)),'w')
    trajectory_f = open(os.path.join(scan_root,'trajectory_{}.log'.format(posfix)),'w')
    from scipy.spatial.transform import Rotation as R
    latest_frame = None
    
    for frame in frame_list:
        frame_id = int(frame.split('-')[-1])
        if frame_id<start_frame: continue
        if frame_id>end_frame: break
        rgbdir = os.path.join(rgb_folder,frame+'.jpg')
        depthdir = os.path.join(depth_folder,frame+'.png')
        posedir = os.path.join(pose_folder,frame+'.txt')
        assert os.path.exists(rgbdir) and os.path.exists(depthdir), 'rgb or depth image not found'
        T_wc = np.loadtxt(posedir)
        # p_wc = T_wc[:3,3]
        # r_wc = R.from_matrix(T_wc[:3,:3])
        # T_vec = np.concatenate((p_wc,r_wc.as_quat()),axis=0)
        if T_wc[3,3]!=1.0: continue
        association_f.write('depth/'+frame+'.png' +' '
                            +'color/'+frame+'.jpg'+'\n')
        latest_frame = frame
        # print(frame)
        trajectory_f.write('{} {} {}\n'.format(counter,counter,counter+1))
        for row in T_wc:
            trajectory_f.write(' '.join([str(x) for x in row])+'\n')
        counter += 1
        
    association_f.close()
    print('Processed {}/{} frames. End frame {}'.format(counter,len(frame_list),latest_frame))
    return end_frame-start_frame

# Step 1
def uncompress_rio_scan(root_dir,scan_name):
    sequence_zip_dir = os.path.join(root_dir,scan_name,'sequence.zip')
    sequence_dir = os.path.join(root_dir,scan_name,'sequence')
    if(os.path.exists(sequence_dir+'/_info.txt')==False):
        print('Preparing {} sequence'.format(scan_name))
        # print(sequence_zip_dir)
        # print(sequence_dir)
        with zipfile.ZipFile(sequence_zip_dir, 'r') as zip_ref:
            zip_ref.extractall(sequence_dir)

#   Step 2                         : shader bug to be solved
def render_scan(root_dir,scan_name): 
    sequence_dir = os.path.join(root_dir,scan_name,'sequence')
    if(os.path.exists(sequence_dir+'/frame-000020.rendered.depth.png')==False):
        print('Rendering {} sequence'.format(scan_name))
        render_program = '/home/lch/Code_ws/SceneGraph/3RScan/c++/rio_renderer/build/rio_renderer_render_all'
        subprocess.Popen([render_program,
                            root_dir, scan_name,
                            'sequence','2'])
        subprocess.wait()

# Step 3

def clear_scannet_folders(dataroot, scans, filetypes=['']):
    import shutil
    
    for scan in scans:
        scan_folder = os.path.join(dataroot,scan)
        for filetype in filetypes:
            folder = os.path.join(dataroot,scan,filetype)
            if os.path.exists(folder):
                print('Clearing {}'.format(folder))
                shutil.rmtree(folder)
        # break


if __name__ == '__main__':
    root_dir = '/data2/ScanNet'
    split = 'test'
    split_file = split+'.txt'
    
    clip_dict = {
                # 'a':{'start':0.0,'end':0.5},
                # 'b':{'start':0.5,'end':1.0},
                'c':{'start':0.1,'end':0.6},
                'd':{'start':0.4,'end':0.9}}
    SMALL_SEQUENCE = 1000
    GENERATE_FRAME_DICT = True
    GENERATE_SUBSCAN_LIST = True
    
    scans = read_scans(os.path.join(root_dir,'splits',split_file))
    frame_numbers = []
    sequence_frame_dict = {}
    clear_scannet_folders(os.path.join(root_dir,split),scans,
                          ['label', 'instance'])

    exit(0)
    for scan_name in scans:  
        # if GENERATE_FRAME_DICT:
        max_frame_id = read_frames(os.path.join(root_dir,split),scan_name)
        for k,v in clip_dict.items():
            start_frame = max_frame_id * v['start']
            end_frame = max_frame_id * v['end']
            if max_frame_id<SMALL_SEQUENCE:
                start_frame = max((start_frame-0.1*max_frame_id),0)
                end_frame   = min((end_frame+0.1*max_frame_id),max_frame_id)
            sequence_frame_dict[scan_name+k] = [int(start_frame),int(end_frame)]
    
    for scan_name in scans:
        # if GENERATE_SUBSCAN_LIST:
            # for POSFIX in clip_dict:
            # for subscene_name, frame_range in sequence_frame_dict.items():
                # POSFIX = subscene_name[-1]
        frame_range_c = sequence_frame_dict[scan_name+'c']
        frame_range_d = sequence_frame_dict[scan_name+'d']
        process_scannet(os.path.join(root_dir,split),scan_name,'c',frame_range_c[0],frame_range_c[1])
        process_scannet(os.path.join(root_dir,split),scan_name,'d',frame_range_d[0],frame_range_d[1])
        
        # break
    print(sequence_frame_dict)
    
    import json
    with open(os.path.join(root_dir,'sub_sequences_dict.json'),'w') as f:
        json.dump(sequence_frame_dict,f)    
    # print('min frames: {}, max frames: {}'.format(min(frame_numbers),max(frame_numbers)))
