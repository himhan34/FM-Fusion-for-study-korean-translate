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

def process_scannet(root_dir,scan_name,posfix,start_ratio,end_ratio):
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
    start_frame = int(N*start_ratio)
    end_frame = int(N*end_ratio)
    frame_list = frame_list[start_frame:end_frame]
    print('trim frame {} to {}'.format(start_frame,end_frame))

    counter = 0
    association_f = open(os.path.join(scan_root,'data_association_{}.txt'.format(posfix)),'w')
    trajectory_f = open(os.path.join(scan_root,'trajectory_{}.log'.format(posfix)),'w')
    from scipy.spatial.transform import Rotation as R
    
    for frame in frame_list:
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
        
        trajectory_f.write('{} {} {}\n'.format(counter,counter,counter+1))
        for row in T_wc:
            trajectory_f.write(' '.join([str(x) for x in row])+'\n')
        counter += 1
        
    association_f.close()
    print('Processed {}/{} frames'.format(counter,len(frame_list)))
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
def process_rio_scan(root_dir,scan_name):
    sequence_dir = os.path.join(root_dir,scan_name,'sequence')
    MAX_FRAMES = 99999999
    prefix = 'frame-'
    color_suffix='.color.jpg'
    depth_suffix='.rendered.depth.png'
    pose_suffix='.pose.txt'
    if(os.path.exists(sequence_dir+'/_info.txt')==False):return False
    print('processing '+scan_name+' ...')
    
    association_dir = os.path.join(root_dir,scan_name,'data_association.txt')
    f_association = open(association_dir,'w')
    f_trajectory = open(os.path.join(root_dir,scan_name,'trajectory.log'),'w')

    count =0
    for i in range(MAX_FRAMES):
        framename = prefix+str(i).zfill(6)
        color_frame = os.path.join(sequence_dir,framename+color_suffix)
        depth_frame = os.path.join(sequence_dir,framename+depth_suffix)
        pose_frame = os.path.join(sequence_dir,framename+pose_suffix)
        if(os.path.exists(color_frame)==False or os.path.exists(pose_frame)==False or os.path.exists(depth_frame)==False): break

        f_association.write('sequence/'+framename+depth_suffix+' sequence/'+ framename+color_suffix+'\n')
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
    print('{} frames are valid and saved'.format(count+1))
    return True


if __name__ == '__main__':

    root_dir = '/data2/ScanNet'
    split = 'val'
    
    clip_dict = {'c':{'start':0.1,'end':0.6},
                'd':{'start':0.4,'end':0.9}}
    
    scans = read_scans(os.path.join(root_dir,'splits',split+'.txt'))
    frame_numbers = []
    # scans = ['scene0064_00']

    for scan_name in scans:   
        for POSFIX in clip_dict:
            START_RATIO = clip_dict[POSFIX]['start']
            END_RATIO = clip_dict[POSFIX]['end']
            num_frames = process_scannet(os.path.join(root_dir,'val'),scan_name,POSFIX,START_RATIO,END_RATIO)
            frame_numbers.append(num_frames)
        
        # break

    print('min frames: {}, max frames: {}'.format(min(frame_numbers),max(frame_numbers)))
