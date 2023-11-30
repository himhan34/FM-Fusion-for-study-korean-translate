import os, sys
import open3d as o3d
import numpy as np
from sync_files import read_scans

class Instance:
    def __init__(self,idx,cloud,label,score):
        self.idx = idx
        self.label = label
        self.score = score
        self.cloud = cloud
    def load_box(self,box):
        self.box = box

def load_scene_graph(folder_dir,min_points=1000):
    # load scene graph
    graph = {}
    with open(os.path.join(folder_dir,'instance_info.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            label_score_vec = parts[1].split('(')
            label = label_score_vec[0]
            score = float(label_score_vec[1].split(')')[0])
            # print('load {}:{}, {}'.format(idx,label,score))
            
            cloud = o3d.io.read_point_cloud(os.path.join(folder_dir,'{}.ply'.format(parts[0])))
            inst_toadd = Instance(idx,cloud,label,score)
            graph[idx] = inst_toadd

        f.close()
        print('Load {} instances '.format(len(graph)))
    
    # load instance boxes
    with open(os.path.join(folder_dir,'instance_box.txt')) as f:
        count=0
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            center = np.array([float(x) for x in parts[1].split(',')])
            rotation = np.array([float(x) for x in parts[2].split(',')])
            extent = np.array([float(x) for x in parts[3].split(',')])
            o3d_box = o3d.geometry.OrientedBoundingBox(center,rotation.reshape(3,3),extent)
            o3d_box.color = (0,0,0)
            
            graph[idx].load_box(o3d_box)
            count+=1
        f.close()
        print('load {} boxes'.format(count))
        
    return graph



def compute_cloud_overlap(cloud_a:o3d.geometry.PointCloud,cloud_b:o3d.geometry.PointCloud,search_radius=0.2):
    # compute point cloud overlap 
    Na = len(cloud_a.points)
    Nb = len(cloud_b.points)
    cloud_a_occupied = np.zeros((Na,1))
    pcd_tree_b = o3d.geometry.KDTreeFlann(cloud_b)
    
    for i in range(Na):
        [k,idx,_] = pcd_tree_b.search_radius_vector_3d(cloud_a.points[i],search_radius)
        if k>1:
            cloud_a_occupied[i] = 1
    iou = np.sum(cloud_a_occupied)/(Na+Nb-np.sum(cloud_a_occupied))
    return iou

def find_association(src_graph:dict[int,Instance],tar_graph:dict[int,Instance]):
    # find association
    Nsrc = len(src_graph)
    Ntar = len(tar_graph)
    iou = np.zeros((Nsrc,Ntar))
    MIN_IOU = 0.5
    SEARCH_RADIUS = 0.2
    assignment = np.zeros((Nsrc,Ntar),dtype=np.int32)

    src_graph_list = [src_idx for src_idx,_ in src_graph.items()]
    tar_graph_list = [tar_idx for tar_idx,_ in tar_graph.items()]
    
    # calculate iou
    for row_,src_idx in enumerate(src_graph_list):
        src_inst = src_graph[src_idx]
        for col_, tar_idx in enumerate(tar_graph_list):
            tar_inst = tar_graph[tar_idx]
            iou[row_,col_] = compute_cloud_overlap(src_inst.cloud,tar_inst.cloud,search_radius=SEARCH_RADIUS)
    
    # find match 
    row_maximum = np.zeros((Nsrc,Ntar),dtype=np.int32)
    col_maximum = np.zeros((Nsrc,Ntar),dtype=np.int32)
    row_maximum[np.arange(Nsrc),np.argmax(iou,1)] = 1 # maximum match for each row
    col_maximum[np.argmax(iou,0),np.arange(Ntar)] = 1 # maximu match for each column
    assignment = row_maximum*col_maximum # maximum match for each row and column
    
    # filter
    valid_assignment = iou>MIN_IOU
    assignment = assignment*valid_assignment
    
    #
    matches = np.argwhere(assignment==1)
    matches = [(src_graph_list[match[0]],tar_graph_list[match[1]]) for match in matches]
    # print(assignment)
    print(matches)
    
    return matches

def get_geometries(graph:dict[int,Instance],translation=np.array([0,0,0])):
    geometries = []
    for idx, instance in graph.items():
        cloud = instance.cloud.translate(translation)
        box = instance.box.translate(translation)
        geometries.append(cloud)
        geometries.append(box)
    return geometries

def get_match_lines(src_graph:dict[int,Instance],tar_graph:dict[int,Instance],matches,translation=np.array([0,0,0])):
    lines = []
    for match in matches:
        src_idx = match[0]
        tar_idx = match[1]
        src_inst = src_graph[src_idx]
        tar_inst = tar_graph[tar_idx]
        src_center = src_inst.box.get_center()
        tar_center = tar_inst.box.get_center()+translation
        # print(src_center.transpose(), tar_center.transpose())
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((src_center,tar_center))),
            lines=o3d.utility.Vector2iVector(np.array([[0,1]])))
        line_set.paint_uniform_color((0,1,0))
        lines.append(line_set)
        
    return lines

if __name__ == '__main__':
    dataroot = '/media/lch/SeagateExp/dataset_/ScanNet'
    graphroot = '/media/lch/SeagateExp/dataset_/ScanNetGraph'
    split= 'train'
    split_file = 'train_mini'
    Z_OFFSET = 5
    MIN_MATCHES = 3
    output_folder= os.path.join(graphroot,'matches')
    scans = read_scans(os.path.join(dataroot, 'splits', split_file + '.txt'))
    valid_scans = {}
    
    for scan in scans:
        scan_dir = os.path.join(graphroot,split,scan)
        if os.path.exists(os.path.join(scan_dir+'a','instance_box.txt'))==False: continue
        print('processing {}'.format(scan))
        
        # load
        src_graph = load_scene_graph(scan_dir+'a')
        tar_graph = load_scene_graph(scan_dir+'b')
        
        # find association
        matches = find_association(src_graph,tar_graph)
        match_lines = get_match_lines(src_graph,tar_graph,matches,np.array([0,0,Z_OFFSET]))
        if len(matches)>MIN_MATCHES: valid_scans[scan] = len(matches)
        
        # save matches
        with open(os.path.join(dataroot,'output_new','matches',scan+'.txt'),'w') as f:
            f.write('# src_instance_id, tar_instance_id\n')
            for match in matches:
                f.write('{};{}\n'.format(match[0],match[1]))
            f.close()
        continue
        
        # visualize
        viz_geometries = get_geometries(src_graph)
        for gem in get_geometries(tar_graph,np.array([0,0,Z_OFFSET])):
            viz_geometries.append(gem)
        viz_geometries.extend(match_lines)
        
        o3d.visualization.draw_geometries(viz_geometries,scan)
        # break    

    # exit(0)
    # save valid scans
    print('find {} pairs of valid scans'.format(len(valid_scans)))
    with open(os.path.join(graphroot, 'splits', split_file + '.txt'),'w') as f:
        for scan, matches_num in valid_scans.items():
            f.write('{}\n'.format(scan))
            print('{}:{}'.format(scan,matches_num))
        f.close()

