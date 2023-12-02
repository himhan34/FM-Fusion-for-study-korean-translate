import os, sys
import argparse
import open3d as o3d
import cv2 
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import csv
from sync_files import read_scans


class Instance:
    def __init__(self,idx,cloud,label,score):
        self.idx = idx
        self.label = label
        self.score = score
        self.cloud = cloud
    def load_box(self,box):
        self.box = box
    def record_cloud_dir(self,cloud_dir):
        self.cloud_dir = cloud_dir # realative path to the scan folder

def load_scene_graph(folder_dir,min_points=1000):
    ''' graph: {'nodes':{idx:Instance},'edges':{idx:idx}}
    '''
    # load scene graph
    nodes = {}
    edges = []
    invalid_nodes = []
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
            inst_toadd.cloud_dir = '{}.ply'.format(parts[0])
            nodes[idx] = inst_toadd

        f.close()
        print('Load {} instances '.format(len(nodes)))
    
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
            if'nan' in line:invalid_nodes.append(idx)

            nodes[idx].load_box(o3d_box)
            count+=1
        f.close()
        print('load {} boxes'.format(count))
    
    # remove invalid nodes
    for idx in invalid_nodes:
        nodes.pop(idx)
        
    return {'nodes':nodes,'edges':edges}

def generate_edges(name:str,graph:dict):
    n = len(graph['nodes'])
    searched_inst = []
    NEIGHBOR_RATIO = 2.0
    floors = []
    floors_names = ['floor','carpet']
    
    # between objects
    for i,inst_i in graph['nodes'].items():
        searched_inst.append(i)
        radius_a = LA.norm(inst_i.box.extent)/2
        if inst_i.label=='wall': radius_a = 0.1
        elif inst_i.label in floors_names: 
            floors.append(i)
            continue
        for j, inst_j in graph['nodes'].items():
            if j in searched_inst: continue
            radius_b = LA.norm(inst_j.box.extent)/2
            if inst_j.label=='wall': radius_b = 0.1
            elif inst_j.label in floors_names: continue
            radius = max(radius_a,radius_b)
            dist = LA.norm(inst_i.box.center - inst_j.box.center)
            if dist<radius*NEIGHBOR_RATIO:
                graph['edges'].append((i,j))
            
            # pass

    # object->floor
    for i, inst_i in graph['nodes'].items():
        if i in floors: continue
        min_dist = 100.0
        closet_floor = None
        
        for j in floors: # find the closet floor
            dist = LA.norm(inst_i.box.center - graph['nodes'][j].box.center)
            if dist<min_dist: 
                min_dist = dist
                closet_floor = j

        # add edge
        if closet_floor is not None:
            graph['edges'].append((i,closet_floor))


    print('Extract {} edges for {} graph'.format(len(graph['edges']),name))
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

def find_association(src_graph:dict,tar_graph:dict):
    # find association
    Nsrc = len(src_graph)
    Ntar = len(tar_graph)
    if Nsrc==0 or Ntar==0: return []
    
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

def get_geometries(graph:dict,translation=np.array([0,0,0]),include_edges=True):
    geometries = []
    STURCTURE_LABELS = ['floor','carpet']
    
    # nodes
    for idx, instance in graph['nodes'].items():
        cloud = instance.cloud.translate(translation)
        box = instance.box.translate(translation)
        geometries.append(cloud)
        geometries.append(box)
    
    # edges
    if include_edges:
        for edge in graph['edges']:
            src_idx = edge[0]
            tar_idx = edge[1]
            src_inst = graph['nodes'][src_idx]
            tar_inst = graph['nodes'][tar_idx]
            src_center = src_inst.box.get_center()
            tar_center = tar_inst.box.get_center()
            # print(src_center.transpose(), tar_center.transpose())
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack((src_center,tar_center))),
                lines=o3d.utility.Vector2iVector(np.array([[0,1]])))
            line_set.paint_uniform_color((1,0,0))
            if src_inst.label in STURCTURE_LABELS or tar_inst.label in STURCTURE_LABELS:
                line_set.paint_uniform_color((0,0,1))
            
            geometries.append(line_set)    
    
    return geometries

def get_instances_color(graph:dict):
    N = len(graph['nodes'])
    cell_width = 300
    cell_height = 80
    row_cells =  5
    num_rows = N//row_cells + 1
    color_img = 255*np.ones((num_rows *cell_height, row_cells * cell_width,3),dtype=np.uint8)
    count = 0
    print('{} nodes have {},{}'.format(N,num_rows,row_cells))
    
    for _, instance in graph['nodes'].items():
        row = count//row_cells
        col = count%row_cells
        color_img[row*cell_height:row*cell_height+cell_height,col*cell_width:col*cell_width+cell_width,:] = 255 * np.asarray(instance.cloud.colors)[0,:]
        cv2.putText(color_img,instance.label,(col*cell_width+10,row*cell_height+cell_height//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),2)
        # print('{}-{},{}-{}'.format(count, instance.label,row,col))
        count += 1
        
    color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)
    return color_img


def get_match_lines(src_graph:dict,tar_graph:dict,matches,translation=np.array([0,0,0])):
    lines = []
    for match in matches:
        src_idx = match[0]
        tar_idx = match[1]
        if src_idx not in src_graph or tar_idx not in tar_graph: continue
        assert src_idx in src_graph, '{} not in src graph'.format(src_idx)
        assert tar_idx in tar_graph, '{} not in tar graph'.format(tar_idx)
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

def save_nodes_edges(graph:dict,output_dir:str):
    
    with open(os.path.join(output_dir,'nodes.csv'),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=['node_id','label','score','center','quaternion','extent','cloud_dir'])
        writer.writeheader()
        for idx,inst in graph['nodes'].items():
            quat = R.from_matrix(inst.box.R.copy()).as_quat()
            label_str = inst.label
            if ' ' in label_str:
                label_str = label_str.replace(' ','_')
            writer.writerow({'node_id':idx,'label':label_str,'score':inst.score,
                            'center':np.array2string(inst.box.center,precision=6,separator=',')[1:-1],
                            'quaternion':np.array2string(quat,precision=6,separator=',')[1:-1],
                            'extent':np.array2string(inst.box.extent,precision=6,separator=',')[1:-1],
                            'cloud_dir':inst.cloud_dir})
        csvfile.close()
        
    with open(os.path.join(output_dir,'edges.csv'),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=['src_id','tar_id'])
        writer.writeheader()
        for edge in graph['edges']:
            writer.writerow({'src_id':edge[0],'tar_id':edge[1]})
        csvfile.close()

def process_scene(scan_dir,out_file_dir,save_feats=True, compute_matches=True):
    # scan_dir,out_file_dir = args
    if os.path.exists(os.path.join(scan_dir+'a','instance_box.txt'))==False or os.path.exists(os.path.join(scan_dir+'b','instance_box.txt'))==False:
        return None
    scan = os.path.basename(scan_dir)
    print('processing {}'.format(scan))
    
    # load
    src_graph = load_scene_graph(scan_dir+'a')
    tar_graph = load_scene_graph(scan_dir+'b')
    src_graph = generate_edges(scan+'a',src_graph)
    tar_graph = generate_edges(scan+'b',tar_graph)
    
    # find association
    if compute_matches:
        matches = find_association(src_graph['nodes'],tar_graph['nodes']) # [(src_idx, tar_idx)]
    else:
        matches = None
    
    # Save Graph: nodes.csv and edges.csv
    if save_feats:
        save_nodes_edges(src_graph,scan_dir+'a')
        save_nodes_edges(tar_graph,scan_dir+'b')
    
    # Save GT: matches.csv
    if out_file_dir is not None and len(matches)>0:
        with open(out_file_dir,'w',newline='') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=['src_id','tar_id'])
            writer.writeheader()
            for match in matches:
                if match[0] in src_graph['nodes'] and match[1] in tar_graph['nodes']:
                    writer.writerow({'src_id':match[0],'tar_id':match[1]})
            csvfile.close()

    return {'scan':scan,'src':src_graph,'tar':tar_graph,'matches':matches}

def load_scene_pairs(scan_dir):
    out = process_scene(scan_dir,None,False,False)
    return (out['src'],out['tar'])

def process_scene_thread(args):
    scan_dir,out_file_dir = args
    ret = process_scene(scan_dir,out_file_dir)
    if ret is None: return None
    else:
        return (ret['scan'],ret['matches'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT association and visualize(optional) for ScanNet')
    parser.add_argument('--dataroot', type=str, default='/data2/ScanNet')
    parser.add_argument('--graphroot', type=str, default='/data2/ScanNetGraph')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--split_file', type=str, default='val_clean')
    parser.add_argument('--Z_OFFSET', type=float, default=5,help='offset the target graph for better viz')
    parser.add_argument('--min_matches', type=int, default=4,help='minimum number of matches to be considered as valid')
    parser.add_argument('--debug_mode', action='store_true',help='visualize each pair of graph')
    parser.add_argument('--viz_edges',action='store_true',help='visualize edges')
    parser.add_argument('--viz_match', action='store_true',help='visualize gt matches')
    parser.add_argument('--viz_colorbar',action='store_true',help='visualize colorbar')
    args = parser.parse_args()
    
    output_folder= os.path.join(args.graphroot,'matches')
    scans = read_scans(os.path.join(args.graphroot, 'splits', args.split_file + '.txt'))
    print('Generate GT and graph for {} pairs of scans'.format(len(scans)))
    
    if args.debug_mode:
        for scan in scans:
            scan_dir = os.path.join(args.graphroot,args.split,scan)
            out = process_scene(scan_dir,os.path.join(output_folder,scan+'.csv'))
            if out is None or out['src'] is None: continue
            
            # visualize
            viz_geometries = get_geometries(out['src'],np.zeros(3),args.viz_edges)
            for gem in get_geometries(out['tar'],np.array([0,0,args.Z_OFFSET]),args.viz_edges):
                viz_geometries.append(gem)
            if args.viz_match:
                viz_geometries.extend(
                    get_match_lines(out['src']['nodes'],out['tar']['nodes'],out['matches']))
            if args.viz_colorbar: 
                color_img_a = get_instances_color(out['src'])
                color_img_b = get_instances_color(out['tar'])
                viz_img = np.vstack((color_img_b,color_img_a))
                cv2.imwrite('/home/lch/Downloads/colorbar.png'.format(scan),viz_img)
                # cv2.imshow('color',color_img)
                # cv2.waitKey(0)
                
            o3d.visualization.draw_geometries(viz_geometries,scan)
            # break    
        exit(0)

    import multiprocessing as mp
    p = mp.Pool(processes=64)
    outs = p.map(process_scene_thread,[(os.path.join(args.graphroot,args.split,scan),os.path.join(output_folder,scan+'.csv')) for scan in scans])    
    p.close()
    p.join()
    print('finished')
    
    exit(0)
    # save valid scans
    with open(os.path.join(args.graphroot, 'splits', args.split + '.txt'),'w') as f:
        for scan_out in outs:
            if scan_out is not None and len(scan_out[1])>=args.min_matches: 
                # scan, matches = scan_out
                f.write('{}\n'.format(scan_out[0]))
                print('{}:{}'.format(scan_out[0],len(scan_out[1])))
        f.close()

