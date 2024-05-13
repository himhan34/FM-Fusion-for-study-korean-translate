import os, glob
import argparse, csv
import open3d as o3d
import numpy as np
from generate_gt_association import read_scans, process_scene, get_geometries, get_match_lines


def load_matches(dir):
    with open(dir, 'r') as f:
        reader = csv.reader(f)
        matches = []
        for row,line in enumerate(reader):
            if row==0:continue
            matches.append(np.array((int(line[0]),int(line[1]))))     
        matches = np.array(matches)   
        return matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT association and visualize(optional) for ScanNet')
    parser.add_argument('--graphroot', type=str, default='/media/lch/SeagateExp/dataset_/ScanNetGraph')
    parser.add_argument('--prediction', type=str, default='pgnn', help='Prediction folder under graph root')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--Z_OFFSET', type=float, default=5,help='offset the target graph for better viz')
    parser.add_argument('--viz',action='store_true')
    parser.add_argument('--correspondences',action='store_true')
    parser.add_argument('--points_similarity',action='store_true')
    args = parser.parse_args()
    
    gt_folder= os.path.join(args.graphroot,'matches')
    scans = read_scans(os.path.join(args.graphroot, 'splits', 'val' + '.txt'))
    sample_scans = np.random.choice(scans,args.samples)    # randomly sample 10 scans
    
    print('Visualize prediction for {} pairs of scans'.format(len(scans)))
    # sample_scans = ['scene0552_01']
    num_pos = 0
    num_neg = 0
    num_gt = 0
    num_scans = 0
    num_inliners = 0
    num_outliers = 0
    
    for scan in sample_scans:
        scan_dir = os.path.join(args.graphroot,args.split,scan)
        pred_folder = os.path.join(args.graphroot,'pred',args.prediction,'{}'.format(scan))
        if os.path.exists(pred_folder):
            out = process_scene(scan_dir,None,0.5,False,False)
            src_graph, tar_graph = out['src'], out['tar']
            gt = load_matches(os.path.join(gt_folder,scan+'.csv'))
            print('gt: ',gt)
            
            num_gt += gt.shape[0]
            pred = np.loadtxt(os.path.join(pred_folder,'instances.txt'),delimiter=',')
            pred_list = [pair for pair in pred.astype(np.int32)]
            print('pred: ',pred_list)
            
            # visualize
            src_geometries = get_geometries(src_graph,include_edges=False)
            tar_geometries = get_geometries(tar_graph,np.array([0,0,args.Z_OFFSET]),include_edges=False)
            map_geometries = src_geometries + tar_geometries
            viz_geometries = map_geometries
            
            pred_lines = get_match_lines(src_graph['nodes'],tar_graph['nodes'],pred_list)
            tp_msg = 'true positive: '
            fp_msg = 'false positive: '
            for pair,pair_line in zip(pred_list,pred_lines):
                # tmp = np.sum(gt-pair,axis=1)
                if pair[0] not in src_graph['nodes'] or pair[1] not in tar_graph['nodes']:
                    continue
                inst_a = src_graph['nodes'][pair[0]]
                inst_b = tar_graph['nodes'][pair[1]]
                check_equal = (gt-pair)==0
                
                if np.sum(check_equal,axis=1).max()==2:
                    # print('tp:{}'.format(pair))
                    pair_line.paint_uniform_color([0,1,0])
                    num_pos +=1
                    tp_msg += '({}_{},{}_{})'.format(pair[0],inst_a.label,pair[1],inst_b.label)
                elif inst_a.label=='floor' or inst_a.label=='carpet':
                    num_pos +=1
                    tp_msg += '({}_{},{}_{})'.format(pair[0],inst_a.label,pair[1],inst_b.label)
                else:
                    num_neg +=1
                    fp_msg += '({}_{},{}_{})'.format(pair[0],inst_a.label,pair[1],inst_b.label)
                    pair_line.paint_uniform_color([1,0,0])
                viz_geometries.append(pair_line)
            print(tp_msg)
            print(fp_msg)
            
            if args.correspondences:
                correspondences = np.load(os.path.join(pred_folder,'pred_pts.npy')) # (Npos,6), [i_m,src_idx,tar_idx,pts_u,pts_v,score]
                im_list = np.unique(correspondences[:,0])
                min_score = 1e-6
                INLIER_RADIUS = 0.2
                MAX_LINES = 20
                assert correspondences.shape[1] == 6
                for i_m in im_list:
                    pair_mask = correspondences[:,0]==i_m
                    src_node_id = correspondences[pair_mask,1][0]
                    tar_node_id = correspondences[pair_mask,2][0]
                    pts_uv = correspondences[pair_mask,3:5].astype(np.int32) # (Npos,2), [u,v]
                    pts_scores = correspondences[pair_mask,5] # (Npos,)
                    valid = pts_scores>min_score
                    pts_uv = pts_uv[valid,:]
            
                    src_cloud_pts = np.asarray(src_graph['nodes'][src_node_id].cloud.points)
                    tar_cloud_pts = np.asarray(tar_graph['nodes'][tar_node_id].cloud.points)
                    src_inst_color = np.asarray(src_graph['nodes'][src_node_id].cloud.colors)[0]
                    src_label = src_graph['nodes'][src_node_id].label
                    tar_label = tar_graph['nodes'][tar_node_id].label
                    if src_label=='floor' or src_label=='carpet': continue
                    max_lines = min(pts_uv.shape[0],MAX_LINES)
                    
                    for i in range(max_lines):
                        src_pt = src_cloud_pts[pts_uv[i,0]]
                        tar_pt = tar_cloud_pts[pts_uv[i,1]]
                        dist = np.linalg.norm(src_pt-(tar_pt-np.array([0,0,args.Z_OFFSET])))
                        if dist>INLIER_RADIUS:
                            num_outliers +=1
                        else:
                            num_inliners +=1
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector(np.array([src_pt,tar_pt]))
                        line.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
                        line.paint_uniform_color(src_inst_color)
                        viz_geometries.append(line)
            
            o3d.visualization.draw_geometries(viz_geometries,scan)

            if args.points_similarity:
                idxs = np.load(os.path.join(pred_folder,'idxs.npy')) # ((P0+P1+..Pm)xNpos, 5), [m,src_idx,tar_idx,src_pt_idx,tar_pt_idx]
                scores = np.load(os.path.join(pred_folder,'scores.npy')) # ((P0+P1+..Pm)xNpos, Npos)
                sim_geometries = []
                for cloud in map_geometries:
                    if not isinstance(cloud,o3d.geometry.PointCloud):
                        continue
                    instance_cloud = o3d.geometry.PointCloud(cloud)
                    instance_cloud.translate(np.array([10,0,0]))
                    cloud.paint_uniform_color([0.9,0.9,0.9])
                    sim_geometries.append(instance_cloud)
                    sim_geometries.append(cloud)
                assert idxs.shape[1] == 5
                
                # i_m = idxs[np.random.choice(idxs.shape[0],1),0]
                gt_list = np.unique(idxs[:,0])
                for i_m in gt_list:
                    pair_mask = idxs[:,0]==i_m
                    src_node_id = idxs[pair_mask,1][0]
                    tar_node_id = idxs[pair_mask,2][0]
                    pts_uv = idxs[pair_mask,3:5] # (Npos,2), [u,v]
                    pair_scores = scores[pair_mask,:] # (Npos,Npos)
                    Npos = pts_uv.shape[0]
                    src_cloud = src_graph['nodes'][src_node_id].cloud
                    tar_cloud = tar_graph['nodes'][tar_node_id].cloud
                    
                    anchor_id = np.random.choice(Npos,1).squeeze()
                    anchor_point = np.asarray(src_cloud.points)[pts_uv[anchor_id,0]]
                    anchor_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    anchor_sphere.compute_vertex_normals()
                    anchor_sphere.paint_uniform_color([1,0,0])
                    anchor_sphere.translate(anchor_point)
                    
                    anchor_scores = pair_scores[anchor_id,:] # (Npos,)
                    anchor_match_points = np.asarray(tar_cloud.points)[pts_uv[:,1]] # (Npos,3)
                    assert anchor_scores.shape[0]==anchor_match_points.shape[0]
                    print('score range: [{},{}], {} points'.format(anchor_scores.min(),anchor_scores.max(),pair_scores.shape[0]))
                    anchor_scores = np.exp(anchor_scores+1e-6)
                    anchor_scores /= anchor_scores.max()
                    print('expand score range: [{},{}]'.format(anchor_scores.min(),anchor_scores.max()))
                    anchor_match_colors = np.concatenate((anchor_scores.reshape(Npos,1),np.zeros((Npos,1)),np.ones((Npos,1))),axis=1) # (Npos,3)   
                    match_cloud = o3d.geometry.PointCloud()
                    match_cloud.points = o3d.utility.Vector3dVector(anchor_match_points)
                    match_cloud.colors = o3d.utility.Vector3dVector(anchor_match_colors)
                    # match_cloud.paint_uniform_color([0,0,1])
                    
                    max_score_id = np.argmax(anchor_scores)
                    max_score_point = anchor_match_points[max_score_id]
                    max_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    max_sphere.compute_vertex_normals()
                    max_sphere.paint_uniform_color([0,1,0])
                    max_sphere.translate(max_score_point)
                    
                    sim_geometries.append(anchor_sphere)
                    sim_geometries.append(max_sphere)
                    sim_geometries.append(match_cloud)
                    # print('correspondence range [{},{}],[{},{}]'.format(
                    #     pts_uv[:,0].min(),pts_uv[:,0].max(),pts_uv[:,1].min(),pts_uv[:,1].max()))
                    
                    # break
                o3d.visualization.draw_geometries(sim_geometries,scan)
                
            num_scans +=1
            # break
    #
    print('summarize {} scans'.format(num_scans))
    print('recall: {}/{}, {:.3f}'.format(num_pos,num_gt,num_pos/num_gt))
    print('precision: {}/{}, {:.3f}'.format(num_pos,num_pos+num_neg,num_pos/(num_pos+num_neg+1e-6)))
    
    if num_inliners>0:
        print('inliner: {}/{}, {:.3f}'.format(num_inliners,num_inliners+num_outliers,num_inliners/(num_inliners+num_outliers+1e-6)))
    