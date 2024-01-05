import os, glob
import argparse, csv
import open3d as o3d
import numpy as np
from generate_gt_association import read_scans, process_scene, get_geometries, get_match_lines, load_scene_pairs


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
    args = parser.parse_args()
    
    gt_folder= os.path.join(args.graphroot,'matches')
    scans = read_scans(os.path.join(args.graphroot, 'splits', 'val' + '.txt'))
    sample_scans = np.random.choice(scans,args.samples)    # randomly sample 10 scans
    
    print('Visualize prediction for {} pairs of scans'.format(len(scans)))
    # scans = ['scene0356_00']
    num_pos = 0
    num_neg = 0
    num_gt = 0
    num_scans = 0
    
    for scan in sample_scans:
        scan_dir = os.path.join(args.graphroot,args.split,scan)
        pred_dir = os.path.join(args.graphroot,'pred',args.prediction,'{}.txt'.format(scan))
        if os.path.exists(pred_dir):
            src_graph, tar_graph = load_scene_pairs(scan_dir,0.5)
            gt = load_matches(os.path.join(gt_folder,scan+'.csv'))
            print('gt: ',gt)
            # data = process_scene(scan_dir,os.path.join(output_folder,scan+'.csv'),False)
            
            # break
            # gt = [np.array((pair)) for pair in data['matches']]
            # gt = np.array(gt)
            num_gt += gt.shape[0]
            pred = np.loadtxt(pred_dir,delimiter=',')
            pred_list = [pair for pair in pred.astype(np.int32)]
            print('pred: ',pred_list)
            
            # visualize
            viz_geometries = get_geometries(src_graph,include_edges=False)
            for gem in get_geometries(tar_graph,np.array([0,0,args.Z_OFFSET]),include_edges=False):
                viz_geometries.append(gem)
            
            pred_lines = get_match_lines(src_graph['nodes'],tar_graph['nodes'],pred_list)
            tp_msg = 'true positive: '
            fp_msg = 'false positive: '
            for pair,pair_line in zip(pred_list,pred_lines):
                # pair_line.paint_uniform_color([1,0,0])
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
            o3d.visualization.draw_geometries(viz_geometries,scan)
            num_scans +=1
            break
    #
    print('summarize {} scans'.format(num_scans))
    print('recall: {}/{}, {:.3f}'.format(num_pos,num_gt,num_pos/num_gt))
    print('precision: {}/{}, {:.3f}'.format(num_pos,num_pos+num_neg,num_pos/(num_pos+num_neg)))