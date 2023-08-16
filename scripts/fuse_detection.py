import os, glob,sys
import numpy as np
import open3d as o3d
import cv2
import json, argparse
from numpy import linalg as LA
from sklearn.preprocessing import normalize

import render_result
import project_util
import time

from prepare_data_inst import read_scannet_segjson

SEMANTIC_NAMES = render_result.SEMANTIC_NAMES
SEMANTIC_IDX = render_result.SEMANTIC_IDXS

class Detection:
    def __init__(self,u0,v0,u1,v1,labels):
        self.u0 = u0
        self.v0 = v0
        self.u1 = u1
        self.v1 = v1
        # self.label_name = label_name # string, semantic name
        # self.conf = conf
        self.labels = labels # K, {label:score}
        self.mask = None # H*W, bool
        self.assignment = ''
        self.valid = True
    
    def cal_centroid(self):
        centroid = np.array([(self.u0+self.u1)/2.0,(self.v0+self.v1)/2.0])
        return centroid.astype(np.int32)
    # def write_label_name(self,label_name):
    #     self.label_name = label_name
    def get_bbox(self):
        bbox = np.array([self.u0,self.v0,self.u1,self.v1])
        return bbox
    
    def add_mask(self,mask):
        self.mask = mask 
    def scanned_pts(self,iuv):
        '''
        Input:
            points_uv: N'*3, [i,u,v]
        Output:
            pt_indices: M, indices of points_uv that are inside the detection, M in [0,N)
        '''
        uv = iuv[:,1:3]
        if self.mask is None:
            valid = (uv[:,0] >= self.u0) & (uv[:,0] < self.u1) & (uv[:,1] >= self.v0) & (uv[:,1] < self.v1)
        else:
            valid = self.mask[uv[:,1],uv[:,0]]
        # print('detection {} has {} valid pts'.format(self.label,np.sum(valid)))
        pt_indices = iuv[valid,0]
        return pt_indices
    def get_label_str(self):
        msg = ''
        for name, score in self.labels.items():
            msg += '{}({:.2f}),'.format(name,score)        
        return msg
        
class Instance:
    def __init__(self,id, J=20):
        self.id = id
        self.os_labels = [] # list of openset labels; each openset label is a map of {label:score}
        self.prob_weight = 0
        self.prob_vector = np.zeros(J,dtype=np.float32) # probability for each nyu20 types, append 'unknown' at the end
        self.pos_observed = 0   # number of times been observed
        self.neg_observed = 0   # number of times been ignored
        self.points = np.array([]) # aggregate positive points 
        self.negative = np.array([]) # aggreate negative points
        self.filter_points= np.array([]) # points after filtering
        self.merged_points = np.array([]) # points after merging geometric segments
        self.centroid = np.zeros(3,dtype=np.float32) # 3D centroid
    
    def get_exist_confidence(self):
        return self.pos_observed/(self.pos_observed+self.neg_observed+1e-6)    
    
    def create_instance(self,observed_pt_indices,labels,fuse_scores=True):
        '''
        Input:
            - observed_pt_indices: (N'),np.int32, indices of points_uv that are inside the detection
        '''
        
        self.pos_observed += 1
        # update the openset labels
        if fuse_scores:
            det_labels = labels      
        else:
            max_label = ''
            max_score = 0.0
            for os_name, score in labels.items():
                if score>max_score:
                    max_score = score
                    max_label = os_name
            assert max_score>0.0, 'no valid openset label'
            det_labels = {max_label: 1.0}
            # self.os_labels[max_label] = [1.0]
        self.os_labels.append(det_labels)
        
        # update points
        self.points = np.array(observed_pt_indices)#.reshape(n,1)
        # print('create instance {} with {} points'.format(self.label,len(self.points)))
    
    def integrate_positive(self,observed_pt_indices,labels,fuse_scores=True):
        self.pos_observed += 1
        self.points = np.concatenate((self.points,observed_pt_indices),axis=0)
        
        if fuse_scores: # update all labels with scores
            self.os_labels.append(labels)        
        else: #update the label with max score
            max_label = ''
            max_score = 0.0
            for os_name, score in labels.items():
                if score>max_score:
                    max_score = score
                    max_label = os_name
            assert max_score>0.0, 'no valid openset label'
            self.os_labels.append({max_label:1.0})
            
    def integrate_negative(self,mask_map, points_uv, points_mask):
        '''
        Input:
            - mask_map: H*W, bool
            - points_uv: N*3, [i,u,v], not scanned points are set to -100
            - points_mask: N, bool
        '''
        unique_pts = np.unique(self.points) # (|P_m|,1)
        uv = points_uv[unique_pts,1:3].astype(np.int32) # (|P_m|,2)
        # assert uv[:,0].min()>=0 and uv[:,0].max()< mask_map.shape[1], 'not viewed points are not processed correct'
        # assert uv[:,0].min()>=0 and uv[:,0].max()< mask_map.shape[0], 'not viewed points are not processed correct'
        view_states = points_mask[unique_pts] # (|P_m|,), bool
        
        negative = np.zeros(unique_pts.shape,dtype=np.bool_) # init all to false
        negative[view_states] = np.logical_not(mask_map[uv[view_states,1],uv[view_states,0]]) # viewd and not in mask are True
        
        # negative = np.logical_not(mask_map[uv[:,1],uv[:,0]]) & view_states
        if negative.sum() > 0:
            self.negative = np.concatenate((self.negative,unique_pts[negative]),axis=0)
        
        # verify
        # check = np.isin(self.negative,self.points)
        # assert check.sum() == self.negative.shape[0], 'negative points not in the instance'        

    def get_points(self,min_fg=0.2,min_viewed=0,verbose=False):
        ''' Return the point indices of the instance
        '''
        M = self.points.max()+1
        indices = np.unique(self.points) # (M',), M'<=M
        assert indices.shape[0]>0, 'instance {} has no points'.format(self.id)

        pos_count, _ = np.histogram(self.points,bins=np.arange(M+1)) # (M)
        neg_count, _ = np.histogram(self.negative,bins=np.arange(M+1)) # (M)
        assert pos_count.shape[0] == M
        positive = pos_count[indices] # (M',)
        negative = neg_count[indices] # (M',)
        
        conf  = positive.astype(np.float32)/(positive+negative+1e-6)
        viewed = positive + negative
        valid = (conf >=min_fg) & (viewed >=min_viewed)
        # valid = (conf >= 0.2) & (positive>=2) # (M',)
        if verbose:
            print('instance {} has {}/{} valid points'.format(self.id,np.sum(valid),len(valid)))
        if valid.sum()<1:
            return np.array([])
        else:
            self.filter_points = indices[valid].astype(np.int32)          
            return indices[valid].astype(np.int32)     
    
    def update_label_probility(self, current_prob, score):
        self.prob_weight += score
        self.prob_vector += current_prob
        
    def merge_segment(self,segment):
        ''' Merge a segment to the instance
        Input:
            - segments: array of of point indices
        '''
        concat_points = np.concatenate((self.merged_points,segment),axis=0)
        self.merged_points = np.unique(concat_points)
        self.merged_points = self.merged_points.astype(np.int32)
    
    def merge_instance(self,other_instance):
        self.os_labels += other_instance.os_labels 
        self.prob_weight += other_instance.prob_weight
        self.prob_vector += other_instance.prob_vector
        self.pos_observed += other_instance.pos_observed
        self.neg_observed += other_instance.neg_observed
        
        self.points = np.concatenate((self.points,other_instance.points),axis=0)
        if other_instance.negative.size>3:
            self.negative = np.concatenate((self.negative,other_instance.negative),axis=0)
    
    def estimate_label(self):
        normalized_prob = self.get_normalized_probability()
        best_label_id = np.argmax(normalized_prob)
        conf = normalized_prob[best_label_id]
        return best_label_id, conf

    def estimate_top_labels(self,k=1):
        top_labels = []
        normalized_prob = self.get_normalized_probability()
        top_k = np.argsort(normalized_prob)[-k:]
        return top_k, normalized_prob[top_k]

    def get_normalized_probability(self):
        probability_normalized = self.prob_vector/(self.prob_vector.sum()+1e-6) # (LA.norm(self.prob_vector)+1e-6)
        return probability_normalized

class LabelFusion:
    def __init__(self,dir,fuse_all_tokens=True,propogate_method='mean'):
        # priors,association_count,likelihood,det_cond_probability,kimera_probability, openset_names, nyu20names = np.load(dir,allow_pickle=True)
        with open(os.path.join(dir,'label_names.json'),'r') as f:
            data = json.load(f)
            openset_names = data['openset_names']
            closet_names = data['closet_names']
        
        likelihood = np.load(os.path.join(dir,'likelihood.npy'))
        
        self.openset_names =  openset_names #[openset_id2name[i] for i in np.where(valid_rows)[0].astype(np.int32)]# list of openset names, (K,)
        self.closet_names = closet_names # In ScanNet, uses NYU20
        # self.priors = priors # (K,), np.float32
        self.likelihood = likelihood
        self.fuse_all_tokens = fuse_all_tokens
        self.propogate_method = propogate_method # mean, multiply
        
        print('{} valid openset names are {}'.format(len(self.openset_names),self.openset_names))
        if fuse_all_tokens:
            print('predictor fuse all token with scores!')

        # if self.use_baseline:
        #     print('-------- using empirical likelihood --------')

        assert len(self.closet_names) == self.likelihood.shape[1], 'likelihood and nyu20 must have same number of rows'
        assert len(self.openset_names) == self.likelihood.shape[0], 'likelihood and openset must have same number of columns,{}!={}'.format(
            len(self.openset_names),self.likelihood.shape[0])
        
    def bayesian_single_prob(self,openset_measurement):
        '''
        single-frame measurement, the measurements if a map of {name:score}
        '''
        prob = np.zeros(len(self.closet_names),np.float32)
        weight = 0.0
        labels = []
        scores = []
        assert isinstance(openset_measurement,dict)

        for zk,score in openset_measurement.items():
            if zk in self.openset_names:
                labels.append(self.openset_names.index(zk))
                scores.append(score)
                
        if len(labels)<1: return prob,weight
        
        labels = np.array(labels)
        scores = np.array(scores)
        
        for i in range(labels.shape[0]):
            prob_vector = self.likelihood[labels[i],:]
            prob_vector = prob_vector / np.sum(prob_vector)
            assert np.abs(prob_vector.sum() - 1.0) < 1e-6, '{}({}) prob vector is not normalized {}'.format(
                self.openset_names[labels[i]],labels[i],self.likelihood[labels[i],:])
            assert np.sum(prob_vector) > 1e-3, '{} prob vector is all zero'.format(self.openset_names[labels[i]])
            prob += scores[i] * prob_vector
            
        weight = 1.0    
        return prob,weight    
    
    def baseline_single_prob(self, openset_measurement):
        max_label =''
        max_score = 0.0
        prob = np.zeros(len(self.closet_names),np.float32)
        weight = 0.0

        for os_name, score in openset_measurement.items():
            if score>max_score:
                max_score = score
                max_label = os_name
        assert max_score>0.0, 'no valid openset label'
        
        if max_label in self.openset_names:
            prob_vector = self.likelihood[self.openset_names.index(max_label),:]
            prob_vector = prob_vector / np.sum(prob_vector)
            prob = max_score * prob_vector 
            weight = 1.0
        return prob, weight

    def estimate_single_prob(self, openset_measurement):
        if self.fuse_all_tokens:
            prob, weight = self.bayesian_single_prob(openset_measurement)
        else:
            prob, weight = self.baseline_single_prob(openset_measurement)
        return prob, weight

    def estimate_batch_prob(self,openset_measurements):
        '''·
        multiple-frame measurement, the measurements if a map of {name:[score1,score2,...]}
        '''
        
        prob = np.zeros((len(self.closet_names),),np.float32)
        weight = 0.0

        if self.propogate_method=='mean':
            for zk in openset_measurements:
                if self.fuse_all_tokens:
                    prob_k, weight_k = self.bayesian_single_prob(zk)
                else:
                    prob_k, weight_k = self.baseline_single_prob(zk)
                    
                if weight_k>1e-6:
                    prob += prob_k
                    weight += weight_k
        else:
            raise NotImplementedError                                
        # elif self.propogate_method=='multiply': #todo
        #     for zk, score in openset_measurements.items():
        #         prob *= self.estimate_current_prob(zk)
        #         weight += score
                       
        
        return prob, weight
    
    #todo: propogate along new measurements
    def update_probability(self,prob_vector,openset_measurement):
        fused_prob_vector = prob_vector
        return fused_prob_vector


class ObjectMap:
    def __init__(self,points,colors):
        self.points = points
        self.colors = colors
        self.instance_map = {}
        print('Init an instance map with {} points'.format(len(points)))
    
    def load_segments(self,fn3):
        '''
        fn3: scannet segmentation file xxxx_vh_clean_2.0.010000.segs.json
        '''
        assert os.path.exists(fn3), 'segmentation file {} does not exist'.format(fn3)
        segments = read_scannet_segjson(fn3)
        self.segments = [np.array(segpoints) for segid,segpoints in segments.items()]
        print('Load {} segments'.format(len(self.segments)))
    
    def load_semantic_names(self,labels):
        self.labels = labels
    
    def insert_instance(self,instance:Instance):
        self.instance_map[str(instance.id)] = instance
        
    def update_result_points(self,min_fg,min_viewed):
        for idx, instance in self.instance_map.items():
            instance.get_points(min_fg=min_fg,min_viewed=min_viewed)
            # print('instance {} has {} points'.format(idx,instance.filter_points.size))
    
    def extract_object_map(self, merged_results=False):
        instance_labels = np.zeros(len(self.points)) -100
        count = 0
        msg =''
        for instance_id,instance in self.instance_map.items():
            if merged_results: p_instance = instance.merged_points
            else:    p_instance = instance.filter_points

            label_id,conf = instance.estimate_label()
            
            if p_instance.size>0:
                composite_label = label_id*1000 + instance.id + 1
                instance_labels[p_instance] = composite_label #instance.id
                count +=1   
                msg += '{},'.format(self.labels[label_id])
            
        print('Extract {}/{} instances to visualize'.format(count,len(self.instance_map)))
        print('Instance labels are: {}'.format(msg))
        return instance_labels
    
    def save_debug_results(self,eval_dir, scene_name, mean_time=0.0):
        output_folder = os.path.join(eval_dir,scene_name)
        if os.path.exists(output_folder)==False:
            os.makedirs(output_folder)
            
        label_debug_file = open(os.path.join(output_folder,'fusion_debug.txt'),'w')
        label_debug_file.write('# mask_file label_id label_conf pos_observe neg_observe point_number label_name \n')
        count = 0
        for instance_id,instance in self.instance_map.items():
            # instance_mask = np.zeros(len(self.points),dtype=np.int32)
            pos_points = instance.points
            neg_points = instance.negative
            if pos_points.shape[0]>0:
                label_instance,label_conf = instance.estimate_label()
                # exist_conf = instance.get_exist_confidence()

                if label_instance <len(SEMANTIC_NAMES):
                    label_name = SEMANTIC_NAMES[label_instance]
                    label_nyu40 = SEMANTIC_IDX[label_instance]
                else:
                    label_name = 'openset'
                    label_nyu40 = 99
                
                mask_file = os.path.join(output_folder,'{}_{}'.format(instance.id,label_name))
                if label_name=='shower curtain':
                    mask_file = os.path.join(output_folder,'{}_shower_curtain'.format(instance.id))
                
                # pred_file.write('{} {} {:.3f}\n'.format(
                    # os.path.basename(mask_file),label_nyu40,label_conf))
                label_debug_file.write('{} {} {:.3f} {} {} {};'.format(
                    os.path.basename(mask_file),label_nyu40,label_conf, instance.pos_observed, instance.neg_observed,pos_points.shape[0]))
                for det in instance.os_labels:
                    for os_name, score in det.items():
                        label_debug_file.write('{}:{:.4f}_'.format(os_name,score))
                    label_debug_file.write(',')
                # for os_name, scores in instance.os_labels.items():
                #     label_debug_file.write('{}'.format(os_name))
                #     for score in scores: label_debug_file.write('_{:.3f}'.format(score))
                #     label_debug_file.write(',')
                label_debug_file.write('\n')
                np.savetxt('{}_pos.txt'.format(mask_file),pos_points,fmt='%d')
                if neg_points.shape[0]>3:
                    np.savetxt('{}_neg.txt'.format(mask_file),instance.negative,fmt='%d')
                count +=1
        label_debug_file.write('# {}/{} instances extracted'.format(count,len(self.instance_map)))
        label_debug_file.write('# mean data association time {:.3f} s'.format(mean_time))
        # pred_file.close()
        label_debug_file.close()

    def save_scannet_results(self, eval_dir, scene_name, merged_results=False):
        output_folder = os.path.join(eval_dir,scene_name)
        if os.path.exists(output_folder)==False:
            os.makedirs(output_folder)
        count = 0
        pred_file = open(os.path.join(output_folder,'predinfo.txt'),'w')
        for instance_id,instance in self.instance_map.items():
            instance_mask = np.zeros(len(self.points),dtype=np.int32)
            if merged_results:p_instance = instance.merged_points
            else:p_instance = instance.filter_points

            label_id,label_conf = instance.estimate_label()
            assert label_id<20, 'contain unexpected label id'.format(label_id)

            if p_instance.size>0 and label_id<20:
                # exist_conf = instance.get_exist_confidence()
                instance_mask[p_instance] = 1
                
                label_name = SEMANTIC_NAMES[label_id]
                label_nyu40 = SEMANTIC_IDX[label_id]
                mask_file = os.path.join(output_folder,'{}_{}.txt'.format(instance.id,label_name))

                if label_name=='shower curtain':
                    mask_file = os.path.join(output_folder,'{}_shower_curtain.txt'.format(instance.id))
                
                pred_file.write('{} {} {:.3f}\n'.format(os.path.basename(mask_file),label_nyu40,label_conf))
                np.savetxt(mask_file,instance_mask,fmt='%d')
                count +=1
        
        pred_file.close()
        print('Extract {}/{} instances to evaluate'.format(count,len(self.instance_map)))

    def extract_all_centroids(self):
        for idx, instance in self.instance_map.items():
            points = self.points[instance.get_points()]
            instance.centroid = np.mean(points,axis=0)
    
    def non_max_suppression(self,ious, scores, threshold):
        ixs = scores.argsort()[::-1]
        ixs_copy = ixs.copy()
        pick = []
        pairs = {}
        
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)
            iou = ious[i, ixs[1:]]
            remove_ixs = np.where(iou > threshold)[0] + 1
            
            if len(remove_ixs)>0:
                if str(i) not in pairs: pairs[str(i)] = ixs[remove_ixs]
                else: pairs[str(i)] = np.concatenate((pairs[str(i)],ixs[remove_ixs]),axis=0)    
                        
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)
            
            
        return np.array(pick,dtype=np.int32), pairs
    
    def merge_conflict_instances(self,nms_iou=0.1,nms_similarity=0.2):
        # SIMILARITY_THRESHOLD = 0.07
        J = len(self.instance_map)
        instance_indices = np.zeros((J,),dtype=np.int32) # (J,), indices \in [0,J) to instance idx
        proposal_points = np.zeros((J,self.points.shape[0]),dtype=np.int32) # (J,N)
        
        # Calculate iou
        i =0
        for idx, instance in self.instance_map.items():
            if instance.merged_points.size<1: continue
            proposal_points[i,instance.merged_points] = 1
            instance_indices[i] = int(idx)
            i+=1
            
        intersection = np.matmul(proposal_points,proposal_points.T) # (J,J)
        proposal_points_number = np.sum(proposal_points,axis=1)+1e-6 # (J,)
        proposal_pn_h = np.tile(proposal_points_number,(J,1)) # (J,J)
        proposal_pn_v = np.tile(proposal_points_number,(J,1)).T # (J,J)
        ious = intersection/(proposal_pn_h+proposal_pn_v-intersection) # (J,J)
        scores = proposal_points_number # (J,)
        
        # NMS
        pick_idxs, merge_groups = self.non_max_suppression(ious,scores,threshold=nms_iou) 
        for root_idx, leaf_idxs in merge_groups.items(): # merge supressed instances
            root_instance = self.instance_map[str(instance_indices[int(root_idx)])]
            for leaf_idx in leaf_idxs:
                assert leaf_idx not in pick_idxs, 'merged leaf instance must be removed'
                leaf_instance = self.instance_map[str(instance_indices[leaf_idx])]
                assert root_instance.merged_points.size >= leaf_instance.merged_points.size, '{} root instance must have more points'.format(scene_name)
                
                similarity = root_instance.get_normalized_probability().dot(leaf_instance.get_normalized_probability())
                if similarity>nms_similarity:
                    # print('Merge instance {} to {}'.format(leaf_instance.id,root_instance.id))
                    root_instance.merge_segment(leaf_instance.merged_points)
                    root_instance.prob_vector += leaf_instance.prob_vector
                else:
                    pick_idxs = np.concatenate((pick_idxs,np.array([leaf_idx],dtype=np.int32)),axis=0)
        print('{}/{} root instances are kept. Leaf instance is merged into the root one.'.format(len(pick_idxs),J))
        
        # Remove filtered instances
        for j in np.arange(J):
            instance_id = instance_indices[j]
            if j not in pick_idxs:
                del self.instance_map[str(instance_id)]
        
    def fuse_instance_segments(self, merge_types, min_segments = 200):
        segment_iou = 0.1 
        S = len(self.segments)
        J = len(self.instance_map)
        instance_indices = np.zeros((J,),dtype=np.int32) # (J,), indices \in [0,J) to instance idx
        iou = np.zeros((S,J),dtype=np.float32)
        # proposal_points = np.zeros((J,self.points.shape[0]),dtype=np.int32) # (J,N)

        # Find overlap between segment and instance
        merged_instances = []
        i_ = 0
        for idx, instance in self.instance_map.items():
            instance_points = instance.filter_points
            label_id, _ = instance.estimate_label()
            assert label_id < len(self.labels)
            # assert instance_points.size>0, 'instance {} has no points'.format(idx)

            if self.labels[label_id] in merge_types:
                for s, seg_points in enumerate(self.segments):
                    if seg_points.size<min_segments: continue
                    overlap = np.intersect1d(seg_points,instance_points)
                    iou[s,i_] = overlap.size/(seg_points.size) #+instance_points.size-overlap.size)
            else:
                instance.merge_segment(instance_points)
                merged_instances.append(int(idx))
            instance_indices[i_] = int(idx)
            i_+=1
            
        print('{}/{} overlaped pairs'.format(np.sum(iou>1e-3),iou.size))
        
        # Merge instance with segments
        count = 0
        for s, seg_points in enumerate(self.segments):
            parent_instances = iou[s,:]>segment_iou
            for parent_instance_idx in instance_indices[parent_instances]:
                root_instance = self.instance_map[str(parent_instance_idx)]
                root_instance.merge_segment(seg_points.astype(np.int32))
                count +=1
                if parent_instance_idx not in merged_instances:merged_instances.append(parent_instance_idx)
            
        print('{}/{} segments are merged into {}/{} instances'.format(
            count,len(self.segments),len(merged_instances),len(self.instance_map)))

        # Remove instance without valid geometric segments
        for j in np.arange(J):
            instance_id = instance_indices[j]
            if instance_id not in merged_instances:
                del self.instance_map[str(instance_id)]
        return None
    

    def remove_small_instances(self, min_points):
        
        remove_idx = []
        for idx, instance in self.instance_map.items():
            if instance.merged_points.size<min_points:
                remove_idx.append(idx)
                
        for idx in remove_idx:
            del self.instance_map[idx]
            # print('Remove instance {} with {} points'.format(idx,instance.filter_points.size))
        
        print('remove {} small instances'.format(len(remove_idx)))
             
    def get_num_instances(self):
        return len(self.instance_map)
    
def read_scans(dir):
    with open(dir,'r') as f:
        scans = []
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
        return scans

def find_overlap(z0,z1):
    '''
    Input: z0,z1: [u0,v0,u1,v1]
    Ouput: overlap ratio, z0 is smaller than z1
    '''
    bbox = np.zeros(4,dtype=np.int32)

    bbox[0] = max(z0[0],z1[0]) # tl
    bbox[1] = max(z0[1],z1[1])
    bbox[2] = min(z0[2],z1[2]) # br
    bbox[3] = min(z0[3],z1[3])
    
    tl = np.array([bbox[0],bbox[1]])
    br = np.array([bbox[2],bbox[3]])
    wh = np.clip(br-tl,a_min=0.0,a_max=None)
    
    overlap = wh[0]*wh[1] #(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    z0_area = (z0[2]-z0[0])*(z0[3]-z0[1])
    z1_area = (z1[2]-z1[0])*(z1[3]-z1[1])
    union = min(z0_area,z1_area)
    overlap = overlap/union
    
    return overlap, z0_area<z1_area

def load_pred(predict_folder, frame_name, write_label_id=False):
    '''
    Output: a list of detections
    '''
    label_file = os.path.join(predict_folder,'{}_label.json'.format(frame_name))
    mask_file = os.path.join(predict_folder,'{}_mask.npy'.format(frame_name))
    
    mask = np.load(mask_file) # (M,H,W), int, [0,1]
    img_height = mask.shape[1]
    img_width = mask.shape[2]
    detections = []
    # valid_detections = []
    labels_msg =''
    MAX_BOX_RATIO=0.95
    # print('pred image dimension: {}x{}'.format(img_width,img_height))
    
    with open(label_file, 'r') as f:
        json_data = json.load(f)
        tags = json_data['tags'] if 'tags' in json_data else None
        masks = json_data['mask']
        raw_tags = None
        invalid_detections = []
        
        if 'raw_tags' in json_data:
            raw_tags = json_data['raw_tags']
        for ele in masks:
            if 'box' in ele:
                # if label[-1]==',':label=label[:-1]
                instance_id = ele['value']-1    
                bbox = ele['box']  
                labels = ele['labels'] # {label:conf}

                if (bbox[2]-bbox[0])/img_width>MAX_BOX_RATIO and (bbox[3]-bbox[1])/img_height>MAX_BOX_RATIO:
                    continue # remove bbox that are too large
                z_ = Detection(bbox[0],bbox[1],bbox[2],bbox[3],labels)            
                z_.add_mask(mask[instance_id,:,:]==1)
                detections.append(z_)
                
            else: # background
                continue  
        f.close()
        
        # print('invalid :{}'.format(invalid_detections))
        detections = [detections[k] for k in range(len(detections)) if k not in invalid_detections]
        
        print('{}/{} detections are loaded in {}'.format(len(detections),len(masks)-1, frame_name))
        return tags, detections

def find_assignment(detections,instances,points_uv, min_iou=0.5,min_observe_points=200, verbose=False):
    '''
    Output: 
        - assignment: (K,2), [k,j] in matched pair. If j=-1, the detection is not matched
    '''
    K = len(detections)
    M = instances.get_num_instances()
    if M==0:
        assignment = np.zeros((K,1),dtype=np.int32) 
        return assignment, []
    
    # MIN_OBSERVE_POINTS = 200
    MIN_OBSERVE_NEGATIVE_POINTS = 2000

    # compute iou
    iou = np.zeros((K,M),dtype=np.float32)
    assignment = np.zeros((K,M),dtype=np.int32)
    # return assignment, []

    for k_,zk in enumerate(detections):
        uv_k = zk.mask # (H,W), bool, detection mask    
        for j_ in range(M):
            l_j = instances.instance_map[str(j_)]
            p_instance = l_j.get_points()

            # if l_j.label != zk.label or p_instance.shape[0]==0: continue
            
            uv_j = points_uv[p_instance,1:3].astype(np.int32)
            observed = np.logical_and(uv_j[:,0]>=0,uv_j[:,1]>=0)   
            if observed.sum() < min_observe_points: continue
            uv_j = uv_j[observed] # (|P_m|,2)

            uv_m = np.zeros(uv_k.shape,dtype=np.bool_)
            uv_m[uv_j[:,1],uv_j[:,0]] = True    # object mask
            if np.sum(uv_m)>0:
                overlap = np.logical_and(uv_k,uv_m)
                iou[k_,j_] = np.sum(overlap)/(np.sum(uv_m)) #+np.sum(uv_k)-np.sum(overlap))
                        
    # find assignment
    assignment[np.arange(K),iou.argmax(1)] = 1
    
    instances_bin = assignment.sum(1) > 1
    if instances_bin.any(): # multiple detections assigned to one instance
        iou_col_max = iou.max(0)
        valid_col_max = np.abs(iou - np.tile(iou_col_max,(K,1))) < 1e-6 # non-maximum set to False
        assignment = assignment & valid_col_max
        
    valid_match = (iou > min_iou).astype(np.int32)
    assignment = assignment & valid_match
    
    # fuse instance confidence
    missed = []
    for j_ in range(M):
        instance_j = instances.instance_map[str(j_)]
        p_instance = instance_j.get_points()
        if p_instance.shape[0]>0:
            uv_j = points_uv[p_instance,1:3].astype(np.int32)
            observed = np.logical_and(uv_j[:,0]>=0,uv_j[:,1]>=0)
            if np.sum(observed)>MIN_OBSERVE_NEGATIVE_POINTS and assignment[:,j_].max()==0:
                instance_j.neg_observed +=1
                missed.append(j_)
    
    if verbose:
        print('---iou----')
        print(iou)
        print('---assignment----')
        print(assignment)
    
    return assignment, missed

def process_scene(args):
    '''
    Args:
        - scene_dir: scene directory
        - dataroot: root directory of the dataset
        - method: method name
        - pred_folder: prediction folder, [prediction_no_augment, prediction_forward]
        - label_predictor: label predictor
        - visualize: bool, visualize the results
    '''
    scene_dir, dataroot, method, pred_folder, label_predictor,visualize = args
    scene_name = os.path.basename(scene_dir)
    
    if 'ScanNet' in dataroot:
        map_dir = os.path.join(scene_dir,'{}_{}'.format(scene_name,'vh_clean_2.ply'))
        DEPTH_SCALE = 1000.0
        RGB_FOLDER = 'color'
        RGB_POSFIX = '.jpg'
        DATASET = 'scannet'
        FRAME_GAP = 20
        MIN_VIEW_POINTS = 200
        MIN_OBJECT_POINTS =200
    elif 'tum' in dataroot:
        map_dir = os.path.join(scene_dir,'{}'.format('mesh_o3d256_ds.ply'))
        DEPTH_SCALE = 5000.0
        RGB_FOLDER = 'rgb'
        RGB_POSFIX = '.png'
        DATASET = 'tum'
        FRAME_GAP = 0.5
        MIN_VIEW_POINTS = 50
        MIN_OBJECT_POINTS =50
        
        data_association = {}
        with open(os.path.join(scene_dir,'data_association.txt'),'r') as f_association:
            for line in f_association.readlines():
                elems = line.strip().split(' ')
                depth_name = elems[0].split('/')[-1][:-4]
                rgb_name = elems[1].split('/')[-1][:-4]
                data_association[rgb_name] = depth_name
            f_association.close()
    elif 'scenenn' in dataroot:
        map_dir = os.path.join(scene_dir,'{}'.format('mesh_o3d256.ply'))
        DEPTH_SCALE = 1000.0
        RGB_FOLDER = 'image'
        RGB_POSFIX = '.png'
        DATASET = 'scenenn'
        FRAME_GAP = 20
        MIN_VIEW_POINTS = 200
        MIN_OBJECT_POINTS =200
    else:
        raise NotImplementedError
        
    # MAP_POSIX = 'vh_clean_2.ply'
    INTRINSIC_FOLDER ='intrinsic'
    PREDICTION_FOLDER = pred_folder

    MIN_FG = 0.2
    MIN_FG_VIEWED = 2
    MIN_IOU = 0.5

    # depth filter related params
    max_dist_gap = 0.2
    depth_kernal_size = 5
    kernal_valid_ratio = 0.2
    kernal_max_var = 0.15
    
    # Instances parameters
    if os.path.exists(os.path.join(scene_dir,'tmp'))==False:
        os.mkdir(os.path.join(scene_dir,'tmp'))
        
    # Init
    K_rgb, K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,INTRINSIC_FOLDER))
    rgb_out_dim = depth_dim
    K_rgb_out = project_util.adjust_intrinsic(K_rgb,rgb_dim,rgb_out_dim)
    predict_folder = os.path.join(scene_dir,PREDICTION_FOLDER)
    predict_frames =  glob.glob(os.path.join(predict_folder,'*_label.json'))  
    print('---- {}/{} find {} prediction frames'.format(DATASET,scene_name,len(predict_frames)))
        
    # Load dense map
    # map_dir = os.path.join(scene_dir,'{}_{}'.format(scene_name,MAP_POSIX))
    pcd = o3d.io.read_point_cloud(map_dir)
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)
    normals = np.tile(np.array([0,0,1]),(points.shape[0],1))
    N = points.shape[0]
    
    instance_map = ObjectMap(points,colors)
    instance_map.load_semantic_names(label_predictor.closet_names)
    print('load {} points'.format(N))
    
    # Integration
    prev_frame_stamp = 0
    time_array = np.array([])
    
    for i,pred_frame in enumerate(sorted(predict_frames)):   
        frame_name = os.path.basename(pred_frame).split('_')[0] 
        if DATASET=='scannet':
            frame_stamp = float(frame_name.split('-')[-1])
            depth_dir = os.path.join(scene_dir,'depth',frame_name+'.png')
        elif DATASET=='tum':
            frame_stamp = float(frame_name)
            if frame_name not in data_association: continue # some rgb frames in tum are skipped due to temporal gap
            depth_frame = data_association[frame_name]
            depth_dir = os.path.join(scene_dir,'depth',depth_frame+'.png')
            # print(depth_dir)
        elif DATASET =='scenenn':
            frame_stamp = float(frame_name[5:])
            depth_dir = os.path.join(scene_dir,'depth',frame_name.replace('image','depth')+'.png')
        
        if (frame_stamp-prev_frame_stamp) < FRAME_GAP: continue
        # if frameidx>300: break
        
        rgbdir = os.path.join(scene_dir,RGB_FOLDER,frame_name+RGB_POSFIX)
        pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        if os.path.exists(pose_dir)==False:
            print('no pose file for frame {}. Stop the fusion.'.format(frame_name))
            break

        # load rgbd, pose
        rgbimg = cv2.imread(rgbdir)
        raw_depth = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        depth = raw_depth.astype(np.float32)/DEPTH_SCALE
        assert raw_depth.shape[0]==depth_dim[0] and raw_depth.shape[1]==depth_dim[1], 'depth image dimension does not match'
        assert depth.shape[0] == rgbimg.shape[0] and depth.shape[1] == rgbimg.shape[1]
        T_wc = np.loadtxt(pose_dir)

        # projection
        mask, points_uv, _, _ = project_util.project(
            points, normals,T_wc, K_rgb_out, rgb_out_dim, 5.0, 0.5) # Nx3
        filter_mask = project_util.filter_occlusion(points_uv,depth,max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        
        # test_mask = points_uv[:,1] >= 0
        # assert mask.sum() == test_mask.sum()
        
        points_uv = np.concatenate([np.arange(N).reshape(N,1),points_uv],axis=1) # Nx4, np.double, [i,u,v,d]
        mask = np.logical_and(mask,filter_mask) # Nx1, bool
        points_uv[~mask] = np.array([-100,-100,-100,-100])
        iuv = points_uv[mask,:3].astype(np.int32)  # current viewed points
        uv_rgb = colors[mask,:3] * 255.0
        count_view_points = np.sum(mask)  
        if count_view_points < MIN_VIEW_POINTS:
            print('drop poor frame with {} valid points'.format(count_view_points))
            continue
        print('{}/{} viewed in frame {}'.format(count_view_points,points.shape[0],frame_name))
        
        # detections
        tags, detections = load_pred(predict_folder, frame_name)
    
        # continue
        if len(detections) == 0:
            print('no detection in current frame')
            continue
        
        # association
        t_start = time.time()
        assignment, missed = find_assignment(detections,instance_map,points_uv,min_iou=MIN_IOU,min_observe_points=MIN_OBJECT_POINTS,verbose=False)
        M = len(instance_map.instance_map)
        
        # integration
        for k,z in enumerate(detections):
            prob_current, weight = label_predictor.estimate_single_prob(z.labels)
            
            if weight<1e-6: continue
            if assignment[k,:].max()==0: # unmatched            
                new_instance = Instance(instance_map.get_num_instances(),J=prob_current.shape[0])
                new_instance.create_instance(z.scanned_pts(iuv),z.labels,True)
                if new_instance.points.shape[0] < MIN_OBJECT_POINTS: continue # 50
                new_instance.update_label_probility(prob_current,weight)
                instance_map.insert_instance(new_instance)
                z.assignment = 'new'
                
            else: # integration
                j = np.argmax(assignment[k,:])
                matched_instance = instance_map.instance_map[str(j)]
                matched_instance.update_label_probility(prob_current, weight)
                matched_instance.integrate_positive(z.scanned_pts(iuv),z.labels,True)
                matched_instance.integrate_negative(z.mask, points_uv, mask)
                z.assignment = 'match'
        t_end = time.time()
        time_array = np.append(time_array,t_end-t_start)
        # print('It takes {} ms for DA'.format((t_end-t_start)*1000))
        
        # visualize 
        if visualize:
            # prj_rgb = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
            # M_ = iuv.shape[0]
            # for m_ in range(M_):
            #     cv2.circle(prj_rgb,(int(iuv[m_,1]),int(iuv[m_,2])),1,(int(uv_rgb[m_,2]),int(uv_rgb[m_,1]),int(uv_rgb[m_,0])),1)

            # out = np.concatenate([rgbimg,prj_rgb],axis=1)
            # cv2.imwrite(os.path.join(scene_dir,'tmp','{}_rgb.jpg'.format(frame_name)),out)      
            # continue

            # visualize instance
            K = len(detections)
            proj_instances = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
            view_instances_centroid = {}
            
            for j_ in range(M):
                instance_j = instance_map.instance_map[str(j_)]
                uv_j = points_uv[instance_j.get_points(0.0001),:].astype(np.int32)
                observed = np.logical_and(uv_j[:,0]>=0,uv_j[:,1]>=0)
                if observed.sum() < 500:
                    continue
                instance_uv = uv_j[observed,:]
                # cv2.putText(proj_instances, str(np.sum(observed)), (int(instance_uv[0,2]),int(instance_uv[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                for q in np.arange(instance_uv.shape[0]):
                    iuv = instance_uv[q,:]
                    rgb = 255 * colors[iuv[0],:]
                    cv2.circle(proj_instances,(iuv[1],iuv[2]),3,(int(rgb[2]),int(rgb[1]),int(rgb[0])),1)
                centroid = instance_uv[:,1:].mean(axis=0)
                view_instances_centroid[j_] = centroid
                pred_label_id, conf = instance_j.estimate_label()
                semantic_label_name = SEMANTIC_NAMES[pred_label_id] if pred_label_id<len(SEMANTIC_NAMES) else 'openset'
                cv2.putText(proj_instances, semantic_label_name, (int(centroid[0]),int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if j_ in missed:
                    cv2.circle(proj_instances,(int(centroid[0]+10),int(centroid[1]+10)),8,(0,0,255),-1)
                else:
                    cv2.circle(proj_instances,(int(centroid[0]+10),int(centroid[1]+10)),8,(0,255,0),-1)
            
            out = np.concatenate([rgbimg,proj_instances],axis=1)
            
            for k_ in range(K):
                zk = detections[k_]
                # zk_centroid = zk.cal_centroid()
                prob, weight = label_predictor.estimate_single_prob(zk.labels)
                if weight <1e-6: continue
                if zk.assignment=='match': bbox_color = (0,255,0)
                elif zk.assignment=='new': bbox_color = (255,0,255)
                else: bbox_color = (0,0,255)
                cv2.rectangle(out,pt1=(int(zk.u0),int(zk.v0)),pt2=(int(zk.u1),int(zk.v1)),color=bbox_color,thickness=1)
                # cv2.circle(rgbimg,(centroid[1],centroid[0]),5,(0,0,255),1)
                cv2.putText(out, zk.get_label_str(), (int(zk.u0+10),int(zk.v0+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                j = np.argmax(assignment[k_,:])

                if assignment[k_,:].max()>0 and j in view_instances_centroid and zk.assignment=='match':
                #    print('match:{},{}'.format(k_,j))
                   j_centroid = view_instances_centroid[j]
                   cv2.line(out,(int(zk.cal_centroid()[0]),int(zk.cal_centroid()[1])),(int(j_centroid[0]+rgb_out_dim[1]),int(j_centroid[1])),(255,0,0),1)
            
            cv2.imwrite(os.path.join(scene_dir,'tmp','{}_instance.jpg'.format(frame_name)),out)
        
        prev_frame_stamp = frame_stamp
        # if frameidx>=40: 
        # break
    
    # return 0 
    # Export
    instance_map.save_debug_results(os.path.join(dataroot,'debug',method), scene_name, mean_time=time_array.mean())
    
    # Save visualization
    viz_folder = os.path.join(dataroot,'output',method)
    if viz_folder is not None:
        instance_map.update_result_points(min_fg=MIN_FG,min_viewed=MIN_FG_VIEWED)
        instance_labels = instance_map.extract_object_map()
        semantic_colors, instance_colors = render_result.generate_colors(instance_labels.astype(np.int64))
        pcd.colors = o3d.utility.Vector3dVector(semantic_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_semantic.ply'.format(scene_name)),pcd)
        pcd.colors = o3d.utility.Vector3dVector(instance_colors/255.0)
        o3d.io.write_point_cloud(os.path.join(viz_folder,'{}_instance.ply'.format(scene_name)),pcd)

    if DATASET=='scannet':
        instance_map.save_scannet_results(os.path.join(dataroot,'eval',method), scene_name)    


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='scannetv2')
    parser.add_argument('--prior_model', help='directory to likelihood model')
    parser.add_argument('--method', help='method name')
    parser.add_argument('--prediction_folder', help='prediction folder in each scan', default='prediction_no_augment')
    parser.add_argument('--split', help='split', default='val')
    parser.add_argument('--split_file', help='split file name', default='val')
    
    opt = parser.parse_args()
    dataroot = opt.data_root 
    prior_model = opt.prior_model
    PREDICTION_FOLDER = opt.prediction_folder #'prediction_no_augment' # [prediction_no_augment, prediction_forward, prediction_backward]
    method = opt.method 
    scans = read_scans(os.path.join(dataroot,'splits','{}.txt'.format(opt.split_file)))
    split = opt.split #'val'
    FUSE_ALL_TOKENS = True
    VISUALIZATION = False
    valid_scans = []
    # scans = ['scene0011_01']
    # scans = ['rgbd_dataset_freiburg1_room']
    scans = ['255']
    
    for scan in scans:
        if len(glob.glob(os.path.join(dataroot,split,scan,PREDICTION_FOLDER,'*.json')))>0:
            valid_scans.append(scan)
        else:
            print('missing {}'.format(scan))
    
    print('find {}/{} valid scans'.format(len(valid_scans),len(scans)))
    label_predictor = LabelFusion(prior_model, fuse_all_tokens=FUSE_ALL_TOKENS)
    # exit(0)
    
    debug_folder = os.path.join(dataroot,'debug',method)
    viz_folder = os.path.join(dataroot,'output',method)
    eval_folder = os.path.join(dataroot,'eval',method)
    if os.path.exists(debug_folder)==False:os.makedirs(debug_folder)
    if os.path.exists(viz_folder)==False:os.makedirs(viz_folder)
    if os.path.exists(eval_folder)==False:os.makedirs(eval_folder)
    
    for scan_name in valid_scans:
        process_scene((os.path.join(dataroot,split,scan_name), dataroot, method, PREDICTION_FOLDER, label_predictor,VISUALIZATION))
        break
    exit(0)
    
    import multiprocessing as mp
    p = mp.Pool(processes=16)
    p.map(process_scene, [(os.path.join(dataroot,split,scan_name), dataroot, method, PREDICTION_FOLDER, label_predictor, VISUALIZATION) for scan_name in valid_scans])
    p.close()
    p.join()
    
