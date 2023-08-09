import os, glob,sys
import numpy as np
import open3d as o3d
import cv2
import json
from numpy import linalg as LA
from sklearn.preprocessing import normalize

import render_result
import project_util

SEMANTIC_NAMES = render_result.SEMANTIC_NAMES
SEMANTIC_IDX = render_result.SEMANTIC_IDXS

class Detection:
    def __init__(self,u0,v0,u1,v1,labels):
        self.u0 = u0
        self.v0 = v0
        self.u1 = u1
        self.v1 = v1
        # self.label = -1 # int [0,20), index in the semantic_names, invalid is -1
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
    def __init__(self,id):
        self.id = id
        self.os_labels = [] # list of openset labels; each openset label is a map of {label:score}
        self.prob_weight = 0
        self.prob_vector = np.zeros((20),dtype=np.float32) # probability for each nyu20 types, append 'unknown' at the end
        self.pos_observed = 0   # number of times been observed
        self.neg_observed = 0   # number of times been ignored
        self.points = np.array([]) # aggregate positive points 
        self.negative = np.array([]) # aggreate negative points
        self.centroid = np.zeros(3,dtype=np.float32) # 3D centroid
    
    def get_exist_confidence(self):
        return self.pos_observed/(self.pos_observed+self.neg_observed+1e-6)    
    
    def create_instance(self,observed_pt_indices,labels,use_baseline=False):
        '''
        Input:
            - observed_pt_indices: (N'),np.int32, indices of points_uv that are inside the detection
        '''
        
        self.pos_observed += 1
        # update the openset labels
        if use_baseline:
            max_label = ''
            max_score = 0.0
            for os_name, score in labels.items():
                if score>max_score:
                    max_score = score
                    max_label = os_name
            assert max_score>0.0, 'no valid openset label'
            det_labels = {max_label: 1.0}
            # self.os_labels[max_label] = [1.0]
        else:        
            det_labels = labels
            # for os_name, score in labels.items():
            #     self.os_labels[os_name] = [score]        
        self.os_labels.append(det_labels)
        
        # update points
        self.points = np.array(observed_pt_indices)#.reshape(n,1)
        # print('create instance {} with {} points'.format(self.label,len(self.points)))
    
    def integrate_positive(self,observed_pt_indices,labels,use_baseline=False):
        self.pos_observed += 1
        self.points = np.concatenate((self.points,observed_pt_indices),axis=0)
        
        if use_baseline: #update the label with max score
            max_label = ''
            max_score = 0.0
            for os_name, score in labels.items():
                if score>max_score:
                    max_score = score
                    max_label = os_name
            assert max_score>0.0, 'no valid openset label'
            self.os_labels.append({max_label:1.0})
            # if max_label in self.os_labels:
            #     self.os_labels[max_label].append(1.0)
            # else:
            #     self.os_labels[max_label] = [1.0]
            
        else: # update all labels with scores
            self.os_labels.append(labels)
            
            # for os_name, score in labels.items():
            #     if os_name in self.os_labels:
            #         self.os_labels[os_name].append(score)
            #     else:
            #         self.os_labels[os_name] = [score]
    
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
        
        self.negative = np.concatenate((self.negative,unique_pts[negative]),axis=0)
        
        # verify, remove the verify later
        check = np.isin(self.negative,self.points)
        assert check.sum() == self.negative.shape[0], 'negative points not in the instance'        

    def get_points(self,min_fg_threshold=0.2,min_positive=0,verbose=False):
        ''' Return the point indices of the instance
        '''
        indices = np.unique(self.points) # (|P_m|,)
        assert indices.shape[0]>0, 'instance {} has no points'.format(self.id)

        pos_count, _ = np.histogram(self.points,bins=np.arange(self.points.max()+2)) # (max(P_m)+1,)
        neg_count, _ = np.histogram(self.negative,bins=np.arange(self.points.max()+2)) # (max(P_m)+1,)
        positive = pos_count[indices] # (|P_m|,)
        negative = neg_count[indices] # (|P_m|,)
        
        conf = positive.astype(np.float32)/(positive+negative+1e-6)
        valid = (conf >= min_fg_threshold) & (positive>=min_positive)
        if verbose:
            print('instance {} has {}/{} valid points'.format(self.id,np.sum(valid),len(valid)))
        if valid.sum()<1:
            return np.array([])
        else:          
            return indices[valid].astype(np.int32)     
    
    def update_label_probility(self, current_prob, score):
        self.prob_weight += score
        self.prob_vector += current_prob
        
    def merge_instance(self,other_instance):
        # self.os_labels += other_instance.os_labels 
        for os_name, score in other_instance.os_labels.items():
            if os_name in self.os_labels:
                self.os_labels[os_name] += score
            else:
                self.os_labels[os_name] = score
        self.prob_weight += other_instance.prob_weight
        self.prob_vector += other_instance.prob_vector
        self.pos_observed += other_instance.pos_observed
        self.neg_observed += other_instance.neg_observed
        
        self.points = np.concatenate((self.points,other_instance.points),axis=0)
        if other_instance.negative.size>3:
            self.negative = np.concatenate((self.negative,other_instance.negative),axis=0)
    
    def estimate_label(self):
        normalized_prob = self.get_normalized_probability()
        best_label = np.argmax(normalized_prob)
        conf = normalized_prob[best_label]
        return best_label, conf

    def estimate_top_labels(self,k=1):
        top_labels = []
        normalized_prob = self.get_normalized_probability()
        top_k = np.argsort(normalized_prob)[-k:]
        return top_k, normalized_prob[top_k]

    def get_normalized_probability(self):
        probability_normalized = self.prob_vector/(self.prob_vector.sum()+1e-6) # (LA.norm(self.prob_vector)+1e-6)
        return probability_normalized

class LabelFusion:
    def __init__(self,dir,use_baseline=False,propogate_method='mean'):
        # priors,association_count,likelihood,det_cond_probability,kimera_probability, openset_names, nyu20names = np.load(dir,allow_pickle=True)
        openset_names, nyu20names, likelihood = np.load(dir,allow_pickle=True)
        
        self.openset_names =  openset_names #[openset_id2name[i] for i in np.where(valid_rows)[0].astype(np.int32)]# list of openset names, (K,)
        self.nyu20_names = nyu20names # list of nyu20 names, (J,)
        # self.priors = priors # (K,), np.float32
        self.likelihood = likelihood
        # self.baseline_likelihood = kimera_probability
        self.use_baseline = use_baseline
        self.propogate_method = propogate_method # mean, multiply
        
        print('{} valid openset names are {}'.format(len(self.openset_names),self.openset_names))

        # if self.use_baseline:
        #     print('-------- using empirical likelihood --------')

        assert len(self.nyu20_names) == self.likelihood.shape[1], 'likelihood and nyu20 must have same number of rows'
        assert len(self.openset_names) == self.likelihood.shape[0], 'likelihood and openset must have same number of columns,{}!={}'.format(
            len(self.openset_names),self.likelihood.shape[0])
        
    # todo: remove this
    def estimate_current_prob(self,os_label):
        if os_label in self.openset_names:
            prob_vector = self.likelihood[self.openset_names.index(os_label),:]
            prob_vector = prob_vector/np.sum(prob_vector)   # normalize over total probability
            return prob_vector
        else: return None
        
    def bayesian_single_prob(self,openset_measurement):
        '''
        single-frame measurement, the measurements if a map of {name:score}
        '''
        prob = np.zeros(len(self.nyu20_names),np.float32)
        weight = 0.0
        labels = []
        scores = []

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
            assert np.abs(prob_vector.sum() - 1.0) < 1e-6, 'prob vector is not normalized'
            prob += scores[i] * prob_vector
            
        weight = 1.0    
        return prob,weight    
    
    def baseline_single_prob(self, openset_measurement):
        max_label =''
        max_score = 0.0
        prob = np.zeros(len(self.nyu20_names),np.float32)
        weight = 0.0

        for os_name, score in openset_measurement.items():
            if score>max_score:
                max_score = score
                max_label = os_name
        assert max_score>0.0, 'no valid openset label'
        
        if max_label in self.openset_names:
            prob_vector = self.likelihood[self.openset_names.index(max_label),:]
            prob_vector = prob_vector / np.sum(prob_vector)
            prob = prob_vector 
            weight = 1.0
        return prob, weight


    def estimate_batch_prob(self,openset_measurements):
        '''·
        multiple-frame measurement, the measurements if a map of {name:[score1,score2,...]}
        '''
        
        prob = np.zeros((len(self.nyu20_names),),np.float32)
        weight = 0.0

        if self.propogate_method=='mean':
            for zk in openset_measurements:
                if self.use_baseline:
                    prob_k, weight_k = self.baseline_single_prob(zk)
                else:
                    prob_k, weight_k = self.bayesian_single_prob(zk)
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
    
    def insert_instance(self,instance):
        self.instance_map[str(instance.id)] = instance
        
    def extract_object_map(self, instance_conf = 0.1,pt_conf=0.1,min_pos=2):
        instance_labels = np.zeros(len(self.points)) -100
        count = 0
        for instance_id,instance in self.instance_map.items():
            p_instance = instance.get_points(pt_conf,min_pos,True)
            label_id,conf = instance.estimate_label()
            if p_instance.shape[0]>0 and label_id!=20:
                composite_label = label_id*1000 + instance.id + 1
                instance_labels[p_instance] = composite_label #instance.id
                count +=1   
            
        print('Extract {}/{} instances to visualize'.format(count,len(self.instance_map)))
        return instance_labels
    
    def save_debug_results(self,eval_dir, scene_name):
        output_folder = os.path.join(eval_dir,scene_name)
        if os.path.exists(output_folder)==False:
            os.makedirs(output_folder)
            
        # pred_file = open(os.path.join(output_folder,'predinfo.txt'),'w')
        label_debug_file = open(os.path.join(output_folder,'fusion_debug.txt'),'w')
        # label_debug_file.write('# instance conf: {}, point conf: {} \n'.format(instance_conf, pt_conf))
        label_debug_file.write('# mask_file label_id label_conf pos_observe neg_observe point_number label_name \n')
        count = 0
        for instance_id,instance in self.instance_map.items():
            # instance_mask = np.zeros(len(self.points),dtype=np.int32)
            pos_points = instance.points #instance.get_points(pt_conf)
            neg_points = instance.negative
            if pos_points.shape[0]>0:
                label_instance,label_conf = instance.estimate_label()
                # exist_conf = instance.get_exist_confidence()

                # instance_mask[p_instance] = 1
                # if label_instance==20: # or exist_conf<instance_conf: 
                #     continue
                label_name = SEMANTIC_NAMES[label_instance]
                label_nyu40 = SEMANTIC_IDX[label_instance]
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
        # pred_file.close()
        label_debug_file.close()

    def save_scannet_results(self, eval_dir, scene_name, instance_conf=0.1, pt_conf=0.1, min_pos=2):
        output_folder = os.path.join(eval_dir,scene_name)
        if os.path.exists(output_folder)==False:
            os.makedirs(output_folder)
        count = 0
        pred_file = open(os.path.join(output_folder,'predinfo.txt'),'w')
        for instance_id,instance in self.instance_map.items():
            instance_mask = np.zeros(len(self.points),dtype=np.int32)
            p_instance = instance.get_points(pt_conf,min_pos)
            label_instance,label_conf = instance.estimate_label()

            if p_instance.shape[0]>0 and label_instance!=20:
                exist_conf = instance.get_exist_confidence()
                instance_mask[p_instance] = 1
                
                label_name = SEMANTIC_NAMES[label_instance]
                label_nyu40 = SEMANTIC_IDX[label_instance]
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
    
    # Merge instances that overlap with each other
    # todo: select root instance based on the number of points. Skipping the positive observation filter.
    def merge_instances(self):
        ROOT_INSTANCE ={
            'points':2000,
            'pos':1
        }
        MERGE_THRESHOLD = {'distance':3.0,'iou':0.2,'similarity':0.5}
        ROOT2ROOT_POS = 2
        ROOT2CHILD_POS = 1
        
        # FILTER_THRESHOLD = {'points':500, 'pos':2}
        
        skip_types={'floor','wall','unknown'}
        parent_instances = []
        child_instances = []
        
        # Find root instances
        for idx,instance in self.instance_map.items():
            label_id, label_conf = instance.estimate_label()
            instance_points = instance.get_points(min_positive=ROOT_INSTANCE['pos'])
            
            if SEMANTIC_NAMES[label_id] in skip_types: continue
            if (instance_points.shape[0]>ROOT_INSTANCE['points']):
                parent_instances.append(idx)
                instance_split = 'parent'
            else:
                child_instances.append(idx)
                instance_split = 'child'
            # print('{} instance {}, {} points'.format(
            #         instance_split,SEMANTIC_NAMES[label_id],instance_points.shape[0]))
        
        print('Find {} parent instances:{}'.format(len(parent_instances),parent_instances))
        
        # 1. Merge small instances into the parent instances
        remove_instances = []
        count_merge = 0
        
        for child_id in child_instances:
            # print('checking parent instance {}'.format(SEMANTIC_NAMES[parent_label]))
            child_inst = self.instance_map[child_id]
            child_points = np.zeros(len(self.points),dtype=np.int32)
            child_points_indices = child_inst.get_points(min_positive=1)
            if child_points_indices.shape[0]==0: continue
            child_points[child_points_indices] = 1     
            child_label, _ = child_inst.estimate_label()
            matched_id = -1
            matched_iou = 0.0
            matched_label = ''
            
            for root_id  in parent_instances:
                root_instance = self.instance_map[root_id]
                dist = np.linalg.norm(child_inst.centroid-root_instance.centroid)

                if (dist>MERGE_THRESHOLD['distance']):
                    continue
                # if parent_id in parent_merge_pairs and child_id==parent_merge_pairs[parent_id]:
                #     continue    
                
                parent_points = np.zeros(len(self.points),dtype=np.int32)
                parent_points[root_instance.get_points(min_positive=ROOT2CHILD_POS)] = 1                
                
                iou = np.sum(child_points*parent_points)/min(np.sum(child_points),np.sum(parent_points))
                similarity = np.dot(root_instance.get_normalized_probability(),child_inst.get_normalized_probability())
                # print('{}-{}:{}'.format(SEMANTIC_NAMES[parent_label],SEMANTIC_NAMES[child_label],similarity))
                
                if iou>MERGE_THRESHOLD['iou'] and similarity>MERGE_THRESHOLD['similarity'] and iou>matched_iou:
                    matched_id = root_id
                    matched_iou = iou
                    parent_label, _ = root_instance.estimate_label()
                    matched_label = SEMANTIC_NAMES[parent_label]
                
            if matched_iou>0:
                matched_instance = self.instance_map[matched_id]
                matched_instance.merge_instance(child_inst)
                remove_instances.append(child_id)
                count_merge +=1
                print('Merge instance {}_{} to {}_{}'.format(child_id,SEMANTIC_NAMES[child_label],matched_id,matched_label))

        # todo
        # 2. merge parent instances
        clusteres = {} # {root_node:[child_nodes]}
        match_pairs = {} # {child_node:root_node}
        nodes_size = {} # {node_id:node_size}
        
        for ida, root_id in enumerate(parent_instances): # create clusters
            root_instance = self.instance_map[root_id]
            indices_a = root_instance.get_points(min_positive=ROOT2ROOT_POS)
            if indices_a.size<1: continue
            points_a = np.zeros(len(self.points),dtype=np.int32)
            points_a[indices_a] = 1
            matches_to_newcluster = []
            
            print('{} instance has {} points with {} positive scan'.format(root_id,indices_a.shape[0],ROOT2ROOT_POS))
                     
            for root_b in parent_instances[ida+1:]:
                assert root_id != root_b, 'root_a == root_b'
                if root_id in match_pairs and root_b in match_pairs:
                    continue
                instance_b = self.instance_map[root_b]
                dist = np.linalg.norm(root_instance.centroid-instance_b.centroid)
                similarity = np.dot(root_instance.get_normalized_probability(),instance_b.get_normalized_probability())
                if dist>MERGE_THRESHOLD['distance'] and similarity>MERGE_THRESHOLD['similarity']:
                    continue
                indices_b = instance_b.get_points(min_positive=ROOT2ROOT_POS)
                if indices_b.size<1: continue
                points_b = np.zeros(len(self.points),dtype=np.int32)
                points_b[indices_b] = 1
                iou = np.sum(points_a*points_b)/min(np.sum(points_a),np.sum(points_b))
                
                if iou>MERGE_THRESHOLD['iou']:
                    if root_id in match_pairs:
                        clusteres[match_pairs[root_id]].append(root_b)
                    else:
                        match_pairs[root_b] = root_id
                        matches_to_newcluster.append(root_b)
                    print('Merge parent instances ({},{})'.format(root_id,root_b))
                    nodes_size[root_id] = np.sum(points_a)
                    nodes_size[root_b] = np.sum(points_b)

            # Save cluster
            if len(matches_to_newcluster)>0:
                matches_to_newcluster.append(root_id)
                clusteres[root_id] = matches_to_newcluster
                
        print('---- cluster results ----')
        for k,v in clusteres.items(): # merge each cluster
            # print('{}:{}'.format(k,v))
            dominate_node = ''
            dominate_node_size = 0
            for instance in v:
                if nodes_size[instance]>dominate_node_size:
                    dominate_node = instance
                    dominate_node_size = nodes_size[instance]

            dominate_instance = self.instance_map[dominate_node]
            # break
            for instance in v:
                if instance != dominate_node:
                    dominate_instance.merge_instance(self.instance_map[instance])
                    remove_instances.append(instance)
                    count_merge +=1
                    # print('Merge instance {} to {}'.format(instance,dominate_node))
            
        # 3. Remove instances that are merged
        count_remove = 0
        for id in remove_instances:
            if id in self.instance_map:
                del self.instance_map[id]
                count_remove+=1
                
        print('{} merged, {} removed'.format(count_merge,count_remove))
    
    def filter_conflict_objects(self):
        CONFLICT_IOU = 0.2
        SEARCH_RADIUS = 2.0
        MIN_POSITIVE = 2
        
        count=0
        J = len(self.instance_map)
        conflict_matrix = np.zeros((J,J),dtype=np.int32)
        remove_instances = []
        
        # Build conflict matrix
        for i, instance_i in self.instance_map.items():

            i_points = np.zeros(len(self.points),dtype=np.int32)
            i_point_indices = instance_i.get_points(min_positive=MIN_POSITIVE)
            if i_point_indices.shape[0]==0: continue
            i_points[i_point_indices] = 1  
            
            for j, instance_j in self.instance_map.items():
                if j<=i:continue
                dist = np.linalg.norm(instance_i.centroid-instance_j.centroid)
                if dist>SEARCH_RADIUS: continue
                
                j_points = np.zeros(len(self.points),dtype=np.int32)
                j_point_indices = instance_j.get_points(min_positive=MIN_POSITIVE)
                if j_point_indices.shape[0]==0: continue
                j_points[j_point_indices] = 1  
                
                iou = np.sum(i_points*j_points)/min(np.sum(i_points),np.sum(j_points))
                
                if iou>=CONFLICT_IOU:
                    # conflict_matrix[int(i),int(j)] = 1
                    # conflict_matrix[int(j),int(i)] = 1
                    # print('Conflict between {} and {}'.format(i,j))
                    small_instance = i if i_point_indices.shape[0]<j_point_indices.shape[0] else j
                    remove_instances.append(small_instance)
                    print('[conflict ] instance {}: {} points, {} obsr, instance {}: {} points, {} obsr; iou :{:.3f}'.format(i,i_point_indices.shape[0],instance_i.pos_observed,j,j_point_indices.shape[0],instance_j.pos_observed,iou))
                    count +=1
        print('Find {} conflicts'.format(count))
        
        print('Remove conflict instances:{}'.format(remove_instances))
        count = 0
        for i in remove_instances:
            if i in self.instance_map:
                del self.instance_map[i]
                count +=1
        print('remove {} conflict instances'.format(count))
                
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

def check_exist_labels(source_labels,target_labels):
        exitst = []        
        for label in source_labels:
            if label in target_labels:
                exitst.append(label)
        return exitst

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
    CONFLICT_IGNORE_TYPES = ['floor','bag','pillow','blanket']
    
    with open(label_file, 'r') as f:
        json_data = json.load(f)
        tags = json_data['tags']
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
                # label_names = list(labels.keys())

                if (bbox[2]-bbox[0])/img_width>MAX_BOX_RATIO and (bbox[3]-bbox[1])/img_height>MAX_BOX_RATIO:
                    continue
                z_ = Detection(bbox[0],bbox[1],bbox[2],bbox[3],labels)
                valid_detection_flag = True
                
                for k, prev_z in enumerate(detections): # remove large bbox that surround small bbox
                    if len(check_exist_labels(CONFLICT_IGNORE_TYPES,labels))>0 or len(check_exist_labels(CONFLICT_IGNORE_TYPES,prev_z.labels))>0:
                        continue
                    
                    overlap, z_is_small = find_overlap(z_.get_bbox(),prev_z.get_bbox())
                    # print('labels: {},overlap: {}, current small:{}'.format(labels,overlap,z_is_small))
                    if overlap> 0.95 and z_is_small:
                        # detections.remove(prev_z)
                        invalid_detections.append(k)
                        # print(z_.get_bbox())
                        # print(prev_z.get_bbox())
                        # print('{} and {} conflict {}'.format(instance_id,k, overlap))
                    elif overlap>0.95 and not z_is_small:
                        valid_detection_flag = False
                
                if valid_detection_flag:
                    z_.add_mask(mask[instance_id,:,:]==1)
                    detections.append(z_)
                    # print('add detection {}'.format(instance_id))
                    # labels_msg+=label+','
                # print('detection {} has {} valid masks'.format(label,np.sum(mask==instance_id)))

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
        uv_k = zk.mask # (H,W), bool
        for j_ in range(M):
            l_j = instances.instance_map[str(j_)]
            p_instance = l_j.get_points()

            # if l_j.label != zk.label or p_instance.shape[0]==0: continue
            
            uv_j = points_uv[p_instance,1:3].astype(np.int32)
            observed = np.logical_and(uv_j[:,0]>=0,uv_j[:,1]>=0)   
            if observed.sum() < min_observe_points: continue
            uv_j = uv_j[observed] # (|P_m|,2)

            uv_m = np.zeros(uv_k.shape,dtype=np.bool_)
            uv_m[uv_j[:,1],uv_j[:,0]] = True
            if np.sum(uv_m)>0:
                overlap = np.logical_and(uv_k,uv_m)
                iou[k_,j_] = np.sum(overlap)/np.sum(uv_m)
                        
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
    scene_dir, eval_folder, pred_folder, label_predictor,visualize,use_baseline = args
    scene_name = os.path.basename(scene_dir)
    MAP_POSIX = 'vh_clean_2.ply'
    INTRINSIC_FOLDER ='intrinsic'
    PREDICTION_FOLDER = pred_folder #'prediction_refine'
    MIN_VIEW_POINTS = 200
    MIN_OBJECT_POINTS =200
    FRAME_GAP = 20

    # depth filter related params
    # visualize = True
    max_dist_gap = 0.2
    depth_kernal_size = 5
    kernal_valid_ratio = 0.2
    kernal_max_var = 0.15
    # output_dir = os.path.join(scene_dir,PREDICTION_FOLDER)
    
    # Instances parameters
    INST_THRESHOLD=0.1
    FG_THRESHOLD = 0.8
    MIN_IOU = 0.5
    if os.path.exists(os.path.join(scene_dir,'tmp'))==False:
        os.mkdir(os.path.join(scene_dir,'tmp'))
        
    # Init
    K_rgb, K_depth,rgb_dim,depth_dim = project_util.read_intrinsic(os.path.join(scene_dir,INTRINSIC_FOLDER))
    rgb_out_dim = depth_dim
    K_rgb_out = project_util.adjust_intrinsic(K_rgb,rgb_dim,rgb_out_dim)
    predict_folder = os.path.join(scene_dir,PREDICTION_FOLDER)
    predict_frames =  glob.glob(os.path.join(predict_folder,'*_label.json'))  
    print('---- {} find {} prediction frames'.format(scene_name,len(predict_frames)))
    
    # Load map
    map_dir = os.path.join(scene_dir,'{}_{}'.format(scene_name,MAP_POSIX))
    pcd = o3d.io.read_point_cloud(map_dir)
    # pcd.estimate_normals()
    points = np.asarray(pcd.points,dtype=np.float32)
    colors = np.asarray(pcd.colors,dtype=np.float32)
    # normals = np.asarray(pcd.normals,dtype=np.float32)
    normals = np.tile(np.array([0,0,1]),(points.shape[0],1))
    N = points.shape[0]
    
    instance_map = ObjectMap(points,colors)
    
    # Integration
    for i,pred_frame in enumerate(sorted(predict_frames)):   
        frame_name = os.path.basename(pred_frame).split('_')[0] 
        frameidx = int(frame_name.split('-')[-1])
        if frameidx % FRAME_GAP != 0: continue
        # if frameidx>300: break
        # print('frame id: {}'.format(frameidx))
        
        rgbdir = os.path.join(scene_dir,'color',frame_name+'.jpg')
        pose_dir = os.path.join(scene_dir,'pose',frame_name+'.txt')
        depth_dir = os.path.join(scene_dir,'depth',frame_name+'.png')

        # load rgbd, pose
        rgbimg = cv2.imread(rgbdir)
        raw_depth = cv2.imread(depth_dir,cv2.IMREAD_UNCHANGED)
        depth = raw_depth.astype(np.float32)/1000.0
        assert raw_depth.shape[0]==depth_dim[0] and raw_depth.shape[1]==depth_dim[1], 'depth image dimension does not match'
        assert depth.shape[0] == rgbimg.shape[0] and depth.shape[1] == rgbimg.shape[1]
        pose = np.loadtxt(pose_dir)

        # projection
        mask, points_uv, _, _ = project_util.project(
            points, normals,pose, K_rgb_out, rgb_out_dim, 5.0, 0.5) # Nx3
        filter_mask = project_util.filter_occlusion(points_uv,depth,max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        
        # test_mask = points_uv[:,1] >= 0
        # assert mask.sum() == test_mask.sum()
        
        points_uv = np.concatenate([np.arange(N).reshape(N,1),points_uv],axis=1) # Nx4, np.double, [i,u,v,d]
        mask = np.logical_and(mask,filter_mask) # Nx1, bool
        points_uv[~mask] = np.array([-100,-100,-100,-100])
        iuv = points_uv[mask,:3].astype(np.int32)  
        uv_rgb = colors[mask,:3] * 255.0
        count_view_points = np.sum(mask)  
        if count_view_points < MIN_VIEW_POINTS:
            print('drop poor frame with {} valid points'.format(count_view_points))
            continue
        print('{}/{} viewed in frame {}'.format(count_view_points,points.shape[0],frame_name))
        
        # detections
        tags, detections = load_pred(predict_folder, frame_name)
    
        if len(detections) == 0:
            print('no detection in current frame')
            continue
        
        # association
        
        assignment, missed = find_assignment(detections,instance_map,points_uv,min_iou=MIN_IOU,min_observe_points=MIN_OBJECT_POINTS,verbose=False)
        M = len(instance_map.instance_map)
        # integration
        for k,z in enumerate(detections):
            prob_current, weight = label_predictor.bayesian_single_prob(z.labels)
            
            if weight<1e-6: continue
            if assignment[k,:].max()==0: # unmatched            
                new_instance = Instance(instance_map.get_num_instances())
                new_instance.create_instance(z.scanned_pts(iuv),z.labels,use_baseline)
                if new_instance.points.shape[0] < MIN_OBJECT_POINTS: continue # 50
                new_instance.update_label_probility(prob_current,weight)
                instance_map.insert_instance(new_instance)
                z.assignment = 'new'
                
            else: # integration
                j = np.argmax(assignment[k,:])
                matched_instance = instance_map.instance_map[str(j)]
                matched_instance.update_label_probility(prob_current, weight)
                matched_instance.integrate_positive(z.scanned_pts(iuv),z.labels,use_baseline)
                matched_instance.integrate_negative(z.mask, points_uv, mask)
                z.assignment = 'match'
        
        # visualize 
        if visualize:
            # prj_rgb = np.zeros((rgb_out_dim[0],rgb_out_dim[1],3),np.uint8)
            # M = iuv.shape[0]
            # for m_ in range(M):
            #     cv2.circle(prj_rgb,(int(iuv[m_,1]),int(iuv[m_,2])),1,(int(uv_rgb[m_,2]),int(uv_rgb[m_,1]),int(uv_rgb[m_,0])),1)

            # out = np.concatenate([rgbimg,prj_rgb],axis=1)
            # cv2.imwrite(os.path.join(scene_dir,'tmp','{}_rgb.jpg'.format(frame_name)),out)      

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
                cv2.putText(proj_instances, SEMANTIC_NAMES[pred_label_id], (int(centroid[0]),int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if j_ in missed:
                    cv2.circle(proj_instances,(int(centroid[0]+10),int(centroid[1]+10)),8,(0,0,255),-1)
                else:
                    cv2.circle(proj_instances,(int(centroid[0]+10),int(centroid[1]+10)),8,(0,255,0),-1)
            
            out = np.concatenate([rgbimg,proj_instances],axis=1)
            
            for k_ in range(K):
                zk = detections[k_]
                # zk_centroid = zk.cal_centroid()
                prob, weight = label_predictor.estimate_batch_prob(zk.labels)
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
        
        # if frameidx>=40: 
        # break
    
    # Export
    instance_map.save_debug_results(eval_folder, scene_name)
    # composite_labels = instance_map.extract_object_map(instance_conf=INST_THRESHOLD,pt_conf=FG_THRESHOLD) 
    # np.save(os.path.join(eval_folder,scene_name,'result.npy'),composite_labels)

if __name__=='__main__':
    dataroot = '/data2/ScanNet'
    prior_model = '/home/cliuci/code_ws/OpensetFusion/measurement_model/prompt/prompt_likelihood.npy'
    MIN_DET_RATE = 0.08
    PREDICTION_FOLDER = 'prediction_aligned_prompts'
    USE_BASELINE = False
    result_folder = os.path.join(dataroot,'debug','bayesian+')
    scans = read_scans(os.path.join(dataroot,'splits','val_clean.txt'))
    split = 'val'
    # scans = ['scene0633_01']
    
    valid_scans = []
    
    for scan in scans:
        if len(glob.glob(os.path.join(dataroot,split,scan,PREDICTION_FOLDER,'*.json')))>0:
            valid_scans.append(scan)
        else:
            print('missing {}'.format(scan))
    
    print('find {}/{} valid scans'.format(len(valid_scans),len(scans)))
    
    VISUALIZATION = False
    label_predictor = LabelFusion(prior_model, use_baseline=USE_BASELINE)
    # exit(0)

    if os.path.exists(result_folder)==False:
        os.makedirs(result_folder)
    
    # for scan_name in valid_scans:
    #     process_scene((os.path.join(dataroot,split,scan_name), result_folder, PREDICTION_FOLDER, label_predictor,VISUALIZATION,USE_BASELINE))
    #     break
    # exit(0)
    
    import multiprocessing as mp
    p = mp.Pool(processes=32)
    p.map(process_scene, [(os.path.join(dataroot,split,scan_name), result_folder, PREDICTION_FOLDER, label_predictor, VISUALIZATION, USE_BASELINE) for scan_name in valid_scans])
    p.close()
    p.join()
    
