import os, glob,sys
import numpy as np
import open3d as o3d
import open3d.core as o3c
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

def extract_instance_points_thread(args):
    '''
    Input:
    - gt_points: (N,3), np.float32
    - instance_points: (M,3), np.float32
    - min_dist: float
    Output:
    - global_point_indices: (K,), np.int32, indices of gt_points that are inside the instance_points
    '''
    # MIN_DIST   = 0.1
    gt_points,instance_points,min_dist = args
    
    instance_pcd = o3d.geometry.PointCloud()
    instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
    instance_kd_tree = o3d.geometry.KDTreeFlann(instance_pcd)
    
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    
    N = len(gt_pcd.points)
    global_point_indices = []
    
    for i in np.arange(N):
        [k,idx,_] = instance_kd_tree.search_radius_vector_3d(gt_pcd.points[i], min_dist)
        if k>0:global_point_indices.append(i)
    
    return np.array(global_point_indices).astype(np.int32)


class Detection:
    def __init__(self,u0,v0,u1,v1,labels):
        self.u0 = u0
        self.v0 = v0
        self.u1 = u1
        self.v1 = v1
        self.labels = labels # K, {label:score}
        self.mask = None # H*W, bool
        self.assignment = ''
        self.valid = True
    
    def cal_centroid(self):
        centroid = np.array([(self.u0+self.u1)/2.0,(self.v0+self.v1)/2.0])
        return centroid.astype(np.int32)

    def get_bbox(self):
        ''' [u0,v0,u1,v1] '''
        bbox = np.array([self.u0,self.v0,self.u1,self.v1])
        return bbox
    
    def get_bbox_area(self):
        return (self.u1-self.u0)*(self.v1-self.v0)
    
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
    def __init__(self,id, J=20, device = o3c.Device('cpu:0')):
        self.id = id
        self.os_labels = [] # list of openset labels; each openset label is a map of {label:score}
        # self.voxels= np.array([])  # (N,3), int32, coordinate of voxel grid
        self.mhashmap = o3c.HashMap(100,
            key_dtype=o3c.int32,
            key_element_shape=(3,),
            value_dtypes=(o3c.uint8,),
            value_element_shapes=((1,),),
            device = device)
        
        self.uv_map = None
        self.points = np.array([]) # (N,3), float32, points from voxel grid
        self.dense_points = np.array([]) # (N,), int32, points indices from volumetric map
        self.voxel_length = 0.0
        self.prob_weight = 0
        self.prob_vector = (1/J) + np.zeros(J,dtype=np.float32) # probability for each nyu20 types, append 'unknown' at the end
        self.pos_observed = 1   # number of times been observed
        self.neg_observed = 0   # number of times been ignored
        # self.negative = np.array([]) # aggreate negative points
        # self.filter_points= np.array([]) # points after filtering
        # self.merged_points = np.array([]) # points after merging geometric segments
        self.centroid = np.zeros(3,dtype=np.float32) # 3D centroid
    
    def get_exist_confidence(self):
        return self.pos_observed/(self.pos_observed+self.neg_observed+1e-6)    
    
    def create_voxel_map(self, point_cloud:o3d.geometry.PointCloud, resolution=256.0):
        ''' create with point_cloud in (N,3), np.float32'''
        self.voxel_length = 4.0/resolution
        points = np.asarray(point_cloud.points,dtype=np.float32)
        N = points.shape[0]
        assert N>0, 'point cloud is empty'
        weights = np.ones((N,1),dtype=np.uint8)
        voxels = np.floor(points/self.voxel_length).astype(np.int32) # (N,3)
        self.mhashmap.insert(o3c.Tensor(voxels,dtype=o3c.int32), o3c.Tensor(weights,dtype=o3c.uint8))
        # active_indices = self.mhashmap.active_buf_indices()
        # print('create hashmap {} with {} voxels'.format(self.id,N))
    
    def integrate_volume_debug(self, point_cloud:o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points,dtype=np.float32)
        query_voxels = np.floor(points/self.voxel_length).astype(np.int32) # (N,3)
        query_voxels = o3c.Tensor(query_voxels,dtype=o3c.int32)
        buf_indices, mask  = self.mhashmap.find(query_voxels)
        buf_indices = buf_indices[mask].to(o3c.int64)
        
        # update existed values
        self.mhashmap.value_tensor()[buf_indices] += 1
        new_voxels = query_voxels[mask==False]
        if new_voxels.shape[0]>0:
            new_weights = o3c.Tensor.ones((new_voxels.shape[0],1),dtype=o3c.uint8)
            self.mhashmap.insert(new_voxels, new_weights)
        
        # other update
        self.pos_observed += 1
    
    def query_voxels(self, query_voxels):
        buf_indices, mask  = self.mhashmap.find(query_voxels)
        return buf_indices[mask].to(o3c.int64)
    
    def merge_volume(self, other_hash_table):
        '''
        other_hash_table: o3c.HashMap
        '''
        other_voxels_indices = other_hash_table.active_buf_indices()
        other_voxels_coordinate = other_hash_table.key_tensor()[other_voxels_indices].to(o3c.int32)
        other_voxels_weight = other_hash_table.value_tensor()[other_voxels_indices].to(o3c.uint8)
        
        # query
        recalled_indices, mask = self.mhashmap.find(other_voxels_coordinate)
        recalled_indices = recalled_indices[mask].to(o3c.int64)
        
        # update 
        self.mhashmap.value_tensor()[recalled_indices] += other_voxels_weight[mask]
        
        # insert new 
        new_voxels = other_voxels_coordinate[mask==False]
        new_weights=  other_voxels_weight[mask==False]
        if new_voxels.shape[0]>0:
            self.mhashmap.insert(new_voxels, new_weights)
        
    
    def query_voxel_centroids(self,query_voxels):
        '''
        Input: query_voxels (N,3), o3c.Tensor
        Output: centroids (M,3), np.float32
        '''
        valid_indices = self.query_voxels(query_voxels=query_voxels)

        if valid_indices.shape[0]<1:
            return np.array([]),0.0
        else:
            all_active_indices = self.mhashmap.active_buf_indices().to(o3c.int64)
            valid_ratio = valid_indices.shape[0]/all_active_indices.shape[0]
            vx_centroids = self.mhashmap.key_tensor()[valid_indices].to(o3c.int32).numpy() * self.voxel_length + self.voxel_length/2.0
            return vx_centroids, valid_ratio
    
    def query_from_point_cloud(self,points,inflat_ratio=None,min_weight=1):
        '''
        Query the overlaped voxels from a point cloud.
        Input:
        - points: (N,3), np.float32
        Output:
        - recall_voxels_indices: (M,), np.int64, indices of the voxels that are recalled
        '''

        if inflat_ratio is None:
            query_voxels = np.floor(points/self.voxel_length).astype(np.int32)
            recall_voxels_indices = self.query_voxels(o3c.Tensor(query_voxels,dtype=o3c.int32))
        else:
            # inflat voxel grid map
            inflat_vx_length = self.voxel_length*inflat_ratio
            inflat_voxels = np.floor(self.points/inflat_vx_length).astype(np.int32)
            inflat_hash_table = o3c.HashMap(100,
                key_dtype=o3c.int32,
                key_element_shape=(3,),
                value_dtypes=(o3c.uint8,),
                value_element_shapes=((1,),),
                device = self.mhashmap.device)
            inflat_hash_table.insert(o3c.Tensor(inflat_voxels,dtype=o3c.int32), o3c.Tensor.ones((inflat_voxels.shape[0],1),dtype=o3c.uint8))
            # print('{}->{} voxels number after inflation'.format(self.mhashmap.active_buf_indices().shape[0],inflat_hash_table.active_buf_indices().shape[0]))
            
            #
            query_voxels = np.floor(points/inflat_vx_length).astype(np.int32)
            
            # query from the inflat voxel grid map
            recall_voxels_indices, mask = inflat_hash_table.find(o3c.Tensor(query_voxels,dtype=o3c.int32))
            recall_voxels_indices = recall_voxels_indices[mask].to(o3c.int64)
        
        return recall_voxels_indices
    
    # def refine_voxel_grid(self,global_points,device = o3c.Device('cpu:0')):
    #     global_voxels = np.floor(global_points/self.voxel_length).astype(np.int32)
    #     recall_voxels_indices = self.query_voxels(o3c.Tensor(global_voxels,dtype=o3c.int32))
    
    def read_voxel_grid(self, dir, resolution=256.0):        
        # voxels = np.load(dir)
        self.voxel_length = 4.0/resolution
        self.mhashmap = o3c.HashMap.load(dir)
        active_voxel_indices = self.mhashmap.active_buf_indices().to(o3c.int64)
        
        print('read {} voxels from {}'.format(active_voxel_indices.shape[0],dir))


    def update_voxels_from_dense_points(self,dense_xyz, device = o3c.Device('cpu:0'),
                                        update_voxel_weight=5,
                                        insert_new = True):
        '''
        read the points xyz and update the mhashmap accordingly.
        voxels that been queried are kept, other voxels are removed. 
        New points are inserted into the hashmap.
        '''
        # update_voxel_weight = 5
        # if self.dense_points.shape[0]<1:
        #     return None
        
        # dense_xyz = points[self.dense_points,:] # (N,3)
        if dense_xyz.shape[0]<1:
            return None
        
        dense_voxels = np.floor(dense_xyz/self.voxel_length).astype(np.int32) # (N,3)
        dense_voxels = o3c.Tensor(dense_voxels,dtype=o3c.int32)
                
        # Query voxels
        all_active_indices = self.mhashmap.active_buf_indices()
        queried_indices, mask = self.mhashmap.find(dense_voxels)
        queried_indices = queried_indices[mask]
        
        # Update existed voxels
        if queried_indices.shape[0]>0 and update_voxel_weight>0:
            self.mhashmap.value_tensor()[queried_indices] = update_voxel_weight
        
        # Remove invalid voxels
        toremove_indices = o3c.Tensor(np.setdiff1d(all_active_indices.numpy(),queried_indices.numpy()),
                                        dtype=o3c.int32, device=device)
        if toremove_indices.shape[0]>0:
            active_invalid_keys = self.mhashmap.key_tensor()[toremove_indices]
            self.mhashmap.erase(active_invalid_keys)
        
        # Insert new voxels
        new_voxels = dense_voxels[mask==False]
        if new_voxels.shape[0]>0 and insert_new:
            new_weights = update_voxel_weight * o3c.Tensor.ones((new_voxels.shape[0],1),dtype=o3c.uint8)
            self.mhashmap.insert(new_voxels, new_weights)
        
        print('before integration {} voxels, after {} voxels'.format(all_active_indices.shape[0],self.mhashmap.active_buf_indices().shape[0]))
        return self.mhashmap.active_buf_indices().shape[0]
        
    def update_current_uv(self,uv_map):
        ''' uv_map: H*W, np.unit8, valid uv are set to 1. '''
        # ''' points_uv: N*3, [u,v,d], only valid uv are saved. '''
        self.uv_map = uv_map
    
    def update_points(self):
        ''' update the points from the hashmap'''
        active_indices = self.mhashmap.active_buf_indices()
        self.points = self.mhashmap.key_tensor()[active_indices].to(o3c.int32).numpy() * self.voxel_length + self.voxel_length/2.0
        self.centroid = np.mean(self.points,axis=0)
        # print('instance {} update {} voxel points'.format(self.id,self.points.shape[0]))
    
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


    def save_label_measurements(self, os_labels):
        if os_labels is not None:
            self.os_labels.append(os_labels)
    
    def merge_segment(self,segment_points):
        ''' Merge a segment to instance points
        Input:
            - segment_points: (N,1), np.int32
        '''
        if self.dense_points.size < 1:
            self.dense_points = segment_points
        else:
            self.dense_points = np.concatenate((self.dense_points,segment_points),axis=0)
            self.dense_points = np.unique(self.dense_points)
    
    def estimate_label(self):
        normalized_prob = self.get_normalized_probability()
        best_label_id = np.argmax(normalized_prob)
        conf = normalized_prob[best_label_id]
        return best_label_id, conf

    def get_normalized_probability(self):
        probability_normalized = self.prob_vector/(self.prob_vector.sum()+1e-6) # (LA.norm(self.prob_vector)+1e-6)
        return probability_normalized
    
    def extract_aligned_points(self,global_points,min_weight=1):
        '''
        Extract the points that within its voxel grid map.
        Input:
            - points, N*3, np.float32. The point cloud provided from Volumetric Map or ScanNet.
        Output:
            - buf_indices, np.array(np.uint8). indices of the dense points
        '''
        N = global_points.shape[0]
        points_query = o3c.Tensor(np.floor(global_points/self.voxel_length).astype(np.int32),dtype=o3c.int32)
        # points_states = o3c.Tensor.zeros((N,),dtype=o3c.uint8)
        
        # find value
        buf_indices, mask  = self.mhashmap.find(points_query) # (N,), (N,), int, bool
        buf_indices = buf_indices[mask].to(o3c.int64) # (M,)
        weights = self.mhashmap.value_tensor()[buf_indices,0]
        
        # filter
        valid_mask  = weights>=min_weight # (M,), bool
        valid_points_indices = mask.nonzero()[0]
        
        # print('{}/{} voxels been queried'.format(len(valid_mask),len(self.mhashmap.active_buf_indices())))
        
        return valid_points_indices[valid_mask].numpy()


class LabelFusion:
    def __init__(self,dir,fusion_method=''):
        # priors,association_count,likelihood,det_cond_probability,kimera_probability, openset_names, nyu20names = np.load(dir,allow_pickle=True)
        with open(os.path.join(dir,'label_names.json'),'r') as f:
            data = json.load(f)
            openset_names = data['openset_names']
            closet_names = data['closet_names']
        
        likelihood = np.load(os.path.join(dir,'likelihood_matrix.npy'))
        
        self.openset_names =  openset_names #[openset_id2name[i] for i in np.where(valid_rows)[0].astype(np.int32)]# list of openset names, (K,)
        self.closet_names = closet_names # In ScanNet, uses NYU20
        self.likelihood = likelihood
        self.fusion_method = 'bayesian'
        # self.propogate_method = propogate_method # mean, multiply
        
        print('{} valid openset names are {}'.format(len(self.openset_names),self.openset_names))

        assert len(self.closet_names) == self.likelihood.shape[1], 'likelihood and nyu20 must have same number of rows'
        assert len(self.openset_names) == self.likelihood.shape[0], 'likelihood and openset must have same number of columns,{}!={}'.format(
            len(self.openset_names),self.likelihood.shape[0])
        
    def bayesian_single_prob(self,zk_measurement,multiplicative):
        '''
        single-frame measurement, the measurements if a map of {name:score}
        '''
        if multiplicative:
            zk_likelihood = np.ones(len(self.closet_names),np.float32)
        else:
            zk_likelihood = np.zeros(len(self.closet_names),np.float32)
        # weight = 0.0
        label_measurements = []
        score_measurements = []
        assert isinstance(zk_measurement,dict)

        for y_j,score_j in zk_measurement.items():
            if y_j in self.openset_names:
                label_measurements.append(self.openset_names.index(y_j))
                score_measurements.append(score_j)
                
        if len(label_measurements)<1: return zk_likelihood,False
        
        label_measurements = np.array(label_measurements)
        score_measurements = np.array(score_measurements)
        
        for j in range(label_measurements.shape[0]):
            j_likelihood = score_measurements[j] * self.likelihood[label_measurements[j],:]
            if multiplicative:
                zk_likelihood *= j_likelihood
            else:
                zk_likelihood += j_likelihood
            
            continue
            prob_vector = prob_vector / np.sum(prob_vector)
            assert np.abs(prob_vector.sum() - 1.0) < 1e-6, '{}({}) prob vector is not normalized {}'.format(
                self.openset_names[label_measurements[j]],label_measurements[j],self.likelihood[label_measurements[j],:])
            assert np.sum(prob_vector) > 1e-3, '{} prob vector is all zero'.format(self.openset_names[label_measurements[j]])
            zk_likelihood += score_measurements[j] * prob_vector
            
        # weight = 1.0    
        return zk_likelihood,True    
    
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

    def update_measurement(self, zk_measurement, semantic_probability):
        
        multiplicative=False
        regularization_factor = 1e-6
        if self.fusion_method=='bayesian':
            zk_likelihood, flag = self.bayesian_single_prob(zk_measurement,multiplicative=multiplicative)
            if flag == True:
                if multiplicative:
                    semantic_probability = semantic_probability * zk_likelihood
                    semantic_probability += regularization_factor
                    semantic_probability = semantic_probability / np.sum(semantic_probability)
                else:
                    semantic_probability += zk_likelihood
        else:
            raise NotImplementedError
        
        # if np.abs(np.sum(semantic_probability)-1.0)>1e-6:
        #     print('todebug')    
        
        return semantic_probability, flag
    
    def is_valid_labels(self, zk_labels:dict):
        for name,score in zk_labels.items():
            if name in self.openset_names:
                return True

        return False


class ObjectMap:
    def __init__(self,points,colors):
        self.points:np.array() = None # point cloud from the volumetric map
        self.segments:list[np.array()] = None # list of segments, each segment is a np.array of point indices
        # self.colors = colors
        self.instance_map:dict[str,Instance] = {}
        self.semantic_query_map = dict() # {class: [instance id]}
        self.max_instance_id = 1
    
    def load_dense_points(self,map_dir:str, voxel_size:float):
        dense_map = o3d.io.read_point_cloud(map_dir)
        self.points = np.asarray(dense_map.points,dtype=np.float32) # (N,3)
        # self.global_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(dense_map,voxel_size=voxel_size)
        print('load {} points from {}'.format(self.points.shape[0],map_dir))
    
    def load_segments(self,fn3:str):
        '''
        fn3: scannet segmentation file xxxx_vh_clean_2.0.010000.segs.json
        '''
        if os.path.exists(fn3)==False:
            print('segmentation file {} does not exist'.format(fn3))
        segments = read_scannet_segjson(fn3)
        self.segments = [np.array(segpoints) for segid,segpoints in segments.items()]
        print('Load {} segments'.format(len(self.segments)))

        # check_segments = np.concatenate(self.segments,axis=0)
        # print('{} unique indices'.format(np.unique(check_segments).shape[0]))
    
    def load_semantic_names(self,labels):
        self.labels = labels # closet names
    
    def insert_instance(self,instance:Instance):
        self.instance_map[str(instance.id)] = instance
        self.max_instance_id += 1
    
    def update_instance_with_global_map(self, global_pcd:o3d.geometry.PointCloud, active_instances:list, small_instance_size:int):
        if len(active_instances)<1: return False
        global_points = np.asarray(global_pcd.points,dtype=np.float32)
        remove_instances = []
        for idx in active_instances:
            if idx not in self.instance_map.keys(): continue
            voxels_number = self.instance_map[idx].update_voxels_from_dense_points(global_points,update_voxel_weight=-1,insert_new=False)
            if voxels_number<small_instance_size:
                remove_instances.append(idx)
        
        for idx in remove_instances:
            self.instance_map.pop(idx)  
            # active_instances.remove(idx)   
        print('remove {} instances after fused global volume'.format(len(remove_instances)))
        return True
    
    def update_volume_points(self):
        ''' Update points from voxel grid centroid '''
        for idx, instance in self.instance_map.items():
            instance.update_points()
            instance.uv_map = None
    
    def extract_instance_voxel_map(self):
        ''' Ensemble the voxel map of all the instances into One Instance Map'''
        voxel_coordinates = [] # (N,3), int32
        voxel_points = []
        instances = []
        semantics = []
        min_weights = 2
        
        for idx, instance in self.instance_map.items():
            label_id, conf = instance.estimate_label()
            buf_indices = instance.mhashmap.active_buf_indices()
            voxels = instance.mhashmap.key_tensor()[buf_indices].to(o3c.int32).numpy()
            weights = instance.mhashmap.value_tensor()[buf_indices].to(o3c.uint8).numpy()
            valid = np.squeeze(weights>=min_weights)
            
            if valid.sum()>0:
                voxels = voxels[valid,:]
                # weights = weights[valid,:]
                instance_points = voxels * instance.voxel_length + instance.voxel_length/2.0
                voxel_coordinates.append(voxels)
                voxel_points.append(instance_points)
                instances.append(instance.id * np.ones((voxels.shape[0],1),dtype=np.int32))
                semantics.append(label_id * np.ones((voxels.shape[0],1),dtype=np.int32))
        if len(voxel_coordinates)<1:
            return None,None,None,None
        voxel_coordinates = np.concatenate(voxel_coordinates,axis=0)
        voxel_points = np.concatenate(voxel_points,axis=0)
        instances = np.concatenate(instances,axis=0)
        semantics = np.concatenate(semantics,axis=0)
        instances = np.squeeze(instances)
        semantics = np.squeeze(semantics)
        print('Extract {} instances'.format(len(self.instance_map)))
        # print('Extract {}/{} voxels'.format(voxel_coordinates.shape[0],count))
        
        assert voxel_coordinates.shape[0] == voxel_points.shape[0], 'voxel_coordinates and voxel_points must have same number of rows'
        assert voxel_coordinates.shape[0] == instances.shape[0], 'voxel_coordinates and instances must have same number of rows'
        assert voxel_coordinates.shape[0] == semantics.shape[0], 'voxel_coordinates and semantics must have same number of rows'
        
        return voxel_coordinates, voxel_points, instances, semantics
    
    def extract_object_map(self,external_points = None,extract_from_dense_points=False,min_weight=1):
        '''
        If extract from dense points, the points are from the instance-wise points. 
        Otherwise, the points are from the fusing global point cloud into instance-wise voxel grid map.
        '''
        if external_points is None:
            global_dense_points = self.points
        else:
            global_dense_points = external_points
        
        assert global_dense_points is not None, 'dense map is not loaded'
        points_label = np.zeros(global_dense_points.shape[0]) -100
        count = 0
        msg =''
        # min_weight = 2
        
        for instance_id,instance in self.instance_map.items():
            if extract_from_dense_points:
                points_indices = instance.dense_points
            else:
                points_indices = instance.extract_aligned_points(global_dense_points, min_weight=min_weight)
            label_id,conf = instance.estimate_label()
            
            # print('{} has {} points'.format(self.labels[label_id],points_indices.shape[0]))
            if points_indices.size>0:
                composite_label = label_id*1000 + instance.id + 1
                points_label[points_indices] = composite_label #instance.id
                count +=1   
                msg += '{},'.format(self.labels[label_id])
        # print('{}/{} valid points'.format(np.sum(points_label>0),self.points.shape[0]))
        print('Extract {}/{} instances, {}/{} valid points, to visualize'.format(
            count,len(self.instance_map),np.sum(points_label>0),global_dense_points.shape[0]))
        print('Instance labels are: {}'.format(msg))
        return global_dense_points,points_label
    
    def save_debug_results(self,output_folder, vx_resolution, time_record):
            
        label_debug_file = open(os.path.join(output_folder,'fusion_debug.txt'),'w')
        label_debug_file.write('# mask_file label_id label_conf pos_observe neg_observe point_number label_name \n')
        count = 0
        for idx,instance in self.instance_map.items():
            active_voxels = instance.mhashmap.active_buf_indices().to(o3c.int64)
            
            if active_voxels.shape[0]>0:
                label_instance,label_conf = instance.estimate_label()

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
                    os.path.basename(mask_file),label_nyu40,label_conf,instance.pos_observed, instance.neg_observed,active_voxels.shape[0]))
                for det in instance.os_labels:
                    for os_name, score in det.items():
                        label_debug_file.write('{}:{:.4f}_'.format(os_name,score))
                    label_debug_file.write(',')

                label_debug_file.write('\n')
                instance.mhashmap.save('{}_hashmap.npz'.format(mask_file))
                # np.save('{}_voxels.npy'.format(mask_file),voxels)
                # np.savetxt('{}_pos.txt'.format(mask_file),points,fmt='%d')
                # if neg_points.shape[0]>3:
                #     np.savetxt('{}_neg.txt'.format(mask_file),instance.negative,fmt='%d')
                count +=1
        label_debug_file.write('# {}/{} instances extracted \n'.format(count,len(self.instance_map)))
        label_debug_file.write('# voxel resolution: {} \n'.format(vx_resolution))
        label_debug_file.write('# time record: {} \n'.format(time_record[0]))
        label_debug_file.write('# objects record: {:.0f} {:.0f} \n'.format(time_record[1][0],time_record[1][1]))
        # pred_file.close()
        label_debug_file.close()
        
    # def save_semantic_results(self, eval_dir, gt_points):
        
    
    def save_scannet_results(self, eval_dir, gt_points, semantic_eval_folder=None):
        '''
        Input,
            - eval_dir: str, the folder directory to save the results
            - points: N*3, np.float32, the point cloud provided by ScanNet
        '''
        MIN_DIST = 0.08
        MIN_POINTS = 1
        scene_name = os.path.basename(eval_dir)
        output_folder = eval_dir #os.path.join(eval_dir,scene_name)
        if os.path.exists(output_folder)==False: os.makedirs(output_folder)
        count = 0

        # Extract aliged points in multi-thread
        import multiprocessing as mp
        instance_list = []
        instance_labels = []
        instance_points = []
        for idx, instance in self.instance_map.items():
            label_id,label_conf = instance.estimate_label()
            assert label_id<20, 'contain unexpected label id'.format(label_id)
            instance_labels.append(label_id)
            instance_list.append(idx)
            instance_points.append(instance.points)
        
        p = mp.Pool(32)
        points_indices_list = p.map(extract_instance_points_thread, [(gt_points,points,MIN_DIST) for points in instance_points])
        p.close()
        p.join()
        print('Points aligned')

        # Export results     
        pred_file = open(os.path.join(output_folder,'predinfo.txt'),'w')
        for i, idx in enumerate(instance_list):    
            label_id = instance_labels[i]
            points_indices = points_indices_list[i]
        
            if points_indices.size>MIN_POINTS and label_id<20:
                points_mask = np.zeros(gt_points.shape[0],dtype=np.uint8)
                points_mask[points_indices] = 1
                label_name = SEMANTIC_NAMES[label_id]
                label_nyu40 = SEMANTIC_IDX[label_id]
                mask_file = os.path.join(output_folder,'{}_{}.txt'.format(idx,label_name))

                if label_name=='shower curtain':
                    mask_file = os.path.join(output_folder,'{}_shower_curtain.txt'.format(idx))
                
                pred_file.write('{} {} {:.3f}\n'.format(os.path.basename(mask_file),label_nyu40,label_conf))
                np.savetxt(mask_file,points_mask,fmt='%d')
                count +=1
        pred_file.close()
    
        # Export semantic result
        if semantic_eval_folder is not None:
            semantic_output_labels = np.zeros(gt_points.shape[0],dtype=np.uint8)
            count = 0
            for i, idx in enumerate(instance_list):
                points_indices = points_indices_list[i]
                if points_indices.size>MIN_POINTS and label_id<20:
                    label_nyu40 = SEMANTIC_IDX[label_id]
                    semantic_output_labels[points_indices] = label_nyu40
                    count +=1 
            print('Extract {}/{} instances for semantic evaluation'.format(count,len(self.instance_map)))
            print('{}/{} points are annotated with semantic label'.format(np.sum(semantic_output_labels>0),semantic_output_labels.shape[0]))
            np.savetxt(os.path.join(semantic_eval_folder,'{}.txt'.format(scene_name)),semantic_output_labels,fmt='%d')
        
        print('Extract {}/{} instances to evaluate'.format(count,len(self.instance_map)))

    
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
    
    def fuse_instance_segments(self, merge_types, min_voxel_weight, min_segments = 200, segment_iou = 0.2):
        '''
        After merge segments, points from the segments are written to instance points.
        '''
        
        S = len(self.segments)
        J = len(self.instance_map)
        instance_indices = np.zeros((J,),dtype=np.int32) # (J,), indices \in [0,J) to instance idx
        iou = np.zeros((S,J),dtype=np.float32)

        # Find overlap between segment and instance
        merged_instances = []
        i_ = 0
        for idx, instance in self.instance_map.items():
            points_indices = instance.extract_aligned_points(self.points, min_weight=min_voxel_weight)
            label_id, _ = instance.estimate_label()
            assert label_id < len(self.labels)
            # assert instance_points.size>0, 'instance {} has no points'.format(idx)

            if self.labels[label_id] in merge_types:
                for s, seg_indices in enumerate(self.segments):
                    if seg_indices.size<min_segments: continue
                    overlap = np.intersect1d(seg_indices,points_indices)
                    iou[s,i_] = overlap.size/(seg_indices.size) #+instance_points.size-overlap.size)
            else:
                merged_instances.append(int(idx))
                instance.dense_points = points_indices
                
            instance_indices[i_] = int(idx)
            i_+=1
        print('{} instances skip fuse segments'.format(len(merged_instances)))
        print('{}/{} overlaped pairs'.format(np.sum(iou>1e-3),iou.size))
        
        # Merge instance with segments
        count = 0
        for s, seg_indices in enumerate(self.segments):
            parent_instances = iou[s,:]>segment_iou
            if parent_instances.sum()>0: count +=1
            # seg_points = self.points[seg_indices.astype(np.int32)].astype(np.float32)
            
            for parent_instance_idx in instance_indices[parent_instances]:
                root_instance = self.instance_map[str(parent_instance_idx)]
                root_instance.merge_segment(seg_indices.astype(np.int32))
                if parent_instance_idx not in merged_instances:merged_instances.append(parent_instance_idx)
            
        print('{}/{} segments are merged into {}/{} instances'.format(
            count,len(self.segments),len(merged_instances),len(self.instance_map)))

        # Remove instance without valid geometric segments
        for j in np.arange(J):
            instance_id = instance_indices[j]
            if instance_id not in merged_instances:
                del self.instance_map[str(instance_id)]
        return None
    
    def merge_overlap_instances(self, active_instances, nms_iou, nms_similarity,inflat_ratio=None):
        '''
        Input:
        - active instances: list of instance ids, sorted from small to large
        '''
        if len(active_instances)<2:
            return None

        # Na = len(active_instances)
        merge_pairs = [] # [(a,b)]
        leaf_instances = [] # for debug. it records the instances are merged.
        
        # find pairs to be merged. instance_a is smaller than instance_b
        for a, id_a in enumerate(active_instances[:-1]):
            instance_a = self.instance_map[str(id_a)]
            label_a,_ = instance_a.estimate_label()
            # if self.labels[label_a]=='cabinet':
            #     print(instance_a.get_normalized_probability())
            for b,id_b in enumerate(active_instances[a+1:]):
                instance_b = self.instance_map[str(id_b)]
                semantic_similarity = instance_a.get_normalized_probability().dot(instance_b.get_normalized_probability())
                assert instance_a.points.shape[0]<=instance_b.points.shape[0], 'instance_a must be smaller than instance_b'
                
                if semantic_similarity>nms_similarity:             
                    recalled_voxels = instance_b.query_from_point_cloud(instance_a.points,inflat_ratio=inflat_ratio)
                    overlap = recalled_voxels.shape[0]/instance_a.points.shape[0]
                    if overlap>nms_iou:
                        label_b, _ = instance_b.estimate_label()
                        assert id_b not in leaf_instances, 'instance {} is already in the merged list'.format(id_b)
                        merge_pairs.append((id_a,id_b))
                        leaf_instances.append(id_a)
                        # print('{}-{}'.format(self.labels[label_a],self.labels[label_b]))
                        break # instance_a can only be merged into one instance_b
        
        # merge pairs
        # if len(merge_pairs)>0:
        #     print('!!!!!!!!! {} instance pairs are merged !!!!!!!!!!!!!'.format(len(merge_pairs)))
            
        for (id_a,id_b) in merge_pairs:
            instance_a = self.instance_map[str(id_a)]
            instance_b = self.instance_map[str(id_b)]
            instance_b.merge_volume(instance_a.mhashmap)

            instance_b.prob_vector += instance_a.prob_vector
            instance_b.prob_weight += instance_a.prob_weight
            instance_b.os_labels += instance_a.os_labels
            del self.instance_map[id_a]
        
        return True
  
    def merge_conflict_instances(self,nms_iou=0.1,nms_similarity=0.2):
        J = len(self.instance_map)
        instance_indices = np.zeros((J,),dtype=np.int32) # (J,), indices \in [0,J) to instance idx
        proposal_points = np.zeros((J,self.points.shape[0]),dtype=np.int32) # (J,N)
        
        # Calculate iou
        i =0
        for idx, instance in self.instance_map.items():
            if instance.dense_points.size<1: continue
            proposal_points[i,instance.dense_points] = 1
            instance_indices[i] = int(idx)
            i+=1
            
        intersection = np.matmul(proposal_points,proposal_points.T) # (J,J)
        proposal_points_number = np.sum(proposal_points,axis=1)+1e-6 # (J,)
        proposal_pn_h = np.tile(proposal_points_number,(J,1)) # (J,J)
        proposal_pn_v = np.tile(proposal_points_number,(J,1)).T # (J,J)
        ious = intersection/(proposal_pn_h+proposal_pn_v-intersection) # (J,J)
        # ious = intersection / np.minimum(proposal_pn_h,proposal_pn_v) # (J,J)
        scores = proposal_points_number # (J,)
        
        # NMS
        pick_idxs, merge_groups = self.non_max_suppression(ious,scores,threshold=nms_iou) 
        count = 0
        for root_idx, leaf_idxs in merge_groups.items(): # merge supressed instances
            root_instance = self.instance_map[str(instance_indices[int(root_idx)])]
            for leaf_idx in leaf_idxs:
                assert leaf_idx not in pick_idxs, 'merged leaf instance must be removed'
                leaf_instance = self.instance_map[str(instance_indices[leaf_idx])]
                assert root_instance.dense_points.size >= leaf_instance.dense_points.size, 'root instance must have more points'
                
                similarity = root_instance.get_normalized_probability().dot(leaf_instance.get_normalized_probability())
                if similarity>nms_similarity:
                    # label_id, score = root_instance.estimate_label()
                    # print('Merge instance {} to {}'.format(leaf_instance.id,root_instance.id))
                    root_instance.merge_segment(leaf_instance.dense_points)
                    root_instance.prob_vector += leaf_instance.prob_vector
                    count +=1
                else:
                    pick_idxs = np.concatenate((pick_idxs,np.array([leaf_idx],dtype=np.int32)),axis=0)
        print('{}/{} root instances are kept. {} Leaf instances are merged into the root one.'.format(len(pick_idxs),J, count))
        
        # Remove filtered instances
        for j in np.arange(J):
            instance_id = instance_indices[j]
            if j not in pick_idxs:
                del self.instance_map[str(instance_id)]
        
    def remove_small_instances(self, min_points):
        
        remove_idx = []
        for idx, instance in self.instance_map.items():
            if instance.dense_points.size<min_points:
                remove_idx.append(idx)
                
        for idx in remove_idx:
            del self.instance_map[idx]
            # print('Remove instance {} with {} points'.format(idx,instance.filter_points.size))
        
        print('remove {} small instances'.format(len(remove_idx)))
    
    def remove_rs_small_instances(self,max_negative_observe,min_points):
        ''' Incrementally remove small instances '''
        remove_instances = []
        for idx, instance in self.instance_map.items():
            if instance.neg_observed>=max_negative_observe and instance.points.shape[0]<min_points:
                remove_instances.append(idx)
                print('Remove instance {} with {} points'.format(idx,instance.points.shape[0]))
    
        if len(remove_instances)>0:
            print('remove {} small instances'.format(len(remove_instances)))    
        for idx in remove_instances:
            del self.instance_map[idx]

    
    def verify_curtains(self):
        if len(self.semantic_query_map)<1: 
            print('update the query map before refine spatial objects')
            return None
        
        MAX_DISTANCE = 2.0
        
        curtain_list = []
        if 'curtain' in self.semantic_query_map:
            curtain_list += self.semantic_query_map['curtain']
        if 'shower curtain' in self.semantic_query_map:
            curtain_list += self.semantic_query_map['shower curtain']
        for curtain_idx in curtain_list:
            curtain_instance = self.instance_map[curtain_idx]
            curtain_centroid = curtain_instance.centroid
            in_bath_trigger = False
            
            bath_instance_list = []
            if 'bathtub' in self.semantic_query_map:
                bath_instance_list += self.semantic_query_map['bathtub']
            if 'toilet' in self.semantic_query_map:
                bath_instance_list += self.semantic_query_map['toilet']

            for id_b in bath_instance_list:
                bath_position = self.instance_map[id_b].centroid
                dist = np.linalg.norm(curtain_centroid-bath_position)
                if dist<MAX_DISTANCE:
                    in_bath_trigger = True
                    break
                
            curtain_instance.prob_vector = np.zeros((len(self.labels),),dtype=np.float32)
            if in_bath_trigger:
                curtain_instance.prob_vector[self.labels.index('shower curtain')] = 1.0
            else:
                curtain_instance.prob_vector[self.labels.index('curtain')] = 1.0
                
        print('{} curtains are verified'.format(len(curtain_list)))
    
    def merge_background(self,merge_categories=['floor']):
        
        for category in merge_categories:
            if category in self.semantic_query_map:
                merge_instances = self.semantic_query_map[category]
                if len(merge_instances)<2: continue
                root_instance_id = None
                root_instance_size = 0
                
                # find the root instance
                for idx in merge_instances:
                    instance = self.instance_map[idx]
                    if instance.points.shape[0]>root_instance_size:
                        root_instance_id = idx
                        root_instance_size = instance.points.shape[0]
        
                # merge other instances into the root instance
                root_instance = self.instance_map[root_instance_id]
                for idx in merge_instances:
                    if idx!=root_instance_id:
                        instance = self.instance_map[idx]
                        root_instance.merge_volume(instance.mhashmap)
                        del self.instance_map[idx]
        
                print('{} {} are merged into one'.format(len(merge_instances),category))
    
    def update_semantic_queries(self):
        
        for idx, instance in self.instance_map.items():
            labe_id, score = instance.estimate_label()
            label_name = self.labels[labe_id]
            if label_name not in self.semantic_query_map:
                self.semantic_query_map[label_name] = [idx]
            else:
                self.semantic_query_map[label_name].append(idx)
    
    def update_instances_voxel_grid(self,device):
        for idx, instance in self.instance_map.items():
            instance_xyz = self.points[instance.dense_points,:]
            instance.update_voxels_from_dense_points(instance_xyz,device=device)         
        print('ALL Instances update voxel grid')
    
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

def load_pred(predict_folder, frame_name, valid_openset_names=None):
    '''
    Output: a list of detections
    '''
    label_file = os.path.join(predict_folder,'{}_label.json'.format(frame_name))
    mask_file = os.path.join(predict_folder,'{}_mask.npy'.format(frame_name))
    if os.path.exists(label_file)==False or os.path.exists(mask_file)==False:
        return None, None
    
    mask = np.load(mask_file) # (M,H,W), int, [0,1]
    img_height = mask.shape[1]
    img_width = mask.shape[2]
    detections:list(Detection) = []
    # valid_detections = []
    labels_msg =''
    MAX_BOX_RATIO=0.9
    # print('pred image dimension: {}x{}'.format(img_width,img_height))
    
    # import torch, torchvision

    with open(label_file, 'r') as f:
        json_data = json.load(f)
        tags = json_data['tags'] if 'tags' in json_data else None
        masks = json_data['mask']
        raw_tags = None
        invalid_detections = []
        box_list = []
        
        if 'raw_tags' in json_data:
            raw_tags = json_data['raw_tags']
        for ele in masks:
            if 'box' in ele:
                # if label[-1]==',':label=label[:-1]
                instance_id = ele['value']-1    
                bbox = ele['box']  
                labels = ele['labels'] # {label:conf}

                box_area_normal = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(img_width*img_height)
                if box_area_normal > MAX_BOX_RATIO: continue
                # if (bbox[2]-bbox[0])/img_width>MAX_BOX_RATIO and (bbox[3]-bbox[1])/img_height>MAX_BOX_RATIO:
                #     continue # remove bbox that are too large
                if valid_openset_names is not None:
                    valid = False
                    for label in labels:
                        if label in valid_openset_names:
                            valid = True
                            break
                    if valid==False: continue
                z_ = Detection(bbox[0],bbox[1],bbox[2],bbox[3],labels)
                # box = torch.tensor(z_.get_bbox()).unsqueeze(0)  
                # box_list.append(box)          
                z_.add_mask(mask[instance_id,:,:]==1)
                detections.append(z_)
                
            else: # background
                continue  
        f.close()
        
        
        #todo: check overlap between bbox. scene0011/frame500
        valid_detections = []
        iou_threshold=0.95
        max_overlap = 2
        # box_list = torch.cat(box_list,dim=0)
        # iou = torchvision.ops.box_iou(box_list,box_list)
        # for id_a, det_a in enumerate(detections):
        #     box_a = det_a.get_bbox()
        #     for id_b, det_b in enumerate(detections):
        #         if id_a==id_b:continue
        #         box_b = det_b.get_bbox()
                
        
        # print('invalid :{}'.format(invalid_detections))
        # detections = [detections[k] for k in range(len(detections)) if k not in invalid_detections]
        
        print('{}/{} detections are loaded in {}'.format(len(detections),len(masks)-1, frame_name))
        return tags, detections

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
                    
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        
    return np.array(pick,dtype=np.int32)

def filter_overlap_detections(detections:list(), min_iou:float=0.5):
    K = len(detections)
    if K<3: return detections
    ious = np.zeros((K,K),dtype=np.float32)
    volumes = np.zeros((K,),dtype=np.float32)

    # compute scores and iou    
    for i, zi in enumerate(detections):
        uv_i = zi.mask # (H,W), bool, detection mask
        for j, zj in enumerate(detections):
            if j==i:ious[i,j]=0.0;continue
            uv_j = zj.mask # (H,W), bool, detection mask
            overlap = np.logical_and(uv_i,uv_j)
            ious[i,j] = np.sum(overlap)/(np.sum(uv_i)+np.sum(uv_j)-np.sum(overlap))
        volumes[i] = (np.sum(uv_i))

    volumes = volumes/np.max(volumes)
    scores = np.exp(-volumes)

    # nms
    pick_idxs = non_max_suppression(ious,scores,threshold=min_iou)
    if len(pick_idxs)<K:
        detections = [detections[k] for k in pick_idxs]
        print('!!!!!!!!!remove {} overlaped detections!!!!!!'.format(K-len(pick_idxs)))
    return detections

def filter_detection_depth(detections:list(),depth:np.ndarray):
    K = len(detections)
    if K<3: return detections
    
    min_depth = 2.0
    remove_detection = []
    
    for k,zk in enumerate(detections):
        if 'floor' in zk.get_label_str():
            zk_depth = np.zeros(zk.mask.shape,dtype=np.float32)
            zk_depth[zk.mask] = depth[zk.mask]
            valid_mask = zk_depth > min_depth
            zk.mask = np.logical_and(zk.mask,valid_mask)
            if np.sum(zk.mask)<50:
                remove_detection.append(k)
    
    for k in remove_detection:
        del detections[k]
            
    return detections

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
        print('processing {}'.format(frame_name))
        
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

        # continue
        # projection
        mask, points_uv, _, _ = project_util.project(
            points, normals,T_wc, K_rgb_out, rgb_out_dim, 5.0, 0.5) # Nx3
        filter_mask = project_util.filter_occlusion(points_uv,depth,
                                                    max_dist_gap,depth_kernal_size,kernal_valid_ratio,kernal_max_var)
        
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
                new_instance.save_label_measurements(prob_current,weight)
                instance_map.insert_instance(new_instance)
                z.assignment = 'new'
                
            else: # integration
                j = np.argmax(assignment[k,:])
                matched_instance = instance_map.instance_map[str(j)]
                matched_instance.save_label_measurements(prob_current, weight)
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
    
    # return None
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

    if DATASET=='ScanNet':
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
    # scans = ['255']
    
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
    
