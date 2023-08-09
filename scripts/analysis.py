import os, glob, sys
import json
import numpy as np
import measure_model
from collections import Counter
import scannet_util

raw_label_mapper = scannet_util.read_label_mapping('/data2/ScanNet/scannetv2-labels.combined.tsv', 
                                                  label_from='raw_category', label_to='nyu40id')
name_label_mapper = scannet_util.read_label_mapping('/data2/ScanNet/scannetv2-labels.combined.tsv', 
                                                    label_from='nyu40class', label_to='nyu40id')
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

nyu20_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
nyu20_aug_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

def converter():
    label2name_mapper = {}
    for label,id in name_label_mapper.items():
        label2name_mapper[id] = label
    # print('{} labels'.format(len(label2name_mapper)))
    return label2name_mapper

label2name_mapper = converter()

class DefPrompt:
    def __init__(self, det_name):
        self.det_name = det_name
        
    def insert_prompts(self, prompts):
        self.prompts = prompts
        BATH_RPOMPTS = ['bath','tub','bathtub','toilet']
        if self.det_name =='curtain':
            promplist = self.extract_all_prompts()
            
            for prompt in BATH_RPOMPTS:
                if prompt in promplist:
                    print('Modify curtain to shower curtain!')
                    self.det_name = 'shower curtain'
                    break
            # print('find curtain with prompt: {}'.format(self.prompts))
    
    def extract_all_prompts(self):
        out = []
        for k,v in self.prompts.items():
            out.append(v)
        return out
        
    def extract_related_prompts(self):
        out = []
        for k,v in self.prompts.items():
            if self.det_name.find(v)!=-1:
                out.append(v)
        return out
    
    def find_specific_hist(self,prompts):
        prompt_hist = np.zeros(len(prompts),dtype=np.int32)
        prompt_list = self.extract_all_prompts()
        for i in np.arange(len(prompts)):
            if prompts[i] in prompt_list:
                prompt_hist[i] += 1
        
        return prompt_hist

def convert_curtain_name(det_name,prompts):
    if det_name=='curtain':
        convert = False
        STRONG_PROMPTS = ['bath','tub','bathtub','shower']
        for bath_word in STRONG_PROMPTS:
            if bath_word in prompts:
                convert = True
                break
        if convert:
            return 'shower curtain', prompts.replace('curtain','shower curtain')
        else: return det_name, prompts
    else:
        return det_name, prompts
        

def load_result(dir):
    instances = []
    with open(dir,'r') as f:
        data = json.load(f)
        assert 'frame_gap' in data, 'frame_gap not in json'
        instances_data = data['instances']
        label_idxs = []
        
        for j,instance_d in instances_data.items():
            nyu40_id = raw_label_mapper[instance_d['label_name']]
            if nyu40_id in label2name_mapper:
                nyu40_name = label2name_mapper[nyu40_id]
            else: nyu40_name = 'unknown'
            label_idxs.append(nyu40_name)
            lj = measure_model.Instance(j,nyu40_name,nyu40_id,-1,-1)
            measurements = []
            # print('{}:{}'.format(j,instance_d['label_name']))
            assert len(instance_d['measurements'])>0, 'instance {} has no measurements'.format(j)
            for zk_data in instance_d['measurements']:
                zk = measure_model.Measurement(zk_data['label_name'],zk_data['score'],zk_data['iou'],zk_data['view_points'])
                measurements.append(zk)
            instances.append({'instance':lj,'measurements':measurements})
        f.close()
        print('load results for {} instances'.format(len(instances)))
        # print('label_idxs:{}'.format(label_idxs))
        
        return instances

def load_prompt_results(dir, class_prompt_pairs,prompt_occurance):
    '''
    class_prompt_pairs: {gt_id: openset_prompts}
    '''
    with open(dir,'r') as f:
        data = json.load(f)
        instances = data['instances'] # {j: instance_j}
        associations = data['associations']
        for frame_name, frame_info in associations.items():
            prompts = frame_info['prompts']
            prompt_words = prompts['0'].split('.')
            for word in prompt_words: prompt_occurance.append(word.strip())
            
            # detections = frame_info['detections']
            # print('----- {} ------'.format(frame_name))
            for pair_info in frame_info['matches_gt']:
                assert pair_info['gt'] < len(instances), 'gt {} not in instances'.format(pair_info['gt'])
                match_instance = instances[pair_info['gt']]
                # match_det = detections[pair_info['det']]
                match_instance_nyu_id = raw_label_mapper[match_instance['label_name']]
                if match_instance_nyu_id in class_prompt_pairs:
                    for word in prompt_words:
                        class_prompt_pairs[match_instance_nyu_id].append(word.strip())
            
        
        return class_prompt_pairs, prompt_occurance
        
def load_detection_result(dir,class_pairs, class_hist, detection_occurance):
    '''
    Data Structure,
        1. class pair: {gt_id: openset_name_list}
        2. class_hist: 40, count the number of instances
    '''
    
    with open(dir,'r') as f:
        data = json.load(f)
        instances = data['instances'] # {j: instance_j}
        associations = data['associations']
        count_detections = 0
        count = 0
        
        for frame_name, frame_info in associations.items():
            prompts = frame_info['prompts']
            detections = frame_info['detections']
            # print('----- {} ------'.format(frame_name))
            for pair_info in frame_info['matches_gt']:
                assert pair_info['gt'] < len(instances), 'gt {} not in instances'.format(pair_info['gt'])
                match_instance = instances[pair_info['gt']]
                match_det = detections[pair_info['det']]
                match_instance_nyu_id = raw_label_mapper[match_instance['label_name']]
                if match_instance_nyu_id in class_pairs:     
                    msg = match_instance['label_name']+':'
                    for os_name, score in match_det['labels'].items():
                        if len(os_name.split(' '))>3: continue
                        if os_name=='curtain':
                            # print('{} convert curtain name!!!'.format(dir))
                            os_name, prompts = convert_curtain_name(os_name,prompts)
                        class_pairs[match_instance_nyu_id].append(os_name)
                        detection_occurance.append(os_name)
                        count += 1
                        msg += '{}({:.2f}) '.format(os_name,score)
                    class_hist[match_instance_nyu_id] +=1
                count_detections +=1
                
        print('{} frames, {} detections, {} pairs'.format(len(associations),count_detections,count))
        f.close()
        
        return class_pairs, class_hist, detection_occurance

def extract_det_prompts_mat(class_prompt_pairs,prompt_occurance,class_pairs,J_=20):
    prompt_hist = Counter(prompt_occurance)
    prompt_list = [k for k,v in prompt_hist.items()]
    detect_count_matrix = np.zeros((len(prompt_hist),J_),dtype=np.int32)
    given_count_matrix = np.zeros((len(prompt_hist),J_),dtype=np.int32) + 1e-3
    MIN_DETECTION = 100
    MIN_PROMPTS = 10
    IGNORE_TYPES = ['blanket','bag','computer monitor','pillow','television']
    # IGNORE_TYPES += ['glass door','shelf','bookshelf']
            
    # 1. Construct given and detection count matrix
    for nyu40_id, openset_prompts in class_prompt_pairs.items():
        row_prompt_hist = Counter(openset_prompts)
        row_detect_hist = Counter(class_pairs[nyu40_id])
        gt_id = int(remapper[nyu40_id])
        if gt_id<0: continue    # skip unknown types
        # if gt_id==0 or gt_id==1: continue # todo: add wall and floor
        
        for p_name, p_count in row_prompt_hist.items():
            row_id = prompt_list.index(p_name)
            if p_name in row_detect_hist and p_count>MIN_PROMPTS:
                detect_count_matrix[row_id,gt_id] += min(row_detect_hist[p_name],p_count)
                if row_detect_hist[p_name]>p_count:
                    print('[warning] os {} gt {} wrong:{}>{}'.format(p_name,nyu20_aug_label_names[gt_id],row_detect_hist[p_name],p_count))
            given_count_matrix[row_id,gt_id] += p_count
    
    # 2. Filter
    valid_rows = np.ones(len(prompt_hist),dtype=np.bool_)
    skipped_names = []
    for row in range(len(prompt_list)):
        if np.sum(detect_count_matrix[row,:])<MIN_DETECTION or prompt_list[row] in IGNORE_TYPES:
            valid_rows[row] = False
            skipped_names.append(prompt_list[row])
    print('{}/{} valid prompts'.format(np.sum(valid_rows),len(prompt_list)))
    print('skip names:{}'.format(skipped_names))
    prompt_list = [prompt_list[row] for row in range(len(prompt_list)) if valid_rows[row]]
    detect_count_matrix = detect_count_matrix[valid_rows,:]
    given_count_matrix = given_count_matrix[valid_rows,:]
    
    # filter cells
    filter_cells = detect_count_matrix < MIN_DETECTION
    detect_count_matrix[filter_cells] = 0
    
    # 3. Probability
    likelihood = detect_count_matrix / given_count_matrix
    
    # 4. Hardcode floor and wall
    hardcode_types = {} #{'wall':["wall","tile wall"],'floor':["floor","carpet"]}
    for gt_name, os_names in hardcode_types.items():
        gt_id = nyu20_aug_label_names.index(gt_name)
        likelihood[:,gt_id] = 0.02

        for os_name in os_names:
            os_id = prompt_list.index(os_name)
            likelihood[os_id,gt_id] = 0.9

    print('Construct likelihood matrix between {} openset prompts and {} instances'.format(len(prompt_list),J_))
    return {'rows':prompt_list,'cols':nyu20_aug_label_names,'likelihood':likelihood}

def extract_det_matrix(class_pairs, openset_name_mapper, J_=21):
    '''
        Association matrix (K,J), all openset names are used.
    '''
    
    K_ = len(openset_name_mapper)
    detections_gt_matrix = np.zeros((K_,J_),dtype=np.float32)
    
    for nyu40_id,openset_names in class_pairs.items():
        gt_id = int(remapper[nyu40_id])
        if gt_id<0: continue # unknonwn
        
        pred_labels = openset_names #[pair.det_name for pair in pairlist]
        openset_hist = Counter(pred_labels)
        assert gt_id<J_,'gt_id {} exceeds J_ {}'.format(gt_id,J_)
        
        for o_name, o_count in openset_hist.items():
            o_label_id = openset_name_mapper[o_name]
            assert o_label_id<K_,'o_label_id {} exceeds K_ {}'.format(o_label_id,K_)
            detections_gt_matrix[o_label_id,gt_id] = o_count
        
        # association_matrix[:,gt_id] /= np.sum(association_matrix[:,gt_id])
        
        print('nyu class {}({}): {}'.format(nyu20_aug_label_names[gt_id],nyu40_id,Counter(pred_labels)))
    
    return detections_gt_matrix

def extract_prompts_connections(class_pairs, J_ = 21):
    '''
    gt_prompts: {gt_name: {related_prompts:_, } }
    '''
    gt_prompts = dict()
    
    for nyu40_id,pairlist in class_pairs.items():
        gt_id = int(remapper[nyu40_id])
        if gt_id<0: col_id = J_-1 # unknonwn
        else: col_id = gt_id
        gt_name = nyu20_aug_label_names[col_id]
        
        related_prompts = []
        for pair in pairlist:
            related_prompts += pair.extract_related_prompts()
                
        if gt_name not in gt_prompts:
            gt_prompts[gt_name] = related_prompts
        else: gt_prompts[gt_name] += related_prompts

    return gt_prompts
    
def extract_det_prompts(class_pair,openset_names,detections_gt_matrix,gt_name='door'):
    '''
    Find for each detected name, the histogram of given prompt names
    hist_mat: (|top_prompts|,|topdet|+1)
    '''
    
    topdet = 8
    top_prompts = ['door','doorway','closet','cabinet','bathroom']
    nyu40id = name_label_mapper[gt_name]
    gt_id = int(remapper[nyu40id])
    if gt_id<0: gt_id = 20 # unknonwn
    
    valid_openset_flags = detections_gt_matrix[:,gt_id]>0
    valid_openset_names = np.array(openset_names)[valid_openset_flags]
    valid_openset_counts = detections_gt_matrix[valid_openset_flags,gt_id]
    srt_indices = np.argsort(valid_openset_counts)[::-1]
    valid_openset_names = valid_openset_names[srt_indices]
    valid_openset_counts = valid_openset_counts[srt_indices]
    
    pairlist = class_pair[nyu40id]
    hist_mat = np.zeros((topdet,len(top_prompts)+1),dtype=np.int32)
    
    for k in np.arange(topdet): # iterate over top detections
        det_name = valid_openset_names[k]
        det_count = valid_openset_counts[k]
        hist_col = np.zeros(len(top_prompts),dtype=np.int32)    # histogram of each prompt
        
        for pair in pairlist:
            if pair.det_name==det_name:        
                hist_col += pair.find_specific_hist(top_prompts)
            # prompts = pair.extract_all_prompts()
        
        hist_mat[k,:-1] = hist_col
        hist_mat[k,-1] = det_count
        # det_prompt = [pair for pair in pairlist if pair.det_name==det_name][0]
        # print('det {}({}): {}'.format(det_name,det_count,det_prompt.extract_related_prompts()))
    
    # print('matrix shape {}'.format(hist_mat.shape))
    return {'top_prompts':top_prompts,
            'valid_openset_names':valid_openset_names[:topdet], 
            'hist_mat':hist_mat}

def create_kimera_probability(dir):
    # 'benchmark/output/categories.json'
    ## Baseline from Fusion++
    empirical_association = json.load(open(dir,'r'))
    valid_openset = []
    objects = empirical_association['objects']
    for gt_id, gt_name in enumerate(nyu20_aug_label_names):
        assert gt_name in objects
        openset_names = objects[gt_name]['main']
        for openset in openset_names:
            if openset not in valid_openset: 
                valid_openset.append(openset)
            
    empirical_probability = np.zeros((len(valid_openset),len(nyu20_aug_label_names))) + 0.1
    for gt_id, gt_name in enumerate(nyu20_aug_label_names):
        for openset in objects[gt_name]['main']:
            openset_id = valid_openset.index(openset)
            empirical_probability[openset_id, gt_id] = 0.9
    return {'rows':valid_openset, 'cols':nyu20_aug_label_names, 'likelihood':empirical_probability}

if __name__=='__main__':
    results_dir = '/data2/ScanNet/measurements/prompts'
    output_folder = '/home/cliuci/code_ws/OpensetFusion/measurement_model/prompt'
    files = glob.glob(os.path.join(results_dir,'*.json'))
    print('analysis {} scan results'.format(len(files)))
    
    ## Init
    count = 0
    class_pairs = {i: [] for i in np.arange(40)} # {nyu40id: [detected_os_labels]}
    class_prompt_pairs = {i: [] for i in np.arange(40)} # {nyu40id: [given_os_prompts] }
    class_hist = np.zeros(40,np.int32)
    prompt_occurance = []
    detection_occurance = []
    
    # Read data
    for file in files:
        scan_name = os.path.basename(file).split('.')[0]
        print('processing {}'.format(scan_name))
        class_pairs, class_hist, detection_occurance = load_detection_result(file, class_pairs, class_hist, detection_occurance)
        class_prompt_pairs, prompt_occurance = load_prompt_results(file, class_prompt_pairs, prompt_occurance)
        count += 1
    
    class21_hist = np.zeros(21,np.int32)
    for i in np.arange(20):
        class21_hist[i] = class_hist[nyu20_label_idxs[i]]
    
    print('--------- summarize {} scans with {} instances ---------'.format(count,len(class_pairs)))

    prompt_histogram = Counter(prompt_occurance)
    det_histogram = Counter(detection_occurance)
    for k,v in det_histogram.items():
        if len(k.split(' '))>2:
            print('{}:{}'.format(k,v))
    
    exit(0)
    openset_names = []
    openset_name_mapper = {} # key: pred label name, value: id
    openset_id_mapper = {} # key: id, value: name
    # CHECK_TYPES =['room','kitchen','bedroom','bathroom','stool']
    
    for gt_id, openset_name_list in class_pairs.items():
        # print('{}:{}'.format(gt_id,openset_name_list))
        for openset_type in openset_name_list:
            # openset_type = pair.det_name
            if openset_type not in openset_name_mapper:
                id_ = len(openset_name_mapper)
                openset_name_mapper[openset_type] =id_
                openset_id_mapper[id_] = openset_type
                openset_names.append(openset_type)
                # if openset_type in CHECK_TYPES:
                #     print('find {}'.format(openset_type))
    # exit(0)
    # print('find {} openset names :{}'.format(len(openset_names),openset_names))

    print('--------------- analysis ----------------')
    detections_gt_matrix = extract_det_matrix(class_pairs, openset_name_mapper)
    prompt_det_probability = extract_det_prompts_mat(class_prompt_pairs,prompt_occurance,class_pairs)
    kimera_probability = create_kimera_probability('/home/cliuci/code_ws/OpensetFusion/measurement_model/categories.json')
    
    K_ = detections_gt_matrix.shape[0]
    J_ = detections_gt_matrix.shape[1]
    print('Construct association between {} gt instances and {} openset types'.format(J_,K_))
    # exit(0)

    # gt_prompts = extract_prompts_connections(class_pairs)
    
    # print(detections_gt_matrix.sum(axis=0))
    # print('find {} unique pred names:{}'.format(len(openset_name_mapper),openset_name_mapper))
    print('--------------- Export Data ----------------')
    import export
    export.likelihood_matrix(prompt_det_probability, output_folder,'prompt_likelihood')
    export.likelihood_matrix(kimera_probability,output_folder,'kimera_probability')
    exit(0)

    ASSOCIATION_THRESHOLD = 0.05
    likelihood_data =(detections_gt_matrix, openset_id_mapper, nyu20_aug_label_names)
    export.class_model(likelihood_data,output_folder,ASSOCIATION_THRESHOLD)
    # export.prompts_histogram(gt_prompts, output_folder, door_det_prompt)
    export.instance_histogram(nyu20_aug_label_names,class21_hist, output_folder)
    
    exit(0)
    
    nyu40_name_list = name_label_mapper.keys()
    openset_name_list = [k for k,v in openset_name_mapper.items()]
    assert len(nyu20_aug_label_names) == J_
    
    assert len(openset_name_list) == K_, 'openset_name_list:{} K_:{}'.format(len(openset_name_list),K_)
    
    out = (detections_gt_matrix, openset_id_mapper, nyu20_aug_label_names)
    # out = np.array([association_matrix, openset_id_mapper, nyu20_label_names]).astype(np.object_)
    np.save(os.path.join(output_folder,'association_matrix.npy'),out, allow_pickle=True)
    
    # with open(os.path.join(output_folder,'pair_measurements.json'),'w') as f:
    #     json.dump(match_pair_measurements,f)
    #     f.close()
    

    