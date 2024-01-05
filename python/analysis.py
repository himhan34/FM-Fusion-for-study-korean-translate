import os, glob, sys
import json
import numpy as np
from collections import Counter

remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

nyu20_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
nyu20_aug_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

def converter(name_label_mapper_):
    label2name_mapper = {}
    for label,id in name_label_mapper_.items():
        label2name_mapper[id] = label
    return label2name_mapper

def load_detection_results(dir, class_info):
    '''
        class_info: {nyu40id: {viewed:_, prompts:[], detections:[]}}
    '''
    
    with open(dir,'r') as f:
        data = json.load(f)
        associations = data['associations']
        MIN_VIEW = 1000

        for frame_name, frame_info in associations.items():
            # 'prompts','detection','instances'
            for instance in frame_info['instances']:
                gt_label = instance['label']
                if gt_label =='refridgerator': gt_label = 'refrigerator'
                if gt_label not in class_info:
                    class_info[gt_label] = {'viewed':0,'prompts':[],'detections':[]}
                
                if int(instance['viewed'])<MIN_VIEW:
                    continue                
                elif instance['det']>=0:
                    detection = frame_info['detections'][str(instance['det'])]
                    for name, score in detection.items():                
                        class_info[gt_label]['detections'].append(name)
                
                class_info[gt_label]['viewed'] += 1
                # class_info[gt_label]['prompts'] += 
                tags = frame_info['prompts'][0].split('.')
                for tag in tags:
                    class_info[gt_label]['prompts'].append(tag.strip())
        
        return class_info

def load_image_results(dir, image_info):
    '''image_info={scan_name: {viewed:_, prompts:[]}}'''
    
    with open(dir,'r') as f:
        data = json.load(f)
        images = data['ram']
        # MIN_VIEW = 100
        count_gt_sofa_tags = 0

        
        for frame_name, frame_info in images.items():
            # unique observation list
            observation = frame_info['observed']
            unique_observations = []
            for obs in observation:
                if obs not in unique_observations: unique_observations.append(obs)
            if 'refridgerator' in unique_observations:
                unique_observations[unique_observations.index('refridgerator')] = 'refrigerator'
                
            # unique tag list
            unique_tags = [tag.strip() for tag in frame_info['prompts'].split('.')] 
                
            # update for each observed class
            for gt_label in unique_observations:
                if gt_label not in image_info:
                    image_info[gt_label] = {'viewed':1,'prompts':unique_tags}
                else:
                    image_info[gt_label]['viewed'] += 1
                    # image_info[gt_label]['prompts'] += unique_tags
                    updated_prompts = image_info[gt_label]['prompts'] + unique_tags
                    image_info[gt_label]['prompts'] = updated_prompts

                if gt_label=='sofa': #and tag=='bookshelf':
                    count_gt_sofa_tags += len(unique_tags)

        return len(images)        

def extract_ram_matrix(image_info, prompt_list):
    '''
        Output:
        - ram_likelihood: (|prompts|,|nyu20_aug_label_names|)
    '''
    
    ram_likelihood = np.zeros((len(prompt_list),len(nyu20_aug_label_names)),dtype=np.float32)
    
    for gt_label, gt_info in image_info.items():
        if gt_label =='debug':
            print('bksf-sofa:{}'.format(gt_info))
            continue
        prompt_hist = Counter(gt_info['prompts'])
        if gt_label not in nyu20_aug_label_names: continue
        col_id = nyu20_aug_label_names.index(gt_label)
        print('-----{} is viewed {} times-----'.format(gt_label,gt_info['viewed']))
        
        for p_name, p_count in prompt_hist.items():
            if p_name not in prompt_list: continue
            row_id = prompt_list.index(p_name)
            ram_likelihood[row_id,col_id] = p_count / gt_info['viewed']
            # print('{}:{},{:.3f}'.format(p_name,p_count,ram_likelihood[row_id,col_id]))
            if p_count> gt_info['viewed']:
                print('({},{}) has {}>{} viewed'.format(p_name,gt_label,p_count,gt_info['viewed']))
            
    return {'rows':prompt_list,'cols':nyu20_aug_label_names,'likelihood':ram_likelihood}

def extract_det_prompts_mat(class_info, prompt_list, J_=20):
    '''Organize the detection and prompt info.
        Output info: {rows:_, cols:_, likelihood:_, det_mat:_, prompt_mat:_}
    '''
    
    # prompt_hist = Counter(prompt_occurance)
    # prompt_list = [k for k,v in prompt_hist.items()]
    detect_count_matrix = np.zeros((len(prompt_list),J_),dtype=np.int32)
    given_count_matrix = np.zeros((len(prompt_list),J_),dtype=np.int32) + 1e-3
    MIN_PRIOR = 0.06
    MIN_DETECTION = 20
    MIN_PROMPTS = 10
    IGNORE_TYPES = ['blanket','bag','computer monitor','pillow','television','dish washer']
    IGNORE_TYPES += ['infant bed','urinal','counter'] # 
    # IGNORE_TYPES += ['dresser']
            
    # 1. Construct given and detection count matrix
    for gt_label, gt_info in class_info.items():
        row_prompt_hist = Counter(gt_info['prompts'])
        row_detect_hist = Counter(gt_info['detections'])
        if gt_label not in nyu20_aug_label_names: continue
        label20_id = nyu20_aug_label_names.index(gt_label)
        # print('{}({})'.format(gt_label,nyu20_label_idxs[label20_id]))
        # label20_id = int(remapper[label40_id])
        # if label20_id<0:continue
        # print('bb')
        
        for p_name, p_count in row_prompt_hist.items():
            if p_name not in prompt_list:continue
            row_id = prompt_list.index(p_name)
            if p_name in row_detect_hist and p_count>MIN_PROMPTS:
                detect_count_matrix[row_id,label20_id] += min(row_detect_hist[p_name],p_count)
                if row_detect_hist[p_name]>p_count:
                    print('[warning] os {} gt {} wrong:{}>{}'.format(p_name,nyu20_aug_label_names[label20_id],row_detect_hist[p_name],p_count))
            given_count_matrix[row_id,label20_id] += p_count
        # print('gt {}({}): {}'.format(gt_label,label20_id,detect_count_matrix[:,label20_id].sum()))
    
    # return 0
    
    # 2. Filter
    valid_rows = np.ones(len(prompt_list),dtype=np.bool_)
    skipped_names = []
    for row in range(len(prompt_list)):
        if np.sum(detect_count_matrix[row,:])<20 or prompt_list[row] in IGNORE_TYPES:
            valid_rows[row] = False
            skipped_names.append(prompt_list[row])
    print('{}/{} valid prompts'.format(np.sum(valid_rows),len(prompt_list)))
    print('skip names:{}'.format(skipped_names))
    prompt_list = [prompt_list[row] for row in range(len(prompt_list)) if valid_rows[row]]
    detect_count_matrix = detect_count_matrix[valid_rows,:]
    given_count_matrix = given_count_matrix[valid_rows,:]
    
    # filter cells with variances
    # detection_priors = normalize(detect_count_matrix, norm='l1', axis=1)
    filter_cells = detect_count_matrix < MIN_DETECTION
    # filter_cells = detection_priors<MIN_PRIOR
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
    
    hardcode_cells = {'dresser':{'refrigerator':0.2}}
    for os_name, gt_dict in hardcode_cells.items():
        for gt_name, probability in gt_dict.items():
            os_id = prompt_list.index(os_name)
            gt_id = nyu20_aug_label_names.index(gt_name)
            likelihood[os_id,gt_id] = probability

    # 5. Check the sum of each row
    for row in range(len(prompt_list)):
        if np.sum(likelihood[row,:])<0.01:
            # print('row {} sum {}'.format(row,np.sum(likelihood[row,:])))
            print('[WARNNING!!!] openset type {} has zero probability vector! It is detected {} times'.format(
                prompt_list[row], np.sum(detect_count_matrix[row,:])))
            # print('row {} : {}'.format(row,likelihood[row,:]))

    print('Construct likelihood matrix between {} openset prompts and {} instances'.format(len(prompt_list),J_))
    return {'rows':prompt_list,'cols':nyu20_aug_label_names,'likelihood':likelihood,
            'det_mat':detect_count_matrix,'prompt_mat':given_count_matrix}

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

def create_kimera_probability(dir, valid_opensets):
    # 'benchmark/output/categories.json'
    ## Baseline from Fusion++
    empirical_association = json.load(open(dir,'r'))
    output_openset = []
    objects = empirical_association['objects']
    for gt_id, gt_name in enumerate(nyu20_aug_label_names):
        assert gt_name in objects
        openset_names = objects[gt_name]['main']
        for openset in openset_names:
            if valid_opensets is None: output_openset.append(openset)
            elif openset in valid_opensets and openset not in output_openset: 
                output_openset.append(openset)
            
    empirical_probability = np.zeros((len(output_openset),len(nyu20_aug_label_names))) + 0.1
    for gt_id, gt_name in enumerate(nyu20_aug_label_names):
        for openset in objects[gt_name]['main']:
            if openset in output_openset:
                openset_id = output_openset.index(openset)
                empirical_probability[openset_id, gt_id] = 0.9
    return {'rows':output_openset, 'cols':nyu20_aug_label_names, 'likelihood':empirical_probability}

def reorder_openset_names(openset_names, prompt_det_probability):
    '''
    Reorder openset names and likelihood matrix accordingly
    '''
    assert len(openset_names) == len(prompt_det_probability['rows']), 'openset names must be identical'
    reordered_names = []
    reorder_likelihood = np.zeros((len(openset_names),len(prompt_det_probability['cols'])))
    reorder_det_matrix = np.zeros((len(openset_names),len(prompt_det_probability['cols'])))
    reorder_prompt_matrix = np.zeros((len(openset_names),len(prompt_det_probability['cols'])))
    
    for k, name in enumerate(prompt_det_probability['rows']):
        assert name in openset_names, 'name {} not in openset_names'.format(name)
        new_k = openset_names.index(name)
        reorder_likelihood[new_k] = prompt_det_probability['likelihood'][k]
        if 'det_mat' in prompt_det_probability:
            reorder_det_matrix[new_k] = prompt_det_probability['det_mat'][k]
            reorder_prompt_matrix[new_k] = prompt_det_probability['prompt_mat'][k]
        
    prompt_det_probability['rows'] = openset_names
    prompt_det_probability['likelihood'] = reorder_likelihood
    if 'det_mat' in prompt_det_probability:
        prompt_det_probability['det_mat'] = reorder_det_matrix
        prompt_det_probability['prompt_mat'] = reorder_prompt_matrix
    return prompt_det_probability

def concat_openset_names(probability_map, dir):
    empirical_association = json.load(open(dir,'r'))
    opensets = empirical_association['opensets']
    K_ = len(probability_map['rows'])
    J_ = len(probability_map['cols'])
    MATCH_PROBABILITY = 0.99
    probability_map['rows'] += opensets
    probability_map['cols'] += opensets
    expand_likelihood = np.zeros((len(probability_map['rows']),len(probability_map['cols']))) + (1-MATCH_PROBABILITY)
    expand_likelihood[:K_,:J_] = probability_map['likelihood']
    
    for k, openset in enumerate(opensets):
        expand_likelihood[K_+k,J_+k] = MATCH_PROBABILITY
    
    return {'rows':probability_map['rows'], 'cols':probability_map['cols'], 'likelihood':expand_likelihood}

if __name__=='__main__':
    
    ################### SET PARAMTERS ###########################
    prediction_folder = 'prediction_augment' #'bayesian'
    results_dir = '/data2/ScanNet/measurements/'+prediction_folder
    output_folder = 'measurement_model'
    kimera_model_dir = output_folder+'/categories.json' # manually set hard association. given by ChatGPT
    ################### SET PARAMTERS ###########################

    files = glob.glob(os.path.join(results_dir,'*.json'))
    print('analysis {} scan results'.format(len(files)))

    ## Init
    count = 0
    count_detections = 0
    count_views = 0
    count_frames = 0
    class_summary_info = {} # {nyu40id: {viewed:_, prompts:[], detections:[]}}
    image_summary_info = {'debug':0} # {scan_name: {viewed:_, prompts:[]}}
    
    # Read and summarize data 
    for file in files:
        scan_name = os.path.basename(file).split('.')[0]
        print('reading {}'.format(scan_name))
        class_summary_info = load_detection_results(file, class_summary_info)
        count_frames = load_image_results(file, image_summary_info)
        count += 1
        # break
    
    for gtlabel, info in class_summary_info.items():
        count_detections += len(info['detections'])
        count_views += info['viewed']
        print('{}, {} viewed, prompts:{}, detections:{}'.format(gtlabel,info['viewed'],len(info['prompts']),len(info['detections'])))
    
    print('--------- summarize {} frames ---------'.format(count_frames))
    print('--------- summarize {} scans with {} instances ---------'.format(count,len(class_summary_info)))
    print('--------- {} detections, {} views ---------'.format(count_detections,count_views))
    # exit(0)
    openset_names = []
    
    for gt_id, match_info in class_summary_info.items():
        for name in match_info['detections']:
            if name not in openset_names:
                openset_names.append(name)
    print('available openset names:{}'.format(openset_names))
    # exit(0)

    print('--------------- analysis ----------------')
    # detections_gt_matrix = extract_det_matrix(class_pairs, openset_name_mapper)
    gdino_likelihood = extract_det_prompts_mat(class_summary_info, openset_names)
    ram_likelihood = extract_ram_matrix(image_summary_info, gdino_likelihood['rows'])
    # exit(0)

    kimera_probability = create_kimera_probability(kimera_model_dir, gdino_likelihood['rows'])
    gdino_likelihood = reorder_openset_names(kimera_probability['rows'], gdino_likelihood)
    ram_likelihood = reorder_openset_names(kimera_probability['rows'], ram_likelihood)
    
    print('--------------- Export Data ----------------')
    import export
    export.general_likelihood_matrix(ram_likelihood, os.path.join(output_folder,'bayesian'),'ram_likelihood')
    export.general_likelihood_matrix(gdino_likelihood, os.path.join(output_folder,'bayesian'),'detection_likelihood')
    multip_likelihood = {'likelihood':gdino_likelihood['likelihood']*ram_likelihood['likelihood'],
                         'rows':gdino_likelihood['rows'],
                         'cols':gdino_likelihood['cols']}
    export.general_likelihood_matrix(multip_likelihood, os.path.join(output_folder,'bayesian'), 'likelihood_matrix')
    
    export.general_likelihood_matrix(kimera_probability,os.path.join(output_folder,'hardcode'),'likelihood_matrix')
