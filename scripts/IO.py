import os, glob
import numpy as np 


def write_scenes_results(out_dir:str, 
                         scenes_names:list, 
                         scenes_result_array:np.ndarray,
                         header:str=None):
    N = len(scenes_names)
    # assert(N==scenes_result_array.shape[0])
    if N!=scenes_result_array.shape[0]:
        return None
    
    with open(out_dir, 'w') as f:
        if header is not None:
            f.write(header+'\n')
        
        for i in range(N):
            f.write('{}: '.format(scenes_names[i]))
            f.write(' '.join(['{:.1f}'.format(x) for x in scenes_result_array[i]])+'\n')
        
        f.close()

def load_scenes_results(dir:str):
    scenes_names = []
    keys = []
    scenes_results = []
    with open(dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line: 
                keys = line[1:].strip().split(' ')
                keys = [x.strip() for x in keys]
                keys = keys[1:]
            else:
                eles = line.strip().split()
                src_name = eles[0].split('-')[0].strip()
                ref_name = eles[0].split('-')[1].strip()
                scenes_names.append([src_name, ref_name])
                scenes_results.append([float(x) for x in eles[1:]])
        
        f.close()
        scenes_results = np.array(scenes_results)
        
        #
        output_dict = {'scenes_names':scenes_names}
        for key, result in zip(keys, scenes_results.T):
            output_dict[key] = result
        
        return output_dict

def read_nodes_matches(dir:str):
    src_centroids = []
    ref_centroids = []
    with open(dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#' in line: continue
            eles = line.strip().split()
            src_centroid = [float(x) for x in eles[2:5]]
            ref_centroid = [float(x) for x in eles[5:8]]
            src_centroids.append(np.array(src_centroid))
            ref_centroids.append(np.array(ref_centroid))
        
        f.close()
        src_centroids = np.array(src_centroids)
        ref_centroids = np.array(ref_centroids)
        
        return src_centroids, ref_centroids