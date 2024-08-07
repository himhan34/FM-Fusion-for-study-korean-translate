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
            f.write(' '.join(['{:.3f}'.format(x) for x in scenes_result_array[i]])+'\n')
        
        f.close()
