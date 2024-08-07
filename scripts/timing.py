import os, sys
import numpy as np


class TimeAnalysis():
    def __init__(self, header_=None):
        self.header = [''] # (S,)
        if header_ is not None:
            self.header = header_
        self.frames_data = [] # (K,S)
    
    def add_frame_data(self, data: np.ndarray, preprocess=False):
        if preprocess:
            data = self.pre_process(data)
            data = self.pre_process(data)
        # data = self.pre_process(data)
        self.frames_data.append(data)

    def pre_process(self, data_matrix: np.ndarray):
        ''' Remove a row with maximum value  and a row with minimum value'''
        if data_matrix.shape[0]<3:
            return data_matrix
        
        max_idx = np.argmax(data_matrix.sum(axis=1))
        min_idx = np.argmin(data_matrix.sum(axis=1))
        data_matrix = np.delete(data_matrix, [max_idx, min_idx], axis=0)
        return data_matrix
        

    def analysis(self,verbose=False, save_path=None):
        if(len(self.frames_data)==0):
            return
        summary_data_matrix = np.vstack(self.frames_data) # (K,S)
        K = summary_data_matrix.shape[0]
        assert(summary_data_matrix.ndim==2)

        S = summary_data_matrix.shape[1]
        average_data = np.mean(summary_data_matrix, axis=0) # (S,)
        average_frame_sum_time = np.mean(np.sum(summary_data_matrix, axis=1)) 

        if verbose:
            msg = '----------- Summary {} frames -----------\n'.format(K)
            for ele in self.header:
                msg += ele + ' '
            msg += '  Sum\n'

            for i in range(S):
                msg += '  {:.1f}   '.format(average_data[i])
            msg += '  {:.1f} \n'.format(average_frame_sum_time)
            
            print(msg)

        return summary_data_matrix