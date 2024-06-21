import os 
import numpy as np

def load_eval_file(dir):
    
    with open(dir,'r') as f:
        lines = f.readlines()[1:]
        loop_frames = []
        for line in lines:
            if'#' in line:
                continue
            eles = line.split(',')
            tp = int(eles[0].strip())
            ap = int(eles[1].strip())
            rmse = float(eles[2].strip())
            iou = float(eles[3].strip())
            
            loop_frames.append({'tp':tp, 'ap':ap, 'rmse':rmse, 'iou':iou})
        
        return loop_frames
        



class RegistrationEvaluation:
    '''Evaluate scene graph registration result'''
    def __init__(self,ious_splits:list):
        self.ious = []
        self.tps = [] # true positives
        self.aps = [] # all positives
        self.rmses = []
        
        self.ious_splits = ious_splits

    def record_loop_frames(self, loop_frames:list):
        
        for frame in loop_frames:
            self.tps.append(frame['tp'])
            self.aps.append(frame['ap'])
            self.rmses.append(frame['rmse'])
            self.ious.append(frame['iou'])
                
    def analysis(self):
        K = len(self.ious)
        SEG = len(self.ious_splits) - 1
        RMSE_THRESH = 0.2
        print('Summary {} loop frames in {} split'.format(K, SEG))

        self.ious = np.array(self.ious)
        self.tps = np.array(self.tps)
        self.aps = np.array(self.aps)
        self.rmses = np.array(self.rmses)
        
        for i in range(SEG):
            start = self.ious_splits[i]
            end = self.ious_splits[i+1]
            print('Split {}-{}:'.format(start,end))
            mask = (self.ious >= start) & (self.ious < end)
            tp = self.tps[mask].sum()
            ap = self.aps[mask].sum()
            inst_prec = tp / ap
            rmse = self.rmses[mask]
            recall = (rmse<RMSE_THRESH).sum() / np.sum(mask)
            
            
            print('{} frames, TP Inst.: {}/{}, Inst.P.:{:.3f}, RR.:{:.3f} Mean Reg Error: {:.3f}'.format(
                np.sum(mask),tp,ap,recall,inst_prec,np.mean(rmse)))


        
        
    


 