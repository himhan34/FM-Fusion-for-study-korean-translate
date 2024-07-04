import os, glob
import subprocess


if __name__ == '__main__':
    # args
    dataroot = '/data2/sgslam'
    output_folder = os.path.join(dataroot,'output','two_agent+')
    scan_pairs = [['uc0110_00a', 'uc0110_00b'],
                  ['uc0110_00a', 'uc0110_00c'], # opposite trajectory
                  ['uc0115_00a', 'uc0115_00b'], # opposite trajectory
                  ['uc0204_00a', 'uc0204_00b'], # opposite trajectory
                  ['uc0204_00a', 'uc0204_00c'],
                  ['uc0107_00a', 'uc0107_00b']
                ]
    exe_dir = 'build/cpp/TestRegister'
    
    for pair in scan_pairs:
        print('******** processing pair: {} ***********'.format(pair))
        src_scene = pair[0]
        ref_scene = pair[1]
        frames_dirs = glob.glob(os.path.join(output_folder, src_scene, ref_scene, 'frame*.txt'))
        for frame_dir in sorted(frames_dirs):
            frame_name = os.path.basename(frame_dir).split('.')[0]
            print('   --- processing frame: {} ---'.format(frame_name))
            
        
        
            cmd = '{} --output_folder {} --src_scene {} --ref_scene {} --frame_name {}'.format(
                exe_dir, output_folder, src_scene, ref_scene, frame_name)
            
            subprocess.run(cmd, 
                        stdin=subprocess.PIPE,
                        shell=True)
        
            break
        break
        
        
        
        
        

    
    
    
    