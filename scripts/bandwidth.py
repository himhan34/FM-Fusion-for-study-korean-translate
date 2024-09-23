import os
import numpy as np

class BandwidthEvaluator():
    def __init__(self) -> None:
        # nodes and points number in each frame
        # frame number is N
        self.nodes = [] # (N,1)
        self.points = [] # (N,1)
        
        self.dim_nodes_float = 128 + 3 # (dim,), [feature, centroids] in float
        self.dim_nodes_int16 = 2 # (dim,), [node_id, instance_id] in int16
        self.dim_points_float = 3 # (dim,), [x,y,z,] in float
        self.dim_points_int16 = 1 # (dim,), [node_id] in int16
        self.dim_points_light = 256 + 3 # (dim,), [feature, centroids] in float

    def update(self, dir):
        ''' read from a scene bandwidth log file'''
    
        with open(dir, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if '#' in line: continue
                elements = line.split(' ')        
                nodes_number = int(elements[4])
                points_number = int(elements[5])
                
                self.nodes.append(nodes_number)
                self.points.append(points_number)
                count += 1
                
            print('Update {} frames of bandwidth from {}'.format(count, dir))
    
    def calculate(self, num_nodes, num_points):
        sum_bw_nodes = ((num_nodes * self.dim_nodes_float * 4 ) +
                        (num_nodes * self.dim_nodes_int16 * 2)) / 1024
        sum_bw_points = ((num_points * self.dim_points_float * 4 ) +
                         (num_points * self.dim_points_int16 * 2)) / 1024
        
        return sum_bw_nodes, sum_bw_points    
    
    def get_bandwidth(self):
        ''' get the average bandwidth of the scene'''
        if len(self.nodes)<1: 
            print('No bandwidth data available')
            return
        nodes = np.array(self.nodes)
        points = np.array(self.points)
        lt_points = 128 * nodes # Each node has 128 points
        
        print('---------- Summary Bandwidth {} frames -----------'.format(nodes.shape[0]))

        # bandwidth in KBytes
        sum_bw_nodes, sum_bw_points = self.calculate(np.sum(nodes), np.sum(points))
        sum_bw_ltpoints = np.sum(lt_points) * self.dim_points_light * 4 / 1024
        
        total_bw = sum_bw_points + sum_bw_nodes
        
        print('{} nodes, {} points'.format(np.sum(nodes), np.sum(points)))
        print('Our Node Bandwidth: {:.1f} KB, Points Bandwidth: {:.1f} KB, Total: {:.1f} KB, Ave: {:.1f}'.format(
            sum_bw_nodes, sum_bw_points,total_bw,total_bw/nodes.shape[0]))
        print('Light Points Bandwidth: {:.1f} KB'.format(sum_bw_ltpoints))

        dense_frames_maks = points>0
        nodes = nodes[~dense_frames_maks]
        points = points[~dense_frames_maks]
        sum_bw_nodes, sum_bw_points = self.calculate(np.sum(nodes), np.sum(points))
        total_bw = sum_bw_points + sum_bw_nodes
        
        coarse_frame_number = nodes.shape[0] - dense_frames_maks.sum()
        print('Coarse frames: ', (nodes.shape[0]-dense_frames_maks.sum()))
        print('Coarse mode: {} nodes, {} points'.format(np.sum(nodes), np.sum(points)))
        print('Coarse Node Bandwidth: {:.1f} KB, Points Bandwidth: {:.1f} KB, Total: {:.1f} KB, Avr: {:.1f}'.format(
            sum_bw_nodes, sum_bw_points,total_bw, total_bw/coarse_frame_number))
