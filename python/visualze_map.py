import os, glob 
import open3d as o3d
import numpy as np


if __name__ == '__main__':
    
    scene_dir = '/media/lch/SeagateExp/dataset_/ScanNet/val/scene0064_00'
    map_dir = os.path.join(scene_dir,'scene0064_00_semantic.ply')
    

    
    frame_sequence = glob.glob(os.path.join(scene_dir,'pose','*.txt'))
    positions = []
    lines = []
    
    for frame_name in sorted(frame_sequence):
        pose_dir = frame_name
        T_wc = np.loadtxt(pose_dir)
        positions.append(T_wc[:3,3])
        if len(positions) > 1:
            lines.append([len(positions)-2,len(positions)-1])
    
    # exit(0)
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pcd = o3d.io.read_point_cloud(map_dir)
    o3d.visualization.draw_geometries([pcd,lineset])
