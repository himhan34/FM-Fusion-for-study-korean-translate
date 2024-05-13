import os, glob 
import open3d as o3d 
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

def reverse_img(img):
    new_img = np.zeros_like(img)
    mask = img==255
    new_img[mask] = 0
    new_img[~mask] = 255
    return new_img
    
def segment_planes(pcd, visualize=True):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                         ransac_n=7,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    if visualize:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    return inlier_cloud, outlier_cloud

def detect_planer_patches(pcd):
    min_height = 2.0
    max_z_normal = 0.9
    
    #
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=60,
        outlier_ratio=0.6,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))

    geometries = []
    index = 0
    # line_points = []
    # line_indices = []
    wall_oboxes = []
    floor_oboxes = []
    for obox in oboxes:
        extent = obox.extent
        rot = np.array(obox.R)
        rot_vec= R.from_matrix(rot).as_rotvec()
        normal = normalize(rot_vec.reshape(1, -1), norm='l2').squeeze()

        bound = obox.get_max_bound() - obox.get_min_bound()

        # print(extent)

        if np.abs(normal[2])>max_z_normal: # skip horizontal planes
            print('Find floor plane')
            floor_oboxes.append(obox)
            continue
        
        if bound[2]<min_height: # skip short planes
            continue
        
        # print('bound', bound)

        # center = obox.get_center()
        # b = center + normal
        # line_points.append(center)
        # line_points.append(b)
        # line_indices.append([index, index+1])
        # index += 2

        
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
        wall_oboxes.append(obox)
    geometries.append(pcd)
    
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(line_points),
    #     lines=o3d.utility.Vector2iVector(line_indices),
    # )
    # geometries.append(line_set)
    print('export {} wall planes'.format(len(wall_oboxes)))
    print('export {} floor planes'.format(len(floor_oboxes)))

    o3d.visualization.draw_geometries(geometries,
                                    zoom=0.62,
                                    front=[0.4361, -0.2632, -0.8605],
                                    lookat=[2.4947, 1.7728, 1.5541],
                                    up=[-0.1726, -0.9630, 0.2071])

    return wall_oboxes, floor_oboxes

def bev_from_planes(wall_bboxes, floor_bboxes,out_folder):
    min_x = 1000.0
    min_y = 1000.0
    max_x = 0.0
    max_y = 0.0
    scale = 100
    corners_indice = [[0,1,2,3,4,5,6,7]]
    # corners_indice = [[0,1,2,7]]
    # corners_indice = [[0,7]]
    
    # get the min x, y
    for i, box in enumerate(wall_bboxes):
        center = box.get_center()
        for pt in box.get_box_points():
        
            if pt[0] < min_x:
                min_x = center[0]
            if pt[1] < min_y:
                min_y = center[1]
            if pt[0] > max_x:
                max_x = center[0]
            if pt[1] > max_y:
                max_y = center[1]
    
    
    print('min_x', min_x, 'min_y', min_y, 'max_x', max_x, 'max_y', max_y)
    offset = np.array([min_y,min_x]) - np.array([5.0,5.0])
    img_width = int(scale* (max_x-min_x))
    img_height = int(scale* (max_y-min_y))  
    print('img size ', img_height, img_width)
    img_width = 4800
    img_height = 3600
    
    
    bev = np.zeros((img_height,img_width)).astype(np.uint8)
    floor_bev = np.zeros_like(bev)
    print('offset', offset)
    print('img size', img_height, img_width)
    
    #
    for i,box in enumerate(wall_bboxes):
        center = np.array(box.get_center())[:2] # [x,y]
        center = center[[1,0]] # [y,x]
        
        points = np.array(box.get_box_points())[:,:2] # [x,y]
        points = points[corners_indice].squeeze() # 
        points = points[:,[1,0]] # [y,x]
        
        #
        center = scale * (center - offset)
        points = scale * (points - offset)
        # print('center', center)
        # cv2.circle(bev, (int(center[0]), int(center[1])), 10, 0, -1)
        # cv2.polylines(bev, [points.astype(np.int32)], isClosed=True, color=0, thickness=1)
        cv2.rectangle(bev, (int(points[0,0]), int(points[0,1])), (int(points[-1,0]), int(points[-1,1])), 255, -1)
        cv2.rectangle(bev, (int(points[1,0]), int(points[1,1])), (int(points[2,0]), int(points[2,1])), 255, -1)
        # cv2.line(bev, (int(points[0,0]), int(points[0,1])), (int(points[1,0]), int(points[1,1])), 0, 1)

    #
    for box in floor_bboxes:
        center = np.array(box.get_center())[:2] # [x,y]
        center = center[[1,0]]
        
        points = np.array(box.get_box_points())[:,:2] # [x,y]
        points.squeeze()
        points = points[:,[1,0]] # [y,x]
        
        center = scale * (center - offset)
        points = scale * (points - offset)
        
        cv2.rectangle(floor_bev, (int(points[0,0]), int(points[0,1])), (int(points[-1,0]), int(points[-1,1])), 255, -1)
        cv2.rectangle(floor_bev, (int(points[1,0]), int(points[1,1])), (int(points[2,0]), int(points[2,1])), 255, -1)
        
    print('draw {} floor planes'.format(len(floor_bboxes)))
    # print(np.unique(floor_bev))
    # valid_mask = floor_bev==255
    # bev[~valid_mask] = 255
    
    #
    # out = np.hstack((bev, floor_bev))
    
    print('save to', out_folder)        
    cv2.imwrite(os.path.join(out_folder,'bev.png'), bev)
    cv2.imwrite(os.path.join(out_folder,'floor.png'), floor_bev)
    
    return bev, floor_bev

def watershed(img):
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    room_indices = np.unique(markers)
    viz_markers = np.zeros_like(sure_fg).astype(np.uint8)
    
    
    for room_id in room_indices:
        print(room_id)
        if room_id>1:
            mask = markers==room_id
            viz_markers[mask]=30 * room_id
            # break
    viz_markers = cv2.applyColorMap(viz_markers, cv2.COLORMAP_JET)
    # viz_markers = cv2.applyColorMap((20*viz_markers).astype(np.uint8), cv2.COLORMAP_JET)
    
    # markers = cv2.watershed(img,markers)
    # img[markers == -1] = [255,0,0]
    
    # out = img
    # out = np.hstack((img,dist_transform, sure_fg))
    out = np.hstack((cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR), 
                     viz_markers))
    
    cv2.imshow('watershed', out)
    cv2.waitKey(0)
    
    return out

def segment_rooms(bev, floor_bev):
    import skfmm
    kernel_size = 10
    max_dist = 5.0
    
    # reduce image size
    bev = cv2.resize(bev, (960, 720))
    floor_bev = cv2.resize(floor_bev, (960, 720))
    
    if bev.ndim == 3:
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    if floor_bev.ndim == 3:
        floor_bev = cv2.cvtColor(floor_bev, cv2.COLOR_BGR2GRAY)
    floor_bev = cv2.erode(floor_bev, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)  # shrink floor area
    # floor_bev = cv2.dilate(floor_bev, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)
    floor_mask = floor_bev==255  
    
    # dilate bev
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    dilate_bev = cv2.dilate(bev, kernel, iterations=1)
    
    # erosion bev
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    erode_bev = cv2.erode(bev, kernel, iterations=1)
    
    # wall edges
    edges = cv2.Canny(dilate_bev, 50, 150, apertureSize=3)
    
    #
    processed_bev = dilate_bev.copy()
    processed_bev[~floor_mask] = 255
    processed_bev = reverse_img(processed_bev)
    
    # Floor by distances
    # dist_transform = cv2.distanceTransform(processed_bev,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,70,255,cv2.THRESH_BINARY_INV)
    # processed_bev[sure_fg==0]= 0 
    # out = np.hstack((dilate_bev,floor_bev))
    
    # cv2.imshow('bev', processed_bev)
    # cv2.waitKey(0)
    # return True

    watershed(processed_bev)
    
    return True
    # sdf 
    wall = dilate_bev==255 # True of False
    phi = -1 * np.ones_like(wall).astype(np.float32)
    phi[wall] = 1.0
    sdf = skfmm.distance(phi, dx=0.01)
    sdf = np.abs(sdf)
    
    print('sdf range: ',sdf.min(),sdf.max())

    sdf_offset = sdf - sdf.min()
    sdf_label = (100*sdf_offset).astype(np.uint8)

    outer_region = sdf>1.0
    sdf_label[wall] = 255
    sdf_label[outer_region] = 255
    sdf_label[~floor_mask] = 150
    print('sdf label range', sdf_label.min(), sdf_label.max())
    print(np.unique(sdf_label))
    
    # sdf_viz = sdf_label.copy()
    sdf_viz = cv2.applyColorMap(sdf_label, cv2.COLORMAP_JET) 

    # floodfill to find regions
    h, w = bev.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[1:-1, 1:-1] = (sdf_label==255).astype(np.uint8) # invalid regions are set to 1

    regions = np.zeros((h,w), np.uint8)
    # u,v = 600,400
    # retval, image, mask, rect = cv2.floodFill(sdf_label, mask, (u,v), 255, 0, 10)
    # mask = 255 * mask[1:-1, 1:-1]
    # regions = image
    # cv2.circle(regions, (u,v), 5, 255, -1)
    # cv2.rectangle(regions,(rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, 2)

    # print(retval,rect)
    
    # for v in range(h):
    #     for u in range(w):
    #         if sdf_label[v,u] > 20 and sdf_label[v,u]<120 and mask[v,u]<1:
    #         # if mask[v,u] <1 or floor_mask[v,u]==1:

    #             retval, image, mask, rect = cv2.floodFill(sdf_label, mask, (u,v), 255)
    #             if retval>100 and rect[2]>50 and rect[3]>50:
    #                 cv2.rectangle(regions, 
    #                             (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 
    #                             255, 2)
    #                 print(retval,rect)
    #             cv2.circle(regions, (u,v), 5, 255, -1)
        
    
    # visualize
    # out = np.hstack((bev,dilate_bev))
    out = np.hstack((dilate_bev, processed_bev))
    # out = np.hstack((sdf_viz, cv2.cvtColor(regions, cv2.COLOR_GRAY2BGR)))

    cv2.imshow('bev', out)
    cv2.waitKey(0)
    
        

if __name__ == '__main__':
    map_dir = '/media/lch/SeagateExp/bag/sgslam/val_swap/gfloor/global_rgb.ply'
    # map_dir = '/media/lch/SeagateExp/bag/sgslam/scans/uc0111_00a/mesh_o3d.ply'
    map_folder = os.path.dirname(map_dir)
    max_normals = 0.1
    
    #    
    # global_pcd = o3d.io.read_point_cloud(map_dir)
    # global_pcd.estimate_normals()

    # wall_bboxes, floor_bboxes = detect_planer_patches(global_pcd)
    # bev, floor_bev = bev_from_planes(wall_bboxes, floor_bboxes, map_folder)
    bev = cv2.imread(os.path.join(map_folder, 'bev.png'),-1)
    floor_bev = cv2.imread(os.path.join(map_folder, 'floor.png'),-1)
    
    segment_rooms(bev, floor_bev)
