import os, csv
import numpy as np
import cv2

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']

def get_raw2scannetv2_label_map():
    lines = [line.rstrip() for line in open('/data2/ScanNet/scannetv2-labels.combined.tsv')]
    lines_0 = lines[0].split('\t')
    # print(lines_0)
    # print(len(lines))
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        # if (elements[1] != elements[2]):
            # print('{}: {} {}'.format(i, elements[1], elements[2]))
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

g_raw2scannetv2 = get_raw2scannetv2_label_map()

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# input: scene_types.txt or scene_types_all.txt
def read_scene_types_mapping(filename, remove_spaces=True):
    assert os.path.isfile(filename)
    mapping = dict()
    lines = open(filename).read().splitlines()
    lines = [line.split('\t') for line in lines]
    if remove_spaces:
        mapping = { x[1].strip():int(x[0]) for x in lines }
    else:
        mapping = { x[1]:int(x[0]) for x in lines }        
    return mapping

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.items():
        mapped[image==k] = v
    return mapped.astype(np.uint8)

def visualize_label_image(image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    # imageio.imwrite(filename, vis_image)
    return vis_image

# color by different instances (mod length of color palette)
def visualize_instance_image(image):
    '''
        image: HxW, dtype=np.uint8, in NYU40 label format    
    '''
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    instances = np.unique(image)
    for idx, inst in enumerate(instances):
        vis_image[image==inst] = color_palette[inst%len(color_palette)]
    return vis_image
    # cv2.imwrite(output_dir, vis_image)

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall-0
       (152, 223, 138),		# floor-1
       (31, 119, 180), 		# cabinet-2
       (255, 187, 120),		# bed-3
       (188, 189, 34), 		# chair-4
       (140, 86, 75),  		# sofa-5
       (255, 152, 150),		# table-6
       (214, 39, 40),  		# door-7
       (197, 176, 213),		# window-8
       (148, 103, 189),		# bookshelf-9
       (196, 156, 148),		# picture-10
       (23, 190, 207), 		# counter-11
       (178, 76, 76),  
       (247, 182, 210),		# desk-12
       (66, 188, 102), 
       (219, 219, 141),		# curtain-13
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator-14
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain-15
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet-16
       (112, 128, 144),		# sink-17
       (96, 207, 209), 
       (227, 119, 194),		# bathtub-18
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn-19
       (100, 85, 144)
    ]
