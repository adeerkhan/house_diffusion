import json
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def make_sequence(edges):
    """
    Exact copy of make_sequence from rplanhg_datasets.py
    """
    polys = []
    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False
    
    while len(e_visited) < len(edges):
        if find_next == False:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = not find_next 
        else:
            # look for next edge
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if (v_curr == tuple(e[:2])):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif (v_curr == tuple(e[2:])):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break

        # extract next sequence
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(e[:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    
    polys.append(seq_tracker)
    return polys

def build_graph_exact(rms_type, fp_eds, eds_to_rms, out_size=64):
    """
    Exact copy of build_graph from rplanhg_datasets.py
    """
    # create edges
    triples = []
    nodes = rms_type 
    # encode connections
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                if is_adjacent:
                    triples.append([k, 1, l])
                else:
                    triples.append([k, -1, l])
    
    # get rooms masks
    eds_to_rms_tmp = []
    for l in range(len(eds_to_rms)):                  
        eds_to_rms_tmp.append([eds_to_rms[l][0]])
    
    rms_masks = []
    im_size = 256
    fp_mk = np.zeros((out_size, out_size))
    
    for k in range(len(nodes)):
        # add rooms and doors
        eds = []
        for l, e_map in enumerate(eds_to_rms_tmp):
            if (k in e_map):
                eds.append(l)
        
        # draw rooms
        rm_im = Image.new('L', (im_size, im_size))
        dr = ImageDraw.Draw(rm_im)
        
        for eds_poly in [eds]:
            poly = make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
            poly = [(im_size*x, im_size*y) for x, y in poly]
            if len(poly) >= 2:
                dr.polygon(poly, fill='white')
            else:
                print("Empty room")
                exit(0)
        
        rm_im = rm_im.resize((out_size, out_size))
        rm_arr = np.array(rm_im)
        inds = np.where(rm_arr>0)
        rm_arr[inds] = 1.0
        rms_masks.append(rm_arr)
        
        if rms_type[k] != 15 and rms_type[k] != 17:
            fp_mk[inds] = k+1
    
    # trick to remove overlap
    for k in range(len(nodes)):
        if rms_type[k] != 15 and rms_type[k] != 17:
            rm_arr = np.zeros((out_size, out_size))
            inds = np.where(fp_mk==k+1)
            rm_arr[inds] = 1.0
            rms_masks[k] = rm_arr
    
    # convert to array
    nodes = np.array(nodes)
    triples = np.array(triples)
    rms_masks = np.array(rms_masks)
    return nodes, triples, rms_masks

def remove_walls_exact(json_file_path, output_size=256):
    """
    Wall removal using exact same preprocessing as rplanhg_datasets.py
    """
    # Load JSON data
    with open(json_file_path) as f:
        info = json.load(f)
    
    rms_type = info['room_type']
    rms_bbs = np.array(info['boxes'])
    fp_eds = np.array(info['edges'])
    eds_to_rms = info['ed_rm']
    
    # Exact preprocessing from rplanhg_datasets.py
    rms_bbs = np.array(rms_bbs)/256.0
    fp_eds = np.array(fp_eds)/256.0 
    fp_eds = fp_eds[:, :4]
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl+br)/2.0 - 0.5
    rms_bbs[:, :2] -= shift 
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift 
    
    # Use exact build_graph
    nodes, triples, rms_masks = build_graph_exact(rms_type, fp_eds, eds_to_rms, out_size=64)
    
    print(f"Generated {len(rms_masks)} room masks")
    for i, mask in enumerate(rms_masks):
        print(f"Room {i} (type {rms_type[i]}): mask sum = {np.sum(mask)}")
    
    # Extract polygons from masks exactly like in rplanhg_datasets.py
    room_polygons = []
    room_types = []
    
    for i, (room_mask, room_type) in enumerate(zip(rms_masks, rms_type)):
        # Skip doors and windows
        if room_type == 15 or room_type == 17:
            continue
            
        # Check if mask has any content
        if np.sum(room_mask) == 0:
            print(f"Skipping empty mask for room {i}")
            continue
            
        room_mask = room_mask.astype(np.uint8)
        
        # Resize to 256x256 exactly like reference
        room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
        
        # Find contours exactly like reference
        contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contour = contours[0]
            polygon = contour[:, 0, :]
            
            # Apply exact same transformation as reference:
            # reshape, normalize, center, scale
            polygon = np.reshape(polygon, [len(polygon), 2])/256. - 0.5
            polygon = polygon * 2
            
            room_polygons.append(polygon)
            room_types.append(room_type)
            print(f"Added room {i} with {len(polygon)} points")
    
    return room_polygons, room_types, rms_masks

def visualize_exact(json_file_path):
    """Visualize the exact reference implementation results"""
    # Load original data
    with open(json_file_path) as f:
        info = json.load(f)
    
    room_polygons, room_types, room_masks = remove_walls_exact(json_file_path)
    
    # Room type names
    room_names = {
        1: 'Living Room', 2: 'Bedroom', 3: 'Bathroom', 4: 'Kitchen', 
        5: 'Balcony', 15: 'Window', 17: 'Door'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original with walls
    ax = axes[0]
    ax.set_title('Original Floor Plan (with walls)')
    
    all_boxes = np.array(info['boxes'])
    min_coord = np.min(all_boxes)
    max_coord = np.max(all_boxes)
    
    # Draw rooms as bounding boxes
    for i, (box, room_type) in enumerate(zip(info['boxes'], info['room_type'])):
        if room_type not in [15, 17]:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
            
            room_name = room_names.get(room_type, f'Type {room_type}')
            ax.text(x1 + width/2, y1 + height/2, room_name, 
                   ha='center', va='center', fontsize=8)
    
    # Draw edges
    for edge in info['edges']:
        x1, y1, x2, y2 = edge[:4]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1, alpha=0.7)
    
    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Processed without walls
    ax = axes[1]
    ax.set_title('Processed Floor Plan (walls removed)')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(room_polygons)))
    
    for i, (polygon, room_type) in enumerate(zip(room_polygons, room_types)):
        # Convert from [-1,1] back to original coordinates for plotting
        # Reverse the transformation: polygon * 2 -> polygon / 2
        # polygon - 0.5 -> polygon + 0.5  
        # polygon / 256 -> polygon * 256
        plot_polygon = (polygon / 2 + 0.5) * 256
        
        ax.fill(plot_polygon[:, 0], plot_polygon[:, 1], 
               color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
        
        centroid = np.mean(plot_polygon, axis=0)
        room_name = room_names.get(room_type, f'Type {room_type}')
        ax.text(centroid[0], centroid[1], room_name, 
               ha='center', va='center', fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('wall_removal_exact.png', dpi=150, bbox_inches='tight')
    print("Exact visualization saved as 'wall_removal_exact.png'")
    
    return room_polygons, room_types

if __name__ == "__main__":
    json_file = "dataset/0.json"
    print("Testing exact reference implementation...")
    room_polygons, room_types = visualize_exact(json_file)
    print(f"Found {len(room_polygons)} rooms with types: {room_types}")
