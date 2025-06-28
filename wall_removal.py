# This script processes RPLAN floor plans to remove interior walls spaces. 
import json
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
import os
import glob
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.errors import TopologicalError
import pyclipper

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
        
        # Include all room types in floor plan mask
        if rms_type[k] != 15 and rms_type[k] != 17:
            fp_mk[inds] = k+1
    
    # trick to remove overlap - apply to all room types
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

def _edges_orientation(p_start, p_end, eps=1e-6):
    """Return 'h' for horizontal, 'v' for vertical, None otherwise."""
    dx, dy = p_end - p_start
    if abs(dy) < eps and abs(dx) > eps:
        return 'h'
    if abs(dx) < eps and abs(dy) > eps:
        return 'v'
    return None

def _overlap_1d(a_min, a_max, b_min, b_max):
    return max(a_min, b_min) <= min(a_max, b_max)

def align_adjacent_boundaries(room_polygons, tolerance=0.02):
    """Snap adjacent horizontal/vertical room edges together to remove gaps.

    A simpler, axis-aligned algorithm that avoids the previous polygon-collapse
    issue. Two edges are considered adjacent when:
        • They have the same orientation (horizontal or vertical).
        • Their perpendicular distance is < tolerance.
        • Their projections on the orientation axis overlap.

    The shared coordinate (x for vertical edges, y for horizontal) is replaced
    with the average of the two, ensuring both polygons meet perfectly.
    """

    if len(room_polygons) < 2:
        return room_polygons

    # Work on copies
    aligned = [poly.copy() for poly in room_polygons]

    num_rooms = len(aligned)
    for i in range(num_rooms):
        for j in range(i + 1, num_rooms):
            poly1, poly2 = aligned[i], aligned[j]

            for idx1 in range(len(poly1)):
                s1, e1 = poly1[idx1], poly1[(idx1 + 1) % len(poly1)]
                orient1 = _edges_orientation(s1, e1)
                if orient1 is None:
                    continue

                for idx2 in range(len(poly2)):
                    s2, e2 = poly2[idx2], poly2[(idx2 + 1) % len(poly2)]
                    orient2 = _edges_orientation(s2, e2)
                    if orient1 != orient2 or orient2 is None:
                        continue

                    if orient1 == 'h':  # horizontal -> compare y
                        if abs(s1[1] - s2[1]) > tolerance:
                            continue
                        # Check x-interval overlap
                        a_min, a_max = sorted([s1[0], e1[0]])
                        b_min, b_max = sorted([s2[0], e2[0]])
                        if not _overlap_1d(a_min, a_max, b_min, b_max):
                            continue
                        new_y = 0.5 * (s1[1] + s2[1])
                        poly1[idx1][1] = new_y
                        poly1[(idx1 + 1) % len(poly1)][1] = new_y
                        poly2[idx2][1] = new_y
                        poly2[(idx2 + 1) % len(poly2)][1] = new_y

                    else:  # vertical -> compare x
                        if abs(s1[0] - s2[0]) > tolerance:
                            continue
                        a_min, a_max = sorted([s1[1], e1[1]])
                        b_min, b_max = sorted([s2[1], e2[1]])
                        if not _overlap_1d(a_min, a_max, b_min, b_max):
                            continue
                        new_x = 0.5 * (s1[0] + s2[0])
                        poly1[idx1][0] = new_x
                        poly1[(idx1 + 1) % len(poly1)][0] = new_x
                        poly2[idx2][0] = new_x
                        poly2[(idx2 + 1) % len(poly2)][0] = new_x

    # Clean and validate polygons after snapping
    cleaned = []
    for poly in aligned:
        if len(poly) < 3:
            continue
        cleaned_poly = _clean_polygon(poly)
        if cleaned_poly is not None:
            cleaned.append(cleaned_poly)

    return cleaned

def are_edges_adjacent(p1_start, p1_end, p2_start, p2_end, tolerance):
    """
    Check if two edges are adjacent (parallel and close to each other)
    """
    # Calculate edge vectors
    edge1 = p1_end - p1_start
    edge2 = p2_end - p2_start
    
    # Check if edges are roughly parallel
    edge1_len = np.linalg.norm(edge1)
    edge2_len = np.linalg.norm(edge2)
    
    if edge1_len < 1e-6 or edge2_len < 1e-6:
        return False, 0
    
    edge1_norm = edge1 / edge1_len
    edge2_norm = edge2 / edge2_len
    
    # Check parallelism (dot product should be close to ±1)
    dot_product = np.dot(edge1_norm, edge2_norm)
    if np.abs(dot_product) < 0.9:  # Not parallel enough
        return False, 0
    
    # Check if edges are close to each other
    # Distance from p2_start to line p1_start->p1_end
    dist1 = point_to_line_distance(p2_start, p1_start, p1_end)
    dist2 = point_to_line_distance(p2_end, p1_start, p1_end)
    
    return (dist1 < tolerance) and (dist2 < tolerance), dot_product

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate distance from a point to a line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-6:
        return np.linalg.norm(point_vec)
    
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
    projection = line_start + t * line_vec
    
    return np.linalg.norm(point - projection)

def remove_walls(json_file_path, output_size=256):
    """
    Wall removal that excludes interior doors (type 10) and entrances/front doors (type 9) from the output
    """
    # Load JSON data
    with open(json_file_path) as f:
        info = json.load(f)
    rms_type = info['room_type']
    rms_bbs = np.array(info['boxes'])
    fp_eds = np.array(info['edges'])
    eds_to_rms = info['ed_rm']

    # Filter out doors/entrances (room_type == 9 or 10)
    main_room_indices = [i for i, t in enumerate(rms_type) if t not in [9, 10]]
    filtered_rms_type = [rms_type[i] for i in main_room_indices]
    filtered_rms_bbs = rms_bbs[main_room_indices]

    # Update eds_to_rms to only include main rooms
    filtered_eds_to_rms = []
    edge_keep_indices = []
    for idx, e_map in enumerate(eds_to_rms):
        # Only keep if all referenced rooms are main rooms
        if all(r in main_room_indices for r in e_map):
            # Remap room indices to new filtered indices
            new_map = [main_room_indices.index(r) for r in e_map]
            filtered_eds_to_rms.append(new_map)
            edge_keep_indices.append(idx)

    # Filter fp_eds to only those edges associated with main rooms
    filtered_fp_eds = fp_eds[edge_keep_indices]

    # Normalize and centralize as before
    filtered_rms_bbs = np.array(filtered_rms_bbs)/256.0
    filtered_fp_eds = np.array(filtered_fp_eds)/256.0 
    filtered_fp_eds = filtered_fp_eds[:, :4]
    tl = np.min(filtered_rms_bbs[:, :2], 0)
    br = np.max(filtered_rms_bbs[:, 2:], 0)
    shift = (tl+br)/2.0 - 0.5
    filtered_rms_bbs[:, :2] -= shift 
    filtered_rms_bbs[:, 2:] -= shift
    filtered_fp_eds[:, :2] -= shift
    filtered_fp_eds[:, 2:] -= shift 

    # Use exact build_graph
    nodes, triples, rms_masks = build_graph_exact(filtered_rms_type, filtered_fp_eds, filtered_eds_to_rms, out_size=64)
    # Extract polygons from masks - treat all room types uniformly
    room_polygons = []
    room_types = []
    for i, (room_mask, room_type) in enumerate(zip(rms_masks, filtered_rms_type)):
        # Check if mask has any content
        if np.sum(room_mask) == 0:
            continue
        room_mask = room_mask.astype(np.uint8)
        # Resize to 256x256 exactly like reference
        room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
        # Find contours exactly like reference
        contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = contours[0]
            polygon = contour[:, 0, :]
            # Apply exact same transformation as reference
            polygon = np.reshape(polygon, [len(polygon), 2])/256. - 0.5
            polygon = polygon * 2
            room_polygons.append(polygon)
            room_types.append(room_type)
    return room_polygons, room_types, rms_masks

def convert_polygons_to_edges_and_boxes(room_polygons, room_types, door_polygons=None, door_types=None):
    if door_polygons is None:
        door_polygons = []
    if door_types is None:
        door_types = []
        
    all_polygons = room_polygons + door_polygons
    all_types = room_types + door_types
    
    edges = []
    boxes = []
    eds_to_rms = []
    edge_counter = 0
    
    for room_idx, (polygon, room_type) in enumerate(zip(all_polygons, all_types)):
        polygon_coords = (polygon / 2 + 0.5) * 256
        
        min_x = np.min(polygon_coords[:, 0])
        min_y = np.min(polygon_coords[:, 1])
        max_x = np.max(polygon_coords[:, 0])
        max_y = np.max(polygon_coords[:, 1])
        
        boxes.append([float(min_x), float(min_y), float(max_x), float(max_y)])
        
        for i in range(len(polygon_coords)):
            start_point = polygon_coords[i]
            end_point = polygon_coords[(i + 1) % len(polygon_coords)]
            
            edge = [float(start_point[0]), float(start_point[1]), 
                   float(end_point[0]), float(end_point[1]), room_type, 0]
            edges.append(edge)
            
            eds_to_rms.append([room_idx])
            edge_counter += 1
    
    return edges, boxes, eds_to_rms

def process_json_file_to_json(input_json_path, output_json_path, boundary_offset=0.05):
    try:
        room_polygons_list, room_types_list, _ = remove_walls(input_json_path)
        
        # Align polygons to close gaps between adjacent rooms.
        aligned_polygons = align_adjacent_boundaries(room_polygons_list)
        
        # Create boundary polygons using pyclipper offset
        boundary_polygons = _create_boundary(aligned_polygons, room_types_list, offset=boundary_offset)
        
        # Add boundary polygons to the output with room class 9
        all_output_polygons = aligned_polygons.copy()
        all_output_types = room_types_list.copy()
        
        # Add boundary polygons with room type 9
        boundary_room_type = 9  # Boundary room class
        for boundary_poly in boundary_polygons:
            all_output_polygons.append(boundary_poly)
            all_output_types.append(boundary_room_type)
        
        new_edges, new_boxes, new_ed_rm_mapping = convert_polygons_to_edges_and_boxes(
            all_output_polygons, 
            all_output_types, 
            [],
            []
        )
        
        output_polygons_original_scale = []
        for poly in all_output_polygons:
            output_polygons_original_scale.append((poly / 2 + 0.5) * 256)

        output_data = {
            "room_polygons": [p.tolist() for p in output_polygons_original_scale],
            "rms_type": all_output_types,
            "edgs_to_rms": new_ed_rm_mapping
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Processed {input_json_path} -> {output_json_path} (added {len(boundary_polygons)} boundary polygons)")
        return True
        
    except Exception as e:
        print(f"Error processing {input_json_path}: {str(e)}")
        return False

def process_folder_json_files(input_folder, output_folder, boundary_offset=0.05):
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    success_count = 0
    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        output_path = os.path.join(output_folder, f"{filename}")
        
        if process_json_file_to_json(json_file, output_path, boundary_offset):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(json_files)} files")
    
    return success_count

# -----------------------------------------------------------------------------
# Helper to clean polygons and prevent collapsing/self-intersections
# -----------------------------------------------------------------------------

def _clean_polygon(polygon_coords, min_area=1e-4):
    """Return a cleaned numpy array of polygon coords or None if invalid/too small."""
    try:
        poly = _ShapelyPolygon(polygon_coords)
        # Fix self-intersections
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area:
            return None
        cleaned = np.asarray(poly.exterior.coords[:-1])  # drop duplicate last point
        if cleaned.shape[0] < 3:
            return None
        return cleaned
    except (TopologicalError, ValueError):
        return None

def _create_boundary(room_polygons, room_types, offset=0.05):
    """
    Create boundary polygons around all rooms using pyclipper offset.
    
    Args:
        room_polygons: List of room polygons
        room_types: List of room types corresponding to polygons
        offset: Offset distance for boundary creation (default 0.05)
    
    Returns:
        List of boundary polygons
    """
    # Use polygons directly without additional scaling
    polygons_for_union = []
    # Define ONLY the door types to exclude from union
    door_types = {9, 10}  # Interior Door (10) and Main Door/Entrance (9)
    
    # Iterate through all polygons and their types
    for poly, poly_type in zip(room_polygons, room_types):
        if len(poly) < 3:  # Need at least 3 points for a valid polygon
            continue
        # *** This is the crucial check ***
        # If the polygon type is NOT one of the specified doors, include it
        if poly_type not in door_types:  # Exclude Interior Door and Main Door
            polygons_for_union.append([(float(x), float(y)) for x, y in poly])
    
    # If no valid polygons remain for union (unlikely unless only doors exist)
    if not polygons_for_union:
        return []
    
    # --- The rest of the function uses 'polygons_for_union' ---
    # Use a scale factor for pyclipper to work reliably.
    scale_factor = 10000.0  # Higher scale factor for better precision
    scaled_polys = []
    # Use the filtered list for the union operation
    for p in polygons_for_union:
        scaled_polys.append([(int(x * scale_factor), int(y * scale_factor)) for (x, y) in p])
    
    try:
        pc = pyclipper.Pyclipper()
        pc.AddPaths(scaled_polys, pyclipper.PT_SUBJECT, closed=True)
        # The union is calculated ONLY from non-door polygons
        union_result = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        
        if not union_result:
            return []
        
        # Create offset boundary
        offsetter = pyclipper.PyclipperOffset()
        offsetter.AddPaths(union_result, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        offset_result = offsetter.Execute(offset * scale_factor)
        
        boundary_polygons = []
        for out_poly in offset_result:
            coords = [(x / scale_factor, y / scale_factor) for (x, y) in out_poly]
            if len(coords) >= 3:  # Ensure valid polygon
                boundary_polygons.append(np.array(coords))
        
        return boundary_polygons
        
    except Exception as e:
        print(f"Error creating boundary: {str(e)}")
        return []

