import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def load_processed_data(json_file_path):
    with open(json_file_path) as f:
        info = json.load(f)
    
    room_polygons = [np.array(poly) for poly in info['room_polygons']]
    room_types = info['rms_type']
    
    return room_polygons, room_types

def visualize_floorplan(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    num_samples = min(9, len(json_files))
    selected_files = random.sample(json_files, num_samples)
    
    print(f"Visualizing {num_samples} randomly selected files from {folder_path}")
    
    room_names = {
        1: 'Living room', 
        2: 'Kitchen', 
        3: 'Bedroom', 
        4: 'Bathroom', 
        5: 'Balcony', 
        6: 'Dining room', 
        7: 'Study room',
        8: 'Storage'
    }
    
    room_type_colors = {
        1: '#FFB6C1',  # Light pink - Living room
        2: '#98FB98',  # Pale green - Kitchen
        3: '#87CEEB',  # Sky blue - Bedroom
        4: '#DDA0DD',  # Plum - Bathroom
        5: '#F0E68C',  # Khaki - Balcony
        6: '#E6E6FA',  # Lavender - Dining room
        7: '#FFE4B5',  # Moccasin - Study room
        8: '#D3D3D3',  # Light gray - Storage
        9: '#000000'   # Saddle brown - boundary
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, selected_file_name in enumerate(selected_files):
        json_file_path = os.path.join(folder_path, selected_file_name)
        
        room_polygons, room_types = load_processed_data(json_file_path)
        
        ax = axes[idx]
        ax.set_title(f'{os.path.splitext(selected_file_name)[0]}', fontsize=12, weight='bold')
        
        if room_polygons:
            all_coords = np.vstack(room_polygons)
            min_coord = np.min(all_coords) - 10
            max_coord = np.max(all_coords) + 10
        else:
            min_coord, max_coord = 0, 256
        
        boundary_count = 0
        room_count = 0
        
        # Draw polygons
        for polygon, room_type in zip(room_polygons, room_types):
            room_color = room_type_colors.get(room_type, '#F5F5F5')
            
            if room_type == 9:  # Boundary - draw edges only
                boundary_count += 1
                closed_poly = np.vstack([polygon, polygon[0]])
                ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
                       color=room_color, linewidth=3, zorder=10, alpha=0.8)
            else:  # Regular rooms (1-8) - draw filled with labels
                room_count += 1
                # Draw filled polygon
                ax.fill(polygon[:, 0], polygon[:, 1], 
                       color=room_color, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add room label at center
                centroid = np.mean(polygon, axis=0)
                room_name = room_names.get(room_type, f'Type {room_type}')
                ax.text(centroid[0], centroid[1], room_name, 
                       ha='center', va='center', fontsize=9, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                               alpha=0.9, edgecolor='black', linewidth=0.5))
                           
        ax.set_xlim(min_coord, max_coord)
        ax.set_ylim(min_coord, max_coord)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        # Add info text
        info_text = f"Rooms: {room_count}"
        if boundary_count > 0:
            info_text += f", Boundary: {boundary_count}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
    
    # Hide unused subplots
    for idx in range(num_samples, 9):
        axes[idx].axis('off')
    
    plt.suptitle('Floor Plans with Boundaries', fontsize=16, weight='bold', y=0.95)
    plt.tight_layout()
    plt.show()
    
    return selected_files

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        print(f"Visualizing floor plans from: {folder_path}")
        visualize_floorplan(folder_path)
    else:
        print("Usage: python visualization.py <folder_path>")
        print("Or import the function to use it in your code.")