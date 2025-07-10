import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def point_in_rectangle(point, rect_bounds):
    """Check if point is inside rectangle defined by [xmin, xmax, ymin, ymax]"""
    x, y = point
    xmin, xmax, ymin, ymax = rect_bounds
    return xmin <= x <= xmax and ymin <= y <= ymax

def triangle_centroid(triangle_points):
    """Calculate centroid of triangle"""
    return np.mean(triangle_points, axis=0)

def triangle_area(triangle_points):
    """Calculate area of triangle using cross product"""
    p1, p2, p3 = triangle_points
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def main():
    # Define boundary points
    # Outer square boundary: (±1, ±1)
    outer_boundary = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1]
    ])
    
    # Inner exclusion zone: (±0.5, ±0.5) 
    inner_boundary = np.array([
        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
    ])
    
    # Combine all boundary points
    all_points = np.vstack([outer_boundary, inner_boundary])
    
    # Add some additional points to ensure good triangulation
    # (Optional: add points along edges or interior for better mesh quality)
    additional_points = np.array([
        [0, -1], [-1, 0], [0, 1], [1, 0],  # midpoints of outer edges
        [-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]  # midpoints of inner edges
    ])
    
    all_points = np.vstack([all_points, additional_points])
    
    print(f"Total points: {len(all_points)}")
    print("Points:")
    for i, point in enumerate(all_points):
        print(f"  {i}: ({point[0]:.1f}, {point[1]:.1f})")
    
    # Create Delaunay triangulation
    tri = Delaunay(all_points)
    
    print(f"\nTotal triangles before exclusion: {len(tri.simplices)}")
    
    # Define exclusion zone bounds
    exclusion_bounds = [-0.5, 0.5, -0.5, 0.5]  # [xmin, xmax, ymin, ymax]
    
    # Identify triangles to keep (not in exclusion zone)
    valid_triangles = []
    excluded_triangles = []
    total_valid_area = 0
    total_excluded_area = 0
    
    for i, simplex in enumerate(tri.simplices):
        # Get triangle vertices
        triangle_points = all_points[simplex]
        centroid = triangle_centroid(triangle_points)
        area = triangle_area(triangle_points)
        
        # Check if triangle centroid is in exclusion zone
        if point_in_rectangle(centroid, exclusion_bounds):
            excluded_triangles.append(i)
            total_excluded_area += area
            print(f"Triangle {i}: EXCLUDED (centroid at {centroid[0]:.2f}, {centroid[1]:.2f})")
        else:
            valid_triangles.append(i)
            total_valid_area += area
            print(f"Triangle {i}: VALID (centroid at {centroid[0]:.2f}, {centroid[1]:.2f})")
    
    print(f"\nValid triangles: {len(valid_triangles)}")
    print(f"Excluded triangles: {len(excluded_triangles)}")
    print(f"Valid area: {total_valid_area:.3f}")
    print(f"Excluded area: {total_excluded_area:.3f}")
    print(f"Total area: {total_valid_area + total_excluded_area:.3f}")
    
    # Expected areas for verification
    total_square_area = 4.0  # 2x2 square
    exclusion_square_area = 1.0  # 1x1 square
    expected_valid_area = total_square_area - exclusion_square_area
    print(f"Expected valid area: {expected_valid_area:.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: All triangles with exclusion zone highlighted
    ax1.triplot(all_points[:, 0], all_points[:, 1], tri.simplices, 'k-', alpha=0.3)
    ax1.plot(all_points[:, 0], all_points[:, 1], 'ro', markersize=4)
    
    # Highlight exclusion zone
    exclusion_rect = Rectangle((-0.5, -0.5), 1.0, 1.0, 
                              facecolor='red', alpha=0.3, edgecolor='red')
    ax1.add_patch(exclusion_rect)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('All Triangles + Exclusion Zone')
    
    # Plot 2: Only valid triangles (reachable area)
    for i in valid_triangles:
        triangle_points = all_points[tri.simplices[i]]
        triangle = patches.Polygon(triangle_points, closed=True, 
                                 facecolor='lightblue', edgecolor='blue', alpha=0.7)
        ax2.add_patch(triangle)
    
    ax2.plot(all_points[:, 0], all_points[:, 1], 'ro', markersize=4)
    
    # Show exclusion zone boundary for reference
    exclusion_rect2 = Rectangle((-0.5, -0.5), 1.0, 1.0, 
                               facecolor='none', edgecolor='red', linewidth=2)
    ax2.add_patch(exclusion_rect2)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Valid Triangles Only (Unobserved Area)')
    
    plt.tight_layout()
    plt.savefig("figures/triangulations.png")
    plt.show()
    
    return valid_triangles, excluded_triangles, total_valid_area

valid_triangles, excluded_triangles, remaining_area = main()
