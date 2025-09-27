import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# Use CuPy for GPU acceleration if available, otherwise use NumPy
try:
    import cupy as cp
    has_cupy = True
    print("CuPy is available. Using GPU acceleration.")
except ImportError:
    has_cupy = False
    print("CuPy is not available. Using NumPy on CPU.")

# xp: Alias for either CuPy or NumPy
xp = cp if has_cupy else np

def is_in_menger_sponge(coord, iterations):
    """
    Determine if a point is in the Menger sponge using the Teichmüller lift formulation.
    
    Args:
        coord (tuple): (x, y, z) coordinates in [0,1]^3
        iterations (int): Number of iterations (depth) of the Menger sponge
        
    Returns:
        bool: True if the point is in the Menger sponge, False otherwise
    """
    x, y, z = coord
    for i in range(iterations):
        # Get the current ternary digit
        x_digit = int((x * (3**i)) % 3)
        y_digit = int((y * (3**i)) % 3)
        z_digit = int((z * (3**i)) % 3)
        
        # Menger sponge definition: remove central subcubes
        if x_digit == 1 and y_digit == 1 and z_digit == 1:
            return False
    return True

def create_menger_voxels(iterations, resolution):
    """
    Generate voxel data for the Menger sponge.
    
    Args:
        iterations (int): Number of iterations (depth) of the Menger sponge
        resolution (int): Grid resolution for the voxel representation
        
    Returns:
        numpy.ndarray: Boolean 3D array representing the Menger sponge voxels
    """
    # Create a grid based on the resolution
    grid = xp.linspace(0, 1, resolution)
    x, y, z = xp.meshgrid(grid, grid, grid)
    
    # Combine coordinates into an n×3 array
    points = xp.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Initialize voxel array
    voxels = xp.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Process based on whether CuPy or NumPy is used
    if has_cupy:
        # GPU kernel function for parallel processing
        is_in_menger_kernel = cp.ElementwiseKernel(
            'float32 x, float32 y, float32 z, int32 iterations',
            'bool result',
            '''
            result = true;
            for (int i = 0; i < iterations; i++) {
                int x_digit = int(fmodf(x * powf(3, i), 3));
                int y_digit = int(fmodf(y * powf(3, i), 3));
                int z_digit = int(fmodf(z * powf(3, i), 3));
                if (x_digit == 1 && y_digit == 1 && z_digit == 1) {
                    result = false;
                    break;
                }
            }
            ''',
            'is_in_menger_kernel'
        )
        
        # Parallel processing on GPU
        results = is_in_menger_kernel(points[:, 0].astype(np.float32), 
                                      points[:, 1].astype(np.float32), 
                                      points[:, 2].astype(np.float32), 
                                      np.int32(iterations))
        voxels = results.reshape(resolution, resolution, resolution)
        
        # Transfer results back to CPU
        voxels = cp.asnumpy(voxels)
    else:
        # CPU processing (using NumPy)
        for i, (x, y, z) in enumerate(points):
            if is_in_menger_sponge((x, y, z), iterations):
                idx = np.unravel_index(i, (resolution, resolution, resolution))
                voxels[idx] = True
    
    return voxels

def plot_menger_sponge(voxels):
    """
    Display the Menger sponge in 3D using matplotlib with detailed faces
    
    Args:
        voxels (numpy.ndarray): Boolean 3D array representing the Menger sponge voxels
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Represent voxels as polygons
    vertices = []
    faces = []
    
    # Set color (within 0-1 range)
    base_color = np.array([0.5, 0.5, 0.8])
    
    # Convert voxels to polygons
    for i, j, k in zip(*np.where(voxels)):
        # Normalized coordinates
        x, y, z = i/voxels.shape[0], j/voxels.shape[1], k/voxels.shape[2]
        size = 1.0/voxels.shape[0]
        
        # Eight vertices of the cube
        v = np.array([
            [x, y, z],
            [x+size, y, z],
            [x+size, y+size, z],
            [x, y+size, z],
            [x, y, z+size],
            [x+size, y, z+size],
            [x+size, y+size, z+size],
            [x, y+size, z+size]
        ])
        
        # Six faces of the cube
        f = np.array([
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]],
            [v[0], v[3], v[7], v[4]],
            [v[1], v[2], v[6], v[5]]
        ])
        
        # Add all faces (for simplicity)
        for face in f:
            vertices.append(face)
            faces.append(len(vertices) - 1)
    
    # Create and display polygon collection
    poly = Poly3DCollection([vertices[i] for i in faces], alpha=0.5)
    
    # Set single color (setting different colors for each face may cause errors)
    poly.set_facecolor(base_color)
    poly.set_edgecolor('black')
    ax.add_collection3d(poly)
    
    # Set axis ranges
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Menger Sponge (Teichmüller Lift Formulation)')
    
    plt.tight_layout()
    plt.show()

def plot_menger_simple(voxels):
    """
    Display the Menger sponge in a simplified way (direct voxel rendering)
    to improve performance by reducing the number of faces.
    
    Args:
        voxels (numpy.ndarray): Boolean 3D array representing the Menger sponge voxels
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Directly render voxels
    ax.voxels(voxels, facecolors=[0.5, 0.5, 0.8, 0.3], edgecolor='k')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Menger Sponge (Teichmüller Lift Formulation) - Simple View')
    
    plt.tight_layout()
    plt.show()

def run_visualization(iterations=3, resolution=27, simple_viz=True):
    """
    Run the Menger sponge visualization with specified parameters
    
    Args:
        iterations (int): Number of iterations (depth) of the Menger sponge
        resolution (int): Grid resolution for the voxel representation
        simple_viz (bool): Whether to use simplified visualization
    """
    print(f"Generating Menger sponge: depth={iterations}, resolution={resolution}")
    
    start_time = time.time()
    voxels = create_menger_voxels(iterations, resolution)
    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    print(f"Number of voxels: {np.sum(voxels)}")
    
    # Choose visualization method
    if simple_viz:
        plot_menger_simple(voxels)
    else:
        plot_menger_sponge(voxels)

# Run directly with default parameters: depth=3, resolution=27, simple_viz=True
run_visualization(5, 27, True)

# To change parameters, uncomment and modify the following line
# run_visualization(iterations=2, resolution=9, simple_viz=False)
