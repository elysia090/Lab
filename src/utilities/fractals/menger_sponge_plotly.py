import numpy as np
import plotly.graph_objects as go
from itertools import product

def generate_gray_code(n):
    """
    Generate a cyclic Gray code of length n using a recursive approach.
    """
    if n == 0:
        return ['']
    first_half = generate_gray_code(n - 1)
    second_half = first_half[::-1]
    return ['0' + code for code in first_half] + ['1' + code for code in second_half]

def construct_menger_sponge(iterations):
    """
    Construct a 3D Menger sponge up to the specified number of iterations.
    Returns a list of occupied cube coordinates.
    Each coordinate is a tuple (x, y, z) representing the position of the cube.
    """
    cubes = [(0, 0, 0)]
    for it in range(iterations):
        new_cubes = []
        for (x, y, z) in cubes:
            # Subdivide the current cube into 27 subcubes
            for dx, dy, dz in product([0, 1, 2], repeat=3):
                # Determine if the subcube should be removed
                # Remove the center cube and the centers of each face
                if ((dx == 1 and dy == 1) or 
                    (dx == 1 and dz == 1) or 
                    (dy == 1 and dz == 1)):
                    continue
                new_cubes.append((x * 3 + dx, y * 3 + dy, z * 3 + dz))
        cubes = new_cubes
    return cubes

def generate_knot_coords(num_points=1000, scale=1.0):
    """
    Generate 3D coordinates of a trefoil knot using its parametric equations.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = (2 + np.cos(3 * t)) * np.cos(2 * t)
    y = (2 + np.cos(3 * t)) * np.sin(2 * t)
    z = np.sin(3 * t)
    
    # Normalize to fit within [0, 1]^3
    coords = np.vstack((x, y, z)).T
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    normalized_coords = (coords - min_coords) / (max_coords - min_coords)
    
    return normalized_coords.tolist()

def embed_knot_into_sponge(knot_coords, sponge_coords, sponge_size=3**3):
    """
    Embed the knot coordinates into the Menger sponge coordinates.
    Maps each knot point to the nearest sponge cube.
    """
    embedded_knot = []
    for point in knot_coords:
        # Scale point to sponge grid
        scaled_point = [int(np.floor(coord * sponge_size)) for coord in point]
        # Ensure the point is within the sponge bounds
        scaled_point = [min(coord, sponge_size - 1) for coord in scaled_point]
        embedded_knot.append(tuple(scaled_point))
    return embedded_knot

def tropical_embedding(cubes, gray_codes):
    """
    Apply tropical embedding to the list of cube coordinates using Gray codes.
    Returns a list of embedded coordinates in tropical space.
    """
    embedded = []
    for cube, gray_code in zip(cubes, gray_codes):
        # Convert cube coordinates to floats scaled by 1/(3^iterations)
        # Assuming each iteration scales by 1/3
        x = cube[0] * (1/3)
        y = cube[1] * (1/3)
        z = cube[2] * (1/3)
        
        # Apply Gray code mapping: adding binary Gray code bits as tropical shifts
        # This is a simplistic embedding; more sophisticated mappings can be used
        x += gray_code[0]
        y += gray_code[1]
        z += gray_code[2]
        
        embedded.append((x, y, z))
    return embedded

def plot_menger_sponge_with_knot(tropical_sponge_coords, tropical_knot_coords):
    """
    Plot the Menger sponge and the embedded knot in tropical space using Plotly for interactive visualization.
    """
    # Extract sponge coordinates
    sponge_x = [coord[0] for coord in tropical_sponge_coords]
    sponge_y = [coord[1] for coord in tropical_sponge_coords]
    sponge_z = [coord[2] for coord in tropical_sponge_coords]
    
    # Extract knot coordinates
    knot_x = [coord[0] for coord in tropical_knot_coords]
    knot_y = [coord[1] for coord in tropical_knot_coords]
    knot_z = [coord[2] for coord in tropical_knot_coords]
    
    # Create scatter plot for sponge
    sponge_scatter = go.Scatter3d(
        x=sponge_x, 
        y=sponge_y, 
        z=sponge_z,
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.01
        ),
        name='Menger Sponge'
    )
    
    # Create scatter plot for knot
    knot_scatter = go.Scatter3d(
        x=knot_x, 
        y=knot_y, 
        z=knot_z,
        mode='lines',
        line=dict(
            color='red',
            width=4
        ),
        name='Knot'
    )
    
    # Define layout
    layout = go.Layout(
        title='Tropical Embedded Menger Sponge with Knot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Create figure and display
    fig = go.Figure(data=[sponge_scatter, knot_scatter], layout=layout)
    fig.show()

def main():
    # Parameters
    gray_code_length = 3  # For 3D
    iterations = 4       # Number of recursion steps for Menger sponge
    knot_points = 1000    # Number of points in the knot
    
    # Generate cyclic Gray code
    gray_code_sequence = generate_gray_code(gray_code_length)
    total_gray_codes = len(gray_code_sequence)
    
    # Construct Menger sponge
    sponge_cubes = construct_menger_sponge(iterations)
    total_cubes = len(sponge_cubes)
    print(f"Number of cubes after {iterations} iterations: {total_cubes}")
    
    # Generate Gray codes for each cube
    # If there are more cubes than Gray codes, repeat the Gray codes cyclically
    gray_codes = []
    for i in range(len(sponge_cubes)):
        code = gray_code_sequence[i % total_gray_codes]
        # Convert binary string to tuple of integers
        gray_codes.append(tuple(int(bit) for bit in code))
    
    # Apply tropical embedding to sponge cubes
    tropical_sponge_coords = tropical_embedding(sponge_cubes, gray_codes)
    
    # Generate knot coordinates (e.g., trefoil knot)
    knot_coords = generate_knot_coords(num_points=knot_points, scale=1.0)
    
    # Embed knot into sponge
    embedded_knot = embed_knot_into_sponge(knot_coords, sponge_cubes, sponge_size=3**iterations)
    
    # Apply tropical embedding to knot
    # Ensure that the number of Gray codes matches the number of sponge cubes
    # For simplicity, reuse the existing gray_codes
    # Create a mapping from embedded knot to tropical coordinates
    tropical_knot_coords = []
    for point in embedded_knot:
        # Find the index of the cube in sponge_cubes
        try:
            index = sponge_cubes.index(point)
            tropical_coord = tropical_sponge_coords[index]
            tropical_knot_coords.append(tropical_coord)
        except ValueError:
            # If the point is not found, skip it
            continue
    
    # Plot the embedded Menger sponge and knot
    plot_menger_sponge_with_knot(tropical_sponge_coords, tropical_knot_coords)

if __name__ == "__main__":
    main()
