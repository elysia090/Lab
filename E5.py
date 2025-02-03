import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import plotly.graph_objects as go

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

def plot_menger_sponge(tropical_coords):
    """
    Plot the Menger sponge in tropical space using Plotly for interactive visualization.
    """
    # Extract x, y, z coordinates
    xs = [coord[0] for coord in tropical_coords]
    ys = [coord[1] for coord in tropical_coords]
    zs = [coord[2] for coord in tropical_coords]
    
    # Create a scatter plot
    scatter = go.Scatter3d(
        x=xs, 
        y=ys, 
        z=zs,
        mode='markers',
        marker=dict(
            size=2,
            color=zs,  # Color by z-coordinate for depth perception
            colorscale='Viridis',
            opacity=0.8
        )
    )
    
    # Define layout
    layout = go.Layout(
        title='Tropical Embedded 3D Menger Sponge',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Create figure and display
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

def main():
    # Parameters
    gray_code_length = 3  # For 3D
    iterations = 3        # Number of recursion steps for Menger sponge
    
    # Generate cyclic Gray code
    gray_code_sequence = generate_gray_code(gray_code_length)
    total_gray_codes = len(gray_code_sequence)
    
    # Construct Menger sponge
    cubes = construct_menger_sponge(iterations)
    total_cubes = len(cubes)
    print(f"Number of cubes after {iterations} iterations: {total_cubes}")
    
    # Generate Gray codes for each cube
    # If there are more cubes than Gray codes, repeat the Gray codes cyclically
    gray_codes = []
    for i in range(len(cubes)):
        code = gray_code_sequence[i % total_gray_codes]
        # Convert binary string to tuple of integers
        gray_codes.append(tuple(int(bit) for bit in code))
    
    # Apply tropical embedding
    tropical_coords = tropical_embedding(cubes, gray_codes)
    
    # Plot the embedded Menger sponge
    plot_menger_sponge(tropical_coords)

if __name__ == "__main__":
    main()
