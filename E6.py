import numpy as np
import plotly.graph_objects as go

def generate_gray_code(n: int) -> list[str]:
    """
    Generate a cyclic Gray code of length n using a recursive approach.
    
    Args:
        n (int): Length of the Gray code.
        
    Returns:
        list[str]: A list of binary strings representing the Gray code sequence.
    """
    if n == 0:
        return ['']
    first_half = generate_gray_code(n - 1)
    second_half = first_half[::-1]
    return ['0' + code for code in first_half] + ['1' + code for code in second_half]

def construct_sierpinski_carpet(iterations: int) -> list[tuple[int, int]]:
    """
    Construct a 2D Sierpinski carpet up to the specified number of iterations.
    
    Args:
        iterations (int): Number of recursion steps.
        
    Returns:
        list[tuple[int, int]]: List of occupied square coordinates.
    """
    squares = [(0, 0)]
    for it in range(iterations):
        new_squares = []
        for (x, y) in squares:
            # Subdivide the current square into 9 subsquares
            for dx, dy in [(0, 0), (0, 1), (0, 2), 
                           (1, 0), (1, 2), 
                           (2, 0), (2, 1), (2, 2)]:
                # Skip the center square
                if dx == 1 and dy == 1:
                    continue
                new_squares.append((x * 3 + dx, y * 3 + dy))
        squares = new_squares
    return squares

def generate_curve_coords(num_points: int = 1000, curve_type: str = 'lissajous') -> list[tuple[float, float]]:
    """
    Generate 2D coordinates of a specified curve using parametric equations.
    
    Args:
        num_points (int, optional): Number of points to generate. Defaults to 1000.
        curve_type (str, optional): Type of curve. Defaults to 'lissajous'.
        
    Returns:
        list[tuple[float, float]]: List of 2D coordinates representing the curve.
    """
    if curve_type.lower() == 'lissajous':
        t = np.linspace(0, 2 * np.pi, num_points)
        x = np.cos(3 * t)
        y = np.sin(2 * t)
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    
    coords = np.vstack((x, y)).T
    # Normalize to fit within [0, 1]^2
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    normalized_coords = (coords - min_coords) / (max_coords - min_coords)
    return normalized_coords.tolist()

def tropical_embedding_2d(
    carpet_coords: list[tuple[int, int]],
    gray_codes: list[tuple[int, int]]
) -> list[tuple[float, float]]:
    """
    Apply tropical embedding to the list of carpet coordinates using Gray codes.
    
    Args:
        carpet_coords (list[tuple[int, int]]): List of carpet square coordinates.
        gray_codes (list[tuple[int, int]]): Corresponding Gray codes for each square.
        
    Returns:
        list[tuple[float, float]]: List of embedded coordinates in tropical space.
    """
    embedded = []
    for square, gray_code in zip(carpet_coords, gray_codes):
        x = square[0] * (1 / 3) + gray_code[0]
        y = square[1] * (1 / 3) + gray_code[1]
        embedded.append((x, y))
    return embedded

def plot_carpet_with_curve(
    tropical_carpet_coords: list[tuple[float, float]],
    tropical_curve_coords: list[tuple[float, float]],
    carpet_color: str = 'blue',
    curve_color: str = 'red'
) -> None:
    """
    Plot the 2D carpet and the embedded curve in tropical space using Plotly.
    
    Args:
        tropical_carpet_coords (list[tuple[float, float]]): Embedded carpet coordinates.
        tropical_curve_coords (list[tuple[float, float]]): Embedded curve coordinates.
        carpet_color (str, optional): Color for the carpet points. Defaults to 'blue'.
        curve_color (str, optional): Color for the curve line. Defaults to 'red'.
    """
    # Extract carpet coordinates
    carpet_x, carpet_y = zip(*tropical_carpet_coords)
    
    # Extract curve coordinates
    curve_x, curve_y = zip(*tropical_curve_coords)
    
    # Create scatter plot for carpet
    carpet_scatter = go.Scatter(
        x=carpet_x, 
        y=carpet_y,
        mode='markers',
        marker=dict(
            size=3,
            color=carpet_color,
            opacity=0.5
        ),
        name='Sierpinski Carpet'
    )
    
    # Create scatter plot for curve
    curve_scatter = go.Scatter(
        x=curve_x, 
        y=curve_y,
        mode='lines',
        line=dict(
            color=curve_color,
            width=2
        ),
        name='Curve'
    )
    
    # Define layout
    layout = go.Layout(
        title='Tropical Embedded 2D Sierpinski Carpet with Curve',
        xaxis_title='X',
        yaxis_title='Y',
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Create figure and display
    fig = go.Figure(data=[carpet_scatter, curve_scatter], layout=layout)
    fig.show()

def main(
    gray_code_length: int = 2,
    iterations: int = 3,
    curve_type: str = 'lissajous',
    num_curve_points: int = 1000
) -> None:
    """
    Main function to generate, embed, and visualize the 2D Sierpinski carpet and curve in tropical space.
    
    Args:
        gray_code_length (int, optional): Length of the Gray code. Defaults to 2.
        iterations (int, optional): Number of recursion steps for the Sierpinski carpet. Defaults to 3.
        curve_type (str, optional): Type of curve to generate. Defaults to 'lissajous'.
        num_curve_points (int, optional): Number of points in the curve. Defaults to 1000.
    """
    # Validate inputs
    if gray_code_length <= 0:
        raise ValueError("Gray code length must be a positive integer.")
    if iterations <= 0:
        raise ValueError("Number of iterations must be a positive integer.")
    if num_curve_points <= 0:
        raise ValueError("Number of curve points must be a positive integer.")
    
    # Generate cyclic Gray code
    gray_code_sequence = generate_gray_code(gray_code_length)
    total_gray_codes = len(gray_code_sequence)
    print(f"Generated Gray code sequence of length {gray_code_length}: {gray_code_sequence}")
    
    # Construct Sierpinski carpet
    carpet_squares = construct_sierpinski_carpet(iterations)
    total_squares = len(carpet_squares)
    print(f"Number of squares after {iterations} iterations: {total_squares}")
    
    # Generate Gray codes for each square
    gray_codes = [tuple(int(bit) for bit in gray_code_sequence[i % total_gray_codes]) 
                  for i in range(total_squares)]
    
    # Apply tropical embedding to carpet squares
    tropical_carpet_coords = tropical_embedding_2d(carpet_squares, gray_codes)
    
    # Generate curve coordinates
    curve_coords = generate_curve_coords(num_points=num_curve_points, curve_type=curve_type)
    
    # Apply tropical embedding to curve
    tropical_curve_coords = []
    for point in curve_coords:
        # Find the nearest carpet square
        x, y = point
        distances = [((x - cx)**2 + (y - cy)**2) for (cx, cy) in tropical_carpet_coords]
        nearest_index = distances.index(min(distances))
        tropical_curve_coords.append(tropical_carpet_coords[nearest_index])
    
    # Plot the embedded Sierpinski carpet and curve
    plot_carpet_with_curve(tropical_carpet_coords, tropical_curve_coords)

if __name__ == "__main__":
    main(
        gray_code_length=2,
        iterations=1,
        curve_type='lissajous',
        num_curve_points=1000
    )

    main(
        gray_code_length=2,
        iterations=2,
        curve_type='lissajous',
        num_curve_points=1000
    )
