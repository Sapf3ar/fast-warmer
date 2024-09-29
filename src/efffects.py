import random
import numpy as np
import pyfracgen as pf
from matplotlib import colormaps

# List of available colormap names
COOL_COLORMAPS = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
]

def generate_lyapunov_sequence(
    possible_chars: str,
    length_range: tuple = (5, 20),
    num_fixed_chars: int = 2,
    fixed_char: str = 'A'
) -> str:
    """
    Generates a Lyapunov sequence string for fractal generation.

    The function creates a random sequence of characters from `possible_chars`,
    of random length within `length_range`, and inserts `num_fixed_chars`
    occurrences of `fixed_char` at random positions in the sequence.

    Args:
        possible_chars (str): The set of characters to choose from.
        length_range (tuple): A tuple (min_length, max_length) specifying the length range.
        num_fixed_chars (int): Number of times to insert the `fixed_char` into the sequence.
        fixed_char (str): The character to insert at random positions.

    Returns:
        str: The generated Lyapunov sequence string.
    """
    min_length, max_length = length_range
    sequence_length = random.randint(min_length, max_length)
    
    # Exclude the fixed character initially
    chars_without_fixed = possible_chars.replace(fixed_char, "")
    
    # Generate random characters excluding the fixed character
    random_length = sequence_length - num_fixed_chars
    random_chars = random.choices(chars_without_fixed, k=random_length)
    
    # Insert fixed characters at random positions
    sequence = random_chars.copy()
    positions = random.sample(range(sequence_length), num_fixed_chars)
    for pos in positions:
        sequence.insert(pos, fixed_char)
    
    return ''.join(sequence)

def get_lyapunov_fractal(
    input_chars: str,
    xbound: tuple = (2.5, 3.4),
    ybound: tuple = (3.4, 4.0),
    width: int = 5,
    height: int = 2,
    dpi: int = 100,
    ninit: int = 200,
    niter: int = 200
) -> np.ndarray:
    """
    Generates a Lyapunov fractal image based on the input characters.

    Args:
        input_chars (str): String of possible characters to generate the Lyapunov sequence.
        xbound (tuple): Tuple specifying the range of x-values (x_min, x_max).
        ybound (tuple): Tuple specifying the range of y-values (y_min, y_max).
        width (int): Width of the output image in inches.
        height (int): Height of the output image in inches.
        dpi (int): Dots per inch for the output image.
        ninit (int): Number of initial iterations to discard.
        niter (int): Number of iterations to compute.

    Returns:
        np.ndarray: Numpy array representing the generated image.
    """
    sequence = generate_lyapunov_sequence(input_chars)
    result = pf.lyapunov(
        sequence,
        xbound,
        ybound,
        width=width,
        height=height,
        dpi=dpi,
        ninit=ninit,
        niter=niter
    )

    # Choose two random colormaps
    cmap1_name = random.choice(COOL_COLORMAPS)
    cmap2_name = random.choice(COOL_COLORMAPS)
    cmap1 = colormaps[cmap1_name]
    cmap2 = colormaps[cmap2_name]

    fig, ax = pf.images.markus_lyapunov_image(
        result, cmap1, cmap2, gammas=(6, 1)
    )

    # Extract image data from the figure
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height_px, width_px, 3))
    
    return image

def get_buddhabrot_fractal(
    xbound: tuple = (-1.75, 0.85),
    ybound: tuple = (-1.10, 1.10),
    ncvals: int = 1000,
    update_func=pf.funcs.power,
    horizon: float = 1.0e6,
    maxiters: tuple = (100, 1000, 5000),
    width: int = 5,
    height: int = 2,
    dpi: int = 100
) -> np.ndarray:
    """
    Generates a Buddhabrot fractal image.

    Args:
        xbound (tuple): Tuple specifying the range of x-values (x_min, x_max).
        ybound (tuple): Tuple specifying the range of y-values (y_min, y_max).
        ncvals (int): Number of c-values to sample.
        update_func (callable): The function to update iterations.
        horizon (float): The escape radius for the iteration.
        maxiters (tuple): Tuple specifying maximum iterations for different layers.
        width (int): Width of the output image in inches.
        height (int): Height of the output image in inches.
        dpi (int): Dots per inch for the output image.

    Returns:
        np.ndarray: Numpy array representing the generated image.
    """
    result = pf.buddhabrot(
        xbound,
        ybound,
        ncvals=ncvals,
        update_func=update_func,
        horizon=horizon,
        maxiters=maxiters,
        width=width,
        height=height,
        dpi=dpi,
    )
    gamma_value = random.uniform(0.5, 1.5)
    fig, ax = pf.images.nebula_image(result, gamma=gamma_value)
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height_px, width_px, 3))
    return image
