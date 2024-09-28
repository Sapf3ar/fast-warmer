import pyfracgen as pf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
import random
import string
cool_maps =   ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
def generate_lyapunov_string(input_str: str) -> str:
    # Ensure the output string is between 5 and 20 characters long
    output_length = random.randint(5, 20)
    
    # Randomly choose characters from a set (you can adjust the characters as needed)
    possible_chars = input_str    
    # Generate random characters for the string (excluding "A" initially)
    random_string = ''.join(random.choices(possible_chars.replace("A", ""), k=output_length - 2))
    
    # Insert "A" at two random positions
    positions = random.sample(range(output_length), 2)
    for pos in sorted(positions):
        random_string = random_string[:pos] + 'A' + random_string[pos:]
    
    return random_string


def get_lapunov(in_str:str,xbound=(2.5, 3.4), ybound=(3.4, 4.0), width=5, height=2):
    in_str = generate_lyapunov_string(in_str)
    res = pf.lyapunov(
        in_str, xbound, ybound, width=width, height=height, dpi=100, ninit=200, niter=200
    )
#    fig, ax = pf.images.markus_lyapunov_image(
#        res, colormaps["bone"], colormaps["bone_r"], gammas=(8, 1)
#    )

    map1 =  colormaps[random.sample(cool_maps, k=1)[0]]
    map2 =  colormaps[random.sample(cool_maps, k=1)[0]]
    fig, ax = pf.images.markus_lyapunov_image(
        res, map1, map2, gammas=(6, 1)
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data


def get_random_walk(**kwargs):
    xbound = (-1.75, 0.85)
    ybound = (-1.10, 1.10)
    res = pf.buddhabrot(
        xbound,
        ybound,
        ncvals=1000,
        update_func=pf.funcs.power,
        horizon=1.0e6,
        maxiters=(100, 1000, 5000),
        width=5,
        height=2,
        dpi=100,
    )
    fig, ax = pf.images.nebula_image(tuple(res), gamma=random.random())
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


