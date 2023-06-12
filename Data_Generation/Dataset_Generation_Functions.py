import numpy as np
from Data_Generation.Piecewise_Box_Functions import basic_box_array, back_slash_array, forward_slash_array, hamburger_array, hot_dog_array

# For Internal Testing
# from Piecewise_Box_Functions import basic_box_array, back_slash_array, forward_slash_array, hamburger_array, hot_dog_array
import pandas as pd
import json
import matplotlib.pyplot as plt
from json import JSONEncoder


########################################################################################################################
# Make the data using all the code in Shape_Generation_Functions.py
def make_boxes(image_size: int, densities: list) -> list:
    """
    :param image_size: [int] - the pixel height and width of the generated arrays
    :param densities: [list[float]] - of the desired pixel values to apply to active pixels - Recommend values (0,1]
    :return: list[tuple] - [Array, Density, Thickness of each strut type] this is all the defining information for
    all the generated data.
    """

    matrix = []

    # Establish the maximum thickness for each type of strut
    max_vert = int(np.ceil(1 / 2 * image_size) - 2)
    max_diag = int(image_size - 3)
    max_basic = int(np.ceil(1 / 2 * image_size) - 1)

    # Adds different density values
    for i in range(len(densities)):
        for j in range(1, max_basic):  # basic box loop, always want a border
            basic_box_thickness = j
            array_1 = basic_box_array(image_size, basic_box_thickness)
            if np.unique([array_1]).all() > 0:  # Checks if there is a solid figure
                break

            for k in range(0, max_vert):
                hamburger_box_thickness = k
                array_2 = hamburger_array(image_size, hamburger_box_thickness) + array_1
                array_2 = np.array(array_2 > 0, dtype=int)  # Keep all values 0/1
                if np.unique([array_2]).all() > 0:
                    break

                for l in range(0, max_vert):
                    hot_dog_box_thickness = l
                    array_3 = hot_dog_array(image_size, hot_dog_box_thickness) + array_2
                    array_3 = np.array(array_3 > 0, dtype=int)
                    if np.unique([array_3]).all() > 0:
                        break

                    for m in range(0, max_diag):
                        forward_slash_box_thickness = m
                        array_4 = forward_slash_array(image_size, forward_slash_box_thickness) + array_3
                        array_4 = np.array(array_4 > 0, dtype=int)
                        if np.unique([array_4]).all() > 0:
                            break

                        for n in range(0, max_diag):
                            back_slash_box_thickness = n
                            array_5 = back_slash_array(image_size, back_slash_box_thickness) + array_4
                            array_5 = np.array(array_5 > 0, dtype=int)
                            if np.unique([array_5]).all() > 0:
                                break
                            the_tuple = (array_5*densities[i], densities[i], basic_box_thickness,
                                         forward_slash_box_thickness, back_slash_box_thickness,
                                         hot_dog_box_thickness, hamburger_box_thickness)
                            matrix.append(the_tuple)

    return matrix


########################################################################################################################
# How to read the files
'''
df = pd.read_csv('2D_Lattice.csv')
print(np.shape(df))
row = 1
box = df.iloc[row, 1]
array = np.array(json.loads(box))
plt.imshow(array, vmin=0, vmax=1)
plt.show()
'''
