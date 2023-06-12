import numpy as np
from scipy import signal


def basic_box_array(image_size: int, thickness: int) -> np.ndarray:
    """
    :param image_size: [int] - the size of the image that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell with outer pixels activated based on the desired thickness.
    The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    A = np.ones((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    A[1:-1, 1:-1] = 0  # replaces all internal rows/columns with 0's
    A = add_thickness(A, thickness)
    return A


def back_slash_array(image_size: int, thickness: int) -> np.ndarray:
    """
    :param image_size: [int] - the size of the image that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell with pixels activated along the downward diagonal based
    on the desired thickness. The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    np.fill_diagonal(A, 1)  # fills the diagonal with 1 values
    A = add_thickness(A, thickness)
    return A


def forward_slash_array(image_size: int, thickness: int) -> np.ndarray:
    """
    :param image_size: [int] - the size of the image that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell with pixels activated along the upward diagonal based on the desired
    thickness. The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    np.fill_diagonal(np.fliplr(A), 1)  # Flips the array to then fill the diagonal the opposite direction
    A = add_thickness(A, thickness)
    return A


def hot_dog_array(image_size: int, thickness: int) -> np.ndarray:
    """
    :param image_size: [int] - the size of the image that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell with outer pixel activated from the vertical center based on the
    desired thickness. The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    # Places pixels down the vertical axis to split the box
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    A[:, np.floor((image_size - 1) / 2).astype(int)] = 1  # accounts for even and odd values of image_size
    A[:, np.ceil((image_size - 1) / 2).astype(int)] = 1
    A = add_thickness(A, thickness)
    return A


def hamburger_array(image_size: int, thickness: int) -> np.ndarray:
    """
    :param image_size: [int] - the size of the image that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell with outer pixel activated from the horizontal center based on the
    desired thickness. The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    # Places pixels across the horizontal axis to split the box
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    A[np.floor((image_size - 1) / 2).astype(int), :] = 1  # accounts for even and odd values of image_size
    A[np.ceil((image_size - 1) / 2).astype(int), :] = 1
    A = add_thickness(A, thickness)
    return A


########################################################################################################################
# The function to add thickness to struts in an array
def add_thickness(array_original, thickness: int) -> np.ndarray:
    """
    :param array_original: [ndarray] - an array with thickness 1 of any shape type
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a unit cell that has been convolved to expand the number of pixels activated
    based on the desired thickness. The activated pixels are 1 (white) and the deactivated pixels are 0 (black)
    """
    A = array_original
    if thickness == 0:  # want an array of all 0's for thickness = 0
        A[A > 0] = 0
    else:
        filter_size = 2*thickness - 1 # the size of the filter needs to extend far enough to reach the base shape
        filter = np.zeros((filter_size, filter_size))
        filter[np.floor((filter_size - 1) / 2).astype(int), :] = filter[:, np.floor((filter_size - 1) / 2).astype(int)] =1
        filter[np.ceil((filter_size - 1) / 2).astype(int), :] = filter[:, np.ceil((filter_size - 1) / 2).astype(int)] = 1
        # The filter is made into a '+' shape using these functions
        convolution = signal.convolve2d(A, filter, mode='same')
        A = np.where(convolution <= 1, convolution, 1)
    return A


# The function to efficiently combine arrays in a list
def combine_arrays(arrays):
    output_array = np.sum(arrays, axis=0)  # Add the list of arrays
    output_array = np.array(output_array > 0, dtype=int)  # Convert all values in array to 1
    return output_array
