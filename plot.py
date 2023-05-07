import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_3d_heatmap(img_numpy: np.ndarray, heat_min=0, save_path=None):
    """
    :param img_numpy: 3D numpy array
    :param heat_min: minimum value being plotted
    """
    assert img_numpy.ndim == 3

    # remove points with 0 value
    indices = np.argwhere(img_numpy > heat_min) # shape=(n, 3)
    intensity = []
    for x, y, z in indices:
        intensity.append(img_numpy[x][y][z])

    
    # creating figures
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # setting color bar
    color_map = plt.cm.ScalarMappable(cmap=plt.colormaps['viridis'])
    color_map.set_array(intensity)
    
    # creating the heatmap
    indices = indices.T
    img = ax.scatter(indices[0], indices[1], indices[2],
                     c=intensity, marker='o', s=2, cmap='viridis')
    plt.colorbar(color_map)
    if save_path != None:
        plt.savefig(save_path)
    plt.close()

def plot_mid_slice(img_numpy, size_mutiplier=10, save_path=None):
    """
    Accepts an 3D numpy array and shows median slices in all three planes
    :param img_numpy: 3D numpy array
    """
    
    assert img_numpy.ndim == 3
    n_i, n_j, n_k = img_numpy.shape

    # saggital
    center_i1 = int((n_i - 1) / 2)
    # transverse
    center_j1 = int((n_j - 1) / 2)
    # axial slice
    center_k1 = int((n_k - 1) / 2)

    plot_slices([img_numpy[center_i1, :, :],
                 img_numpy[:, center_j1, :],
                 img_numpy[:, :, center_k1]], size_mutiplier=size_mutiplier, save_path=save_path, ncols=3)

def plot_every_slice(img_numpy, size_mutiplier=1, save_path=None):
    """
    Accepts an 3D numpy array and shows every slices in the first planes
    :param img_numpy: 3D numpy array
    """
    nslices = img_numpy.shape[0]
    slices = []
    for i in range(nslices):
        slices.append(img_numpy[i, :, :])
    plot_slices(slices, size_mutiplier=size_mutiplier, save_path=save_path)


def plot_slices(slices, save_path='show_slice.png', size_mutiplier=1, ncols=5):
    """
    Function to display a row of image slices
    :param img_numpy: A list of numpy 2D image slices
    """
    
    nrows = math.ceil(len(slices) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(size_mutiplier*nrows, size_mutiplier*nrows))
    axes = axes.flatten()

    for i, slice in enumerate(slices):
        im = axes[i].imshow(slice.T, cmap=plt.colormaps['viridis'])
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)
    plt.subplots_adjust(wspace=1)
    if save_path != None:
        fig.savefig(save_path)
    plt.close()