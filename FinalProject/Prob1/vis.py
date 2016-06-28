import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave

def unit_scale(x, eps=1e-8):
    """Scales values of ndarray x into [0, 1]"""
    return (x - x.min()) / (x.max() - x.min() + eps)
    
def visualize_W(W, flt_shape, show=True, save=False, transform=unit_scale):
    """Visualize weight matrix W in which each row corresponds to a filter"""
    grid_width = int(np.ceil(np.sqrt(len(W))))
    
    # Init grid, pad "1 pixel" spaces between filters
    grid = np.zeros((grid_width * flt_shape[0] + grid_width - 1,
                     grid_width * flt_shape[1] + grid_width - 1))

    # Go through flattened filters, reshape them, and put them on grid
    for flt_idx, flt in enumerate(W):
        grid_row = flt_idx / grid_width
        grid_col = flt_idx % grid_width
        if transform:
            flt = transform(flt)
        grid[grid_row*flt_shape[0]+grid_row:grid_row*flt_shape[0]+grid_row+flt_shape[0],
             grid_col*flt_shape[1]+grid_col:grid_col*flt_shape[1]+grid_col+flt_shape[1]]\
             = flt.reshape(flt_shape)

    if show:
        plt.imshow(grid, cmap='gray')
        plt.show()
    if save:
        imsave(save, grid)