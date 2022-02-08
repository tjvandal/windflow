import numpy as np

def split_array(arr, tile_size=128, overlap=16):
    '''
    Split a 3D numpy array into patches for inference
    (Channels, Height, Width)
    Args:
        tile_size: width and height of patches to return
        overlap: number of pixels to overlap between patches
    Returns:
        dict(patches, upper_left): patches and indices of original array
    '''
    arr = arr[np.newaxis]
    width, height = arr.shape[2:4]
    arrs = dict(patches=[], upper_left=[])
    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            i = min(i, width - tile_size)
            j = min(j, height - tile_size)
            arrs['patches'].append(arr[:,:,i:i+tile_size,j:j+tile_size])
            arrs['upper_left'].append([[i,j]])
    arrs['patches'] = np.concatenate(arrs['patches'])
    arrs['upper_left'] = np.concatenate(arrs['upper_left'])
    return arrs['patches'], arrs['upper_left']

