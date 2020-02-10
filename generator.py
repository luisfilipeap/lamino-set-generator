

import numpy as np
from skimage.draw import ellipse
import matplotlib.pyplot as plt

flat_plane_width = 250
flat_plane_height = 250
flat_plane_length = 10

number_of_planes = 10000

def create_hole(data, slice, pos, radiusC, radiusR):
    #function to create a 3D hole inside a parallelide
    _, _, s = data.shape

    for k in range(min(radiusC, radiusR)):
            rr, cc = ellipse(pos[0], pos[1], radiusC-2*k, radiusR-2*k,0)
            if slice-k >= 0:
                data[rr, cc, slice-k] = 0
            if slice+k < s:
                data[rr, cc, slice+k] = 0
    return data


for plane in range(number_of_planes):
    data = np.ones((flat_plane_height, flat_plane_width, flat_plane_length))



if __name__ == "__main__":

    p = np.ones((250,250, 10))
    p = create_hole(p, 5, (125, 125), 4, 4)

    plt.figure()
    plt.imshow(p[:,:,0])
    plt.figure()
    plt.imshow(p[:, :, 2])
    plt.figure()
    plt.imshow(p[:, :, 4])
    plt.figure()
    plt.imshow(p[:, :, 6])
    plt.figure()
    plt.imshow(p[:, :, 8])

    plt.show()