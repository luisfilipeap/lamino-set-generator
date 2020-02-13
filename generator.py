



import numpy as np
from skimage.draw import ellipse
import matplotlib.pyplot as plt
import random
from imageio import imwrite
import os
from skimage.util import random_noise



def create_ellipsoid_hole(data, src, radius, slice, rot):

    h, _, _ = data.shape

    for s in range(4):

        rr, cc = ellipse(src[0], src[1], max(radius[0]-2*s,1), max(radius[1]-2*s,1), rotation=np.deg2rad(rot))
        rr[rr > 255] = 255
        cc[cc > 255] = 255
        data[max(slice-s,0), rr,cc] = 0.6
        data[min(slice+s,h-1), rr, cc] = 0.6

    return data

def create_simple_lamino_set_of_holes():
    dest = ".\\data_plates\\"
    if not os.path.isdir(dest):
        os.mkdir(dest)

    for i in range(10000):
        plane = np.ones((10, 256, 256)) * 0.7

        nr_holes = randint(1, 10)

        for k in range(nr_holes):
            random_posx = randint(0, 255)
            random_posy = randint(0, 255)

            random_radiusx = randint(1, 25)
            random_radiusy = randint(1, 25)

            random_rot = randint(0, 180)
            random_slice = randint(0, 9)
            create_hole(plane, src=(random_posx, random_posy), radius=(random_radiusx, random_radiusy), rot=random_rot,
                        slice=random_slice)

        if not os.path.isdir(dest + "plate_{:05d}".format(i)):
            os.mkdir(dest + "plate_{:05d}".format(i))

        for x in range(10):
            imwrite(dest + "plate_{:05d}\\slice_{}.png".format(i, x), plane[x, :, :])



def wood_floor_slice_circular_fit(slice, pos, radius):
    size = np.shape(slice)

    y = np.uint16(np.linspace(0,size[0],size[0]))
    x = np.uint16(np.linspace(0,size[1],size[1]))

    xv, yv = np.meshgrid(x,y)

    slice[np.power(xv-pos[0],2)+np.power(yv-pos[1],2) <= np.power(radius,2)] = 0

    return slice, radius

def wood_floor_slice_retangular_fit(plane, pos1, pos2):
    size = np.shape(plane)


    y = np.uint16(np.linspace(0,size[0],size[0]))
    x = np.uint16(np.linspace(0,size[1],size[1]))

    xv, yv = np.meshgrid(x,y)

    plane[np.logical_and(np.logical_and(np.logical_and(xv >= pos1[0], xv <= pos2[0]), yv >= pos1[1]), yv <=pos2[1])] = 0

    return plane, pos1[0]

def rect_attachment(slice, dim, pos):
    size = np.shape(slice)
    x0 = max(0,size[1]-random.randint(2,6))
    y0 = max(0, size[0]-pos)
    x1 = size[1]
    y1 = y0 + dim
    slice, xmin = wood_floor_slice_retangular_fit(slice, (x0, y0), (x1, y1))

    return slice, xmin

def circ_attachment(slice, dim, pos):
    size = np.shape(slice)
    xc = size[1]
    yc = max(10, size[0]-pos)

    slice, radius = wood_floor_slice_circular_fit(slice, (xc,yc), dim)

    return slice, radius

def buid_circ_attachments(slice, number=1):
    ydim, xdim = np.shape(slice)

    max_radius = 0
    for n in range(number):
        slice, radius = circ_attachment(slice, random.randint(4,int(ydim/4)), random.randint(3, ydim-3))
        if radius > max_radius:
            max_radius = radius

    mask = slice[:, xdim-max_radius:xdim]
    tmp = np.ones((ydim, xdim-max_radius))
    mask = np.logical_not(mask)
    mask = np.concatenate((mask, tmp), axis=1)
    final = np.logical_and(mask, slice)

    return final, max_radius

def build_rect_attachments(slice, number=1):
    ydim, xdim = np.shape(slice)

    xmin = xdim
    for n in range(number):
        slice, x = rect_attachment(slice, random.randint(1,int(ydim/4)), random.randint(3, ydim-3))
        if x < xmin:
            xmin = x

    mask = slice[:, xmin:xdim]
    tmp = np.ones((ydim, xmin))
    mask = np.logical_not(mask)
    mask = np.concatenate((mask, tmp), axis=1)
    final = np.logical_and(mask, slice)

    return final, xmin


def random_volume(x,y,z):
    base = np.ones((x,y,z))


    maskrect, xmin = build_rect_attachments(np.ones((x,y)), random.randint(1,4))
    maskcirc, rad = buid_circ_attachments(np.ones((x,y)), random.randint(1,3))

    m = min(rad,y-xmin)

    for v in range(z):
        if v < (z/2)-2:
            base[:,:, v] = random_noise(base[:,:, v],mode="pepper")
            base[:,:,v] = np.logical_and(maskrect, base[:,:,v])
        elif v >= (z/2)+2:
            base[:, :, v] = random_noise(base[:, :, v], mode="pepper")
            base[:,:,v] = np.logical_and(maskcirc, base[:,:,v])
        else:
            base[:, :, v] = random_noise(base[:, :, v], mode="pepper")
            base[:,0:m, v] = 0

    return base

if __name__ == "__main__":
    src = "D:\\Datasets\\lamino_attachable\\"
    for k in range(10000):

        block = np.zeros((32, 64, 16))
        v = random_volume(24,48,12)

        block[4:28, 8:56, 2:14] = v

        dest = "lamino_{}\\".format(str(k).zfill(4))

        if not os.path.isdir(src+dest):
            os.mkdir(src+dest)

        for z in range(16):
            imwrite("{}slice_{}.png".format(src+dest, z), block[:,:,z])

