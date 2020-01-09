

import numpy as np
from skimage.draw import ellipse
from imageio import imwrite
from random import randint
import os


def create_hole(data, src, radius, slice, rot):

    h, _, _ = data.shape

    for s in range(4):

        rr, cc = ellipse(src[0], src[1], max(radius[0]-2*s,1), max(radius[1]-2*s,1), rotation=np.deg2rad(rot))
        rr[rr > 99] = 99
        cc[cc > 99] = 99
        data[max(slice-s,0), rr,cc] = 0.6
        data[min(slice+s,h-1), rr, cc] = 0.6


    return data

if __name__ == "__main__":

    dest = "D:\\Datasets\\demo_data_plates\\"
    if not os.path.isdir(dest):
        os.mkdir(dest)

    for i in range(10000):


        plane = np.ones((8, 100, 100))*0.7

        nr_holes = randint(1,10)

        for k in range(nr_holes):
            random_posx = randint(0,100)
            random_posy = randint(0,100)

            random_radiusx = randint(1,12)
            random_radiusy = randint(1,12)

            random_rot = randint(0,180)
            random_slice = randint(0,7)
            create_hole(plane, src = (random_posx, random_posy), radius = (random_radiusx,random_radiusy), rot = random_rot, slice=random_slice)

        total = np.zeros((10, 128, 128))
        total[1:9,14:114,14:114] = plane

        if not os.path.isdir(dest+"plate_{:05d}".format(i)):
            os.mkdir(dest+"plate_{:05d}".format(i))

        for x in range(10):
            imwrite(dest+"plate_{:05d}\\slice_{}.png".format(i,x), total[x,:,:])



