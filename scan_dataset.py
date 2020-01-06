from imageio import imread, imwrite
from object_scan import ScanningObject
import os
import numpy as np

projs = 8

data_src = "D:\\Datasets\\demo_data_plates\\"
data_dest = "D:\\Datasets\\demo_plates_{}_projs\\".format(projs)

if not os.path.isdir(data_dest):
    os.mkdir(data_dest)

for folder in os.listdir(data_src):

    plane = np.zeros((10, 256, 256))
    k = 0
    for file in os.listdir(data_src+folder):
        i = imread(data_src + folder + "\\" + file, pilmode='F')
        plane[k, :, :] = i
        k = k + 1

    setup = ScanningObject(alpha_param=30, n_cells_param=1024, n_proj_param=projs, rec_size_param=256)
    out = setup.run(plane)

    if not os.path.isdir(data_dest+folder):
        os.mkdir(data_dest+folder)

    for k in range(10):
        imwrite(data_dest+folder+"\\slice_{}.png".format(k), out['rec'][k,:,:])

    print(folder)