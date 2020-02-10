from imageio import imread, imwrite
from object_scan import ScanningObject
import os
import numpy as np

import pandas as pd

projs = 100

data_src = "D:\\Datasets\\demo_data_plates\\"
data_dest0 = "D:\\Datasets\\demo_plates_{}_projs\\".format(projs)
data_dest = "D:\\Datasets\\demo_plates_{}_projs\\input-TV\\".format(projs)

if not os.path.isdir(data_dest):
    #os.mkdir(data_dest0)
    os.mkdir(data_dest)

set = pd.read_csv("test2.csv")

for folder in os.listdir(data_src):

    if set.iloc[:,0].str.contains(folder).any():
        print(folder)
        plane = np.zeros((10, 128, 128))
        k = 0
        for file in os.listdir(data_src+folder):
            i = imread(data_src + folder + "\\" + file, pilmode='F')
            plane[k, :, :] = i
            k = k + 1

        setup = ScanningObject(alpha_param=30, n_cells_param=256, n_proj_param=projs, rec_size_param=128)
        out = setup.run(plane, rec_algorithm_param='TV3D')

        if not os.path.isdir(data_dest+folder):
            os.mkdir(data_dest+folder)

        for k in range(10):
            imwrite(data_dest+folder+"\\slice_{}.png".format(k), out['rec'][k,:,:])

