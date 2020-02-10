import os
from imageio import imread
from skimage import measure
import numpy as np
from matplotlib import pyplot as plt

src_gt = "D:\\Datasets\\demo_data_plates\\"
src_sirt = "D:\\Datasets\\demo_plates_4_projs\\input\\"

T = len(os.listdir(src_gt))
sirt_mse = np.zeros((T,128))

t = 0
for folder in os.listdir(src_gt):
    gt_volume = np.zeros((10,128,128))
    sirt_volume = np.zeros((10,128,128))

    k = 0
    for file in os.listdir(src_gt+folder):
        gt_volume[k, :,:] = imread(src_gt+folder+"\\"+file, pilmode="F")/255
        sirt_volume[k, :, :] = imread(src_sirt+folder+"\\"+file, pilmode="F")/255
        k = k + 1


    for z in range(0,128):
        sirt_mse[t, z] = measure.compare_mse(gt_volume[:,:,z], sirt_volume[:,:,z])
    t = t + 1
    print(t/T)

final = np.average(sirt_mse,0)

plt.figure()
plt.plot(final)
plt.plot(np.linspace(13,13,50), np.linspace(0,0.2,50), 'r')
plt.plot(np.linspace(114,114,50), np.linspace(0,0.2,50), 'r')
plt.show()


