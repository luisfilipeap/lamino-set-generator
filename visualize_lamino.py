import os
from imageio import imread, imwrite
import numpy as np

src = "D:\\Datasets\\demo_plates_100_projs\\input\\plate_00263\\"
srcpred = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-UNET3D-4-projs\\plate_00001\\teste\\"

desc = "sirt_100_projs_sample"
view = "coronal"

volume = np.zeros((10,128,128))
count = 0

for file in os.listdir(src):
    print(file)
    volume[count,:,:] = imread(src+file, pilmode="F")
    count = count + 1


if view == "axial":
    imwrite(desc+"_"+view+".png", volume[9,:,:])
elif view == "sagital":
    imwrite(desc + "_" + view + ".png", volume[:, 96, :].transpose())
else:
    imwrite(desc + "_" + view + ".png", volume[:, :, 61].transpose())