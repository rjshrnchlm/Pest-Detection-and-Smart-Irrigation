import os

import cv2
import numpy as np
import pandas as pd

# Read Crop Image
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Crop Image/crop_images'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            fileName = folder + '/' + in_dir[j]
            Img = cv2.imread(fileName)
            Image.append(Img)
            Target.append(i)

        # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Image.npy', Shuffled_Datas)
    np.save('Crop_Target.npy', Shuffled_Target)

# Read Pest Image
an = 0
if an == 1:
    Image = []
    Target = []
    path = './pest Image/pest/train'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            fileName = folder + '/' + in_dir[j]
            image = cv2.imread(fileName)
            resize_img = cv2.resize(image, (256, 256))
            Image.append(resize_img)
            Target.append(i)

    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Image.npy', Shuffled_Datas)
    np.save('Pest_Target.npy', Shuffled_Target)


