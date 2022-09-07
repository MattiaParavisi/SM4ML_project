import pandas as pd
import matplotlib.pyplot  as plt
from PIL import Image
from pathlib import Path
import imagesize
import numpy as np
from PIL import ImageOps
from sklearn.cluster import  KMeans
import tensorflow as tf
import os
from distutils.dir_util import copy_tree

folder_path = './../../CatsDogs'
extensions = []
file = open('out.txt', 'w')
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        try:
            img = Image.open(file_path)
        except:
            print(f'image {file_path} broken')
        try:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            file.write(f'{file_path}: can be converted from {img.mode} to RGB\n')
        except Exception as e:
            print(f'Format not recognized: {e}')
            continue
        if img.format != "JPG" and img.format != "JPEG":
            print(f'format error {file_path}: {img.format}')

folder_path = './../../CatsDogs'
img_meta = {}
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    imgs = [str(img) for img in Path(sub_folder_path).iterdir() if img.suffix == ".jpg"]
    for f in imgs:
        width, height = imagesize.get(f)
        if width == -1 or height == -1:
            print(f"broken size {f}")
            image = Image.open(f)
            width, height = image.size
            print(f"now {width}, {height}")
        img_meta[f] = (width, height)

img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')

print(img_meta_df.head())
print(img_meta_df.Width.quantile(0.1))
print(img_meta_df.Height.quantile(0.1))

df2 = img_meta_df.loc[(img_meta_df['Width'] < 240.0) | (img_meta_df['Height'] < 225.0)]
df2 = df2.reset_index(drop=True)
print(df2)

for f in df2.FileName:
    print("removed: ", f)
    os.remove(f)

cond = img_meta_df['FileName'].isin(df2['FileName'])
df3 = img_meta_df.drop(img_meta_df[cond].index, inplace = False)

print(len(df3), len(img_meta_df))

df3 = df3.reset_index(drop=True)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
points = ax.scatter(df3.Width, df3.Height, color='blue', alpha=0.5, s=df3["Aspect Ratio"]*100, picker=True)
ax.set_title("Image Resolution")
ax.set_xlabel("Width", size=50)
ax.set_ylabel("Height", size=50)
fig.savefig("image_distr.png")
fig.clf()

folder_path = './../../CatsDogs_resized/'
extensions = []
file = open('out.txt', 'w')
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        try:
            img = Image.open(file_path)
        except:
            print(f'image {file_path} broken')
        try:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            file.write(f'{file_path}: can be converted from {img.mode} to RGB\n')
        except Exception as e:
            print(f'Format not recognized: {e}')
            continue
        if img.format != "JPG" and img.format != "JPEG":
            print(f'format error {file_path}: {img.format}')
print('OK')

folder_path = './../../CatsDogs_resized/'
img_meta = {}
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    imgs = [str(img) for img in Path(sub_folder_path).iterdir() if img.suffix == ".jpg"]
    for f in imgs:
        width, height = imagesize.get(f)
        if width == -1 or height == -1:
            print(f"broken size {f}")
            image = Image.open(f)
            width, height = image.size
            print(f"now {width}, {height}")
        img_meta[f] = (width, height)
img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
print(img_meta_df.head())
print(img_meta_df.Width.mean(), img_meta_df.Height.mean())
