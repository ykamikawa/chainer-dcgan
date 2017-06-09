# -*- coding: utf-8 -*-
import os
import math
import random
import glob
import numpy as np
from scipy import misc
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser(description= "face detection from scraping images")
parser.add_argument("input", help="Path to scraping image directory")
parser.add_argument("output", help="Path to output directory")
args = parser.parse_args()

INPUT_DIR = args.input
OUTPUT_DIR = args.output

#ルックアップテーブルの生成
min_table = 50
max_table = 205
diff_table = max_table - min_table
gamma1 = 0.75
gamma2 = 1.5

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )
LUT_G1 = np.arange(256, dtype = 'uint8' )
LUT_G2 = np.arange(256, dtype = 'uint8' )

LUTs = []

# 平滑化用
average_square = (10,10)

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0

for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table

for i in range(max_table, 255):
    LUT_HC[i] = 255

#その他LUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

LUTs.append(LUT_HC)
LUTs.append(LUT_LC)
LUTs.append(LUT_G1)
LUTs.append(LUT_G2)

#左右反転
def flip_left_right(image):
    image = cv2.flip(image, 1)
    return image

#ヒストグラム均一化
def equalizeHistRGB(image):

    RGB = cv2.split(image)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

#ガウシアンノイズ
def addGaussianNoise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss

    return noisy

#salt&pepperノイズ
def addSaltPepperNoise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = image.copy()
    #Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt))
                 for i in image.shape]
    out[coords[:-1]] = (255,255,255)

    #Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper))
             for i in image.shape]
    out[coords[:-1]] = (0,0,0)
    return out

def data_augmentation(image_files, data_num):
    # 画像の読み込み
    #image_lists:画像のリスト image_files:パスのリスト
    image_list = []
    file_num = len(image_files)

    #リストに画像の格納
    for image_file in image_files:
        image_list.append(cv2.imread(image_file))

    #データ数の判定
    if file_num >= data_num:
        return image_list

    #フリップ
    random.shuffle(image_list)
    for image in image_list:
        flipped_image = flip_left_right(image)
        image_list.append(flipped_image)
        if len(image_list) == data_num:
            return image_list
    
    #LUT変換
    random.shuffle(image_list)
    for image in image_list:
        for i, LUT in enumerate(LUTs):
            LUT_image = cv2.LUT(image, LUT)
            image_list.append(LUT_image)
            if len(image_list) == data_num:
                return image_list

    #平滑化
    random.shuffle(image_list)
    for image in image_list:
        gauss_image = cv2.blur(image,average_square)
        image_list.append(gauss_image)
        if len(image_list) == data_num:
            return image_list

    #ヒストグラム均一化
    random.shuffle(image_list)
    for image in image_list:
        equalized_image = equalizeHistRGB(image)
        image_list.append(equalized_image)
        if len(image_list) == data_num:
            return image_list

    #GaussianNoize付加
    random.shuffle(image_list)
    for image in image_list:
        GaussianNoized_image = addGaussianNoise(image)
        image_list.append(GaussianNoized_image)
        if len(image_list) == data_num:
            return image_list

    #SaltPepperNoise付加
    random.shuffle(image_list)
    for image in image_list:
        SaltPepperNoised_image = addSaltPepperNoise(image)
        image_list.append(SaltPepperNoised_image)
        if len(image_list) == data_num:
            return image_list

    return image_list

dir_list = os.listdir(INPUT_DIR)

for dir in dir_list:
    print(dir)
    image_files = glob.glob(os.path.join(INPUT_DIR, dir, "*.jpg"))
    if len(image_files) == 0:
        continue

    image_list = data_augmentation(image_files, 500)

    #保存
    for i, image in enumerate(image_list):
        if not os.path.exists(os.path.join(OUTPUT_DIR, dir)):
            os.mkdir(os.path.join(OUTPUT_DIR, dir))
        cv2.imwrite(os.path.join(OUTPUT_DIR, dir, str(i) + '.jpg'),image)
