from imgaug import augmenters as iaa
import imgaug as ia
import os
import cv2 as cv
import numpy as np
import re

def img_aug(orignal_image,ground_truth_img,N):
    img_array = [orignal_image]
    heatmap_array = [ground_truth_img]
    seq = iaa.Sequential(
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            # rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=0,  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
    seq_det_list = seq.to_deterministic(N)
    for i in range(N):
        new_img = seq_det_list[i].augment_image(orignal_image)
        new_heatmap = seq_det_list[i].augment_image(ground_truth_img)
        img_array.append(new_img)
        heatmap_array.append(new_heatmap)
    return img_array,heatmap_array

def convert_heatmap(heatmap_list):
    pass


def load_data(N=10):
    training_dir = 'training/images'
    ground_truth_dir = 'training/groundtruth'
    train_img_list = os.listdir(training_dir)
    img_list = []
    heatmap_list = []
    for i in train_img_list:
        print(i)
        img = cv.imread(os.path.join(training_dir, i)) / 255
        heatmap = cv.imread(os.path.join(ground_truth_dir,i))/255
        new_img, new_heatmap = img_aug(img,heatmap,N)
        img_list.extend(new_img)
        heatmap_list.extend(new_heatmap)
    img_list = np.array(img_list)
    heatmap_list = np.array(heatmap_list)
    heatmap_list = heatmap_list[:,:,:,:1]
    return img_list,heatmap_list

# count = 0
# bad_list = []
# for i in range(len(x_train)):
#     if np.max(x_train[i]) > 1:
#         count += 1
#         bad_list.append(i)
# print(count)

def load_test_image():
    test_dir = 'test_images/'
    train_img_list = os.listdir(test_dir)
    img_list = []
    img_id = []
    for i in train_img_list:
        print(i)
        img_number = int(re.search(r"\d+", i).group(0))
        img = cv.imread(os.path.join(test_dir, i)) / 255
        img_list.append(img)
        img_id.append(img_number)
    img_list = np.array(img_list)
    return img_list, img_id