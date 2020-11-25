import random
import cv2
import numpy as np


def get_one_hot_label(b_size, y_label):

    """
    A function to generate one-hot encodings.

    Arguments:
    b_size -- batch size
    y_label -- list of class ids of the batch images

    Returns:
    one_hot_label -- returns the generated one-hot labels.

    """
    y_label = np.squeeze(y_label, axis=3)

    one_hot_label = np.zeros((b_size, 256, 512, 7))

    building = (y_label == 11)
    car = (y_label == 26)
    person = (y_label == 24)
    tree = (y_label == 21)
    grass = (y_label == 22)
    truck = (y_label == 27)
    background = np.logical_not(building + car + person + tree + grass + truck)

    one_hot_label[:, :, :, 0] = np.where(background, 1, 0)
    one_hot_label[:, :, :, 1] = np.where(building, 1, 0)
    one_hot_label[:, :, :, 2] = np.where(car, 1, 0)
    one_hot_label[:, :, :, 3] = np.where(person, 1, 0)
    one_hot_label[:, :, :, 4] = np.where(tree, 1, 0)
    one_hot_label[:, :, :, 5] = np.where(grass, 1, 0)
    one_hot_label[:, :, :, 6] = np.where(truck, 1, 0)


    return one_hot_label


def pre_processing(img):

    """
    Pre-preprocessing of the input image.

    Arguments:
    img -- input image

    Returns:
    returns the normalized image having values between -1 to +1

    """
    # Random exposure and saturation (0.9 ~ 1.1 scale)
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Centering helps normalization image (-1 ~ 1 value)
    return img / 127.5 - 1
