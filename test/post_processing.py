import cv2
import numpy as np


def result_map_to_img(pred):

    """
    Generate the segmentation map for the test images.

    Arguments:
    pred -- pedicted output from the model.

    Shows the segmentation map of the test image.

    """
    # each cass is assigned a color in the segmented output.
    color_map = {
        '0': [0, 0, 0],
        '1': [196, 8, 206],
        '2': [27, 22, 186],
        '3': [242, 4, 4],
        '4': [3, 79, 5],
        '5': [10, 211, 17],
        '6': [246, 255, 0],

    }

    res_map = np.zeros((256, 512, 3), dtype=np.uint8)

    argmax_idx = np.argmax(pred, axis=2)

    for i in range(0, 256):
      for j in range(0, 512):
        res_map[i, j] = color_map[str(argmax_idx[i, j])]

    cv2_imshow(res_map)
    cv2.waitKey(0)
