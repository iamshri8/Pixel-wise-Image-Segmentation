import cv2
import numpy as np
from model import unet

labels = ['background', 'person', 'car', 'road', 'sky', 'building', 'tree']


def test():

    """
    Inferencing the test image and generating the segmentation output for the input.

    """
    test_model = unet(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
    test_model.load_weights('path to your trained model')

    x_img = cv2.imread('test.png')
    x_img =cv2.resize(x_img,(512,256))
    cv2.imshow(x_img)
    x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
    x_img = x_img / 127.5 - 1
    x_img = np.expand_dims(x_img, 0)

    pred = test_model.predict(x_img)
    result_map_to_img(pred[0])

test()
