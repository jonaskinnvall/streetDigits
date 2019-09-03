# Lib imoprts
import numpy as np
import matplotlib.pyplot as plt


def imagePP(data):
    # Show images
    for i in range(10):
        print('Train y1: ', data['y'][i])
        plt.imshow(data['X'][:, :, :, i])
        plt.show()

    return data
