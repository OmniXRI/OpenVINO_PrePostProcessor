#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

from openvino.runtime import Model, Shape, op, opset8

# Saving model with preprocessing to IR
from openvino.runtime.passes import Manager
import math
from timeit import default_timer

def main():
    start_time = default_timer()

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    image_path = sys.argv[1]

    # Read input image
    img = cv2.imread(image_path)

    array_alpha = np.array([2.0]) # contrast
    array_beta = np.array([0.0]) # brightness

    # Multiply every pixel value by alpha
    img = cv2.multiply(img, array_alpha)

    # add a beta value to every pixel
    img = cv2.add(img, array_beta)

    # The value of each pixel should be limited in [0, 255]
    img = np.clip(img, 0, 255)

    cv2.imwrite('contrast_opencv_linear.jpg', img)
    
    total_time = default_timer() - start_time
    print("total time : {0:.3f}ms".format(total_time *1000))

    return 0


if __name__ == '__main__':
    sys.exit(main())