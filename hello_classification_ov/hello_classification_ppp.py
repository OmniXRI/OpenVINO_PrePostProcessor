#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import time

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type


def main():
    t0 = time.time()

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # Parsing and validation of input arguments
    if len(sys.argv) != 4:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name>')
        return 1

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device_name = sys.argv[3]

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    t1 = time.time()

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    t2 = time.time()

    # Read input image
    image = cv2.imread(image_path)
    # Add N dimension
    input_tensor = np.expand_dims(image, 0)

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    t3 = time.time()

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    t4 = time.time()

    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model_path, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    t5 = time.time()

    log.info('Starting inference in synchronous mode')
    results = compiled_model.infer_new_request({0: input_tensor})

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    t6 = time.time()

    predictions = next(iter(results.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    probs = predictions.reshape(-1)

    # Get an array of 10 class IDs in descending order of probability
    top_10 = np.argsort(probs)[-10:][::-1]

    header = 'class_id probability'

    log.info(f'Image path: {image_path}')
    log.info('Top 10 results: ')
    log.info(header)
    log.info('-' * len(header))

    for class_id in top_10:
        probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
        log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

    log.info('')

# ----------------------------------------------------------------------------------------------------------------------
    t7 = time.time()

    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')

    total_time = t7 -t0

    print("Total time :          %.3f ms, %.2f %%" % ((t7-t0)*1000,((t7-t0)/total_time *100)))
    print("inital time :         %.3f ms, %.2f %%" % ((t1-t0)*1000,((t1-t0)/total_time *100)))
    print("read model time :     %.3f ms, %.2f %%" % ((t2-t1)*1000,((t2-t1)/total_time *100)))
    print("setup imput time :    %.3f ms, %.2f %%" % ((t3-t2)*1000,((t3-t2)/total_time *100)))
    print("preprocessing time :  %.3f ms, %.2f %%" % ((t4-t3)*1000,((t4-t3)/total_time *100)))
    print("loading model time :  %.3f ms, %.2f %%" % ((t5-t4)*1000,((t5-t4)/total_time *100)))
    print("inference time :      %.3f ms, %.2f %%" % ((t6-t5)*1000,((t6-t5)/total_time *100)))
    print("process output time : %.3f ms, %.2f %%" % ((t7-t6)*1000,((t7-t6)/total_time *100)))

    return 0


if __name__ == '__main__':
    sys.exit(main())