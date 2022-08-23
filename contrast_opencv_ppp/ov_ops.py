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
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    image_path = sys.argv[1]
    device_name = sys.argv[2]

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step2 Model() to create model instead of reading model ----------------------------------
    input_shape = [1, 1200, 1801, 3]
    param_node = op.Parameter(Type.f32, Shape(input_shape))
    
    # Algorithm of Brightness Contrast transformation
    brightness = 0
    contrast = -100
    B = brightness / 255.0
    c = contrast / 255.0
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    constant_1 = np.float32(-127.5 * (1 - B))
    constant_2 = np.float32(k)
    constant_3 = np.float32(127.5 * (1 + B))
    contrast_node_1 = opset8.add(param_node,
      opset8.constant(constant_1))
    contrast_node_2 = opset8.multiply(contrast_node_1, \
      opset8.constant(constant_2))
    contrast_node_3 = opset8.clamp( opset8.add( \
      contrast_node_2, opset8.constant(constant_3)), \
      min_value=0.0, max_value=255.0)
    model = Model(contrast_node_3, [param_node],
      'contrast_adjust')

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input image
    image = cv2.imread(image_path)
    # Add N dimension
    input_tensor = np.expand_dims(image, 0)

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = PrePostProcessor(model)

    _, h, w, _ = input_tensor.shape

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - reuse precision and shape from already available `input_tensor`
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(Type.f32) \
        .set_layout(Layout('NHWC')) \
        .set_spatial_static_shape(h, w) # noqa: ECE001, N400

    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Here we suppose model has 'NHWC' layout for input
    ppp.input().model().set_layout(Layout('NHWC'))

    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 5) Apply preprocessing modifying the original 'model'
    model = ppp.build()

    # Saving model Step
    pass_manager = Manager()
    pass_manager.register_pass(pass_name="Serialize",
    xml_path='simple_model_saved.xml',
    bin_path='simple_model_saved.bin')
    pass_manager.run_passes(model)

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    start_infer_time = default_timer()

    results = compiled_model.infer_new_request({0: input_tensor})

    infer_time = default_timer() - start_infer_time
    print("infer time : {0:.3f}ms".format(infer_time*1000))

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    predictions = next(iter(results.values()))
    cv2.imwrite('contrast.jpg', np.squeeze(predictions))

    return 0


if __name__ == '__main__':
    sys.exit(main())