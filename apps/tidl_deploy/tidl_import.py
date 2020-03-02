"""
* Copyright (c) {2015 - 2017} Texas Instruments Incorporated
*
* All rights reserved not granted herein.
*
* Limited License.
*
* Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
* license under copyrights and patents it now or hereafter owns or controls to make,
* have made, use, import, offer to sell and sell ("Utilize") this software subject to the
* terms herein.  With respect to the foregoing patent license, such license is granted
* solely to the extent that any such patent is necessary to Utilize the software alone.
* The patent license shall not apply to any combinations which include this software,
* other than combinations with devices manufactured by or for TI ("TI Devices").
* No hardware patent is licensed hereunder.
*
* Redistributions must preserve existing copyright notices and reproduce this license
* (including the above copyright notice and the disclaimer and (if applicable) source
* code license limitations below) in the documentation and/or other materials provided
* with the distribution
*
* Redistribution and use in binary form, without modification, are permitted provided
* that the following conditions are met:
*
* *     No reverse engineering, decompilation, or disassembly of this software is
* permitted with respect to any software provided in binary form.
*
* *     any redistribution and use are licensed by TI for use only with TI Devices.
*
* *     Nothing shall obligate TI to provide you with source code for the software
* licensed and provided to you in object code.
*
* If software source code is provided to you, modification and redistribution of the
* source code are permitted provided that the following conditions are met:
*
* *     any redistribution and use of the source code, including any resulting derivative
* works, are licensed by TI for use only with TI Devices.
*
* *     any redistribution and use of any object code compiled from the source code
* and any resulting derivative works, are licensed by TI for use only with TI Devices.
*
* Neither the name of Texas Instruments Incorporated nor the names of its suppliers
*
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* DISCLAIMER.
*
* THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import subprocess
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_TYPE_TF     = 1
MODEL_TYPE_ONNX   = 2
MODEL_TYPE_TFLITE = 3

################################################################################
# Function tidl_check_model: 
#   Check if a given model is supported by TIDL. Currently, only Tensorflow
#   models can be recognized by this script. 
# Input:
#       model_file:        file name of the given model
#       calib_image:       file name of image for TIDL calibration
#       subgraph_name:     base name of TIDL network and params binary files 
#       model_input_shape: shape of the input tensor of the model 
#       tidl_import_tool:  file name of TIDL import tool executable
#       tidl_calib_tool:   file name of TIDL calibration tool executable
# Output: 
#       "subgraph_name"_net.bin:    TIDL network binary file
#       "subgraph_name"_params.bin: TIDL parameters binary file
# Return:
#       1 if the model can run on TIDL 
#       0 if the model cannot run on TIDL
#
# Note: Input tensor shape is not available from every supported model. For 
#       example, mobileNet v2 frozen model doesn't have the shape information. 
#       Therefore, it needs to be provided by the caller. 
################################################################################
def tidl_check_model(model_file, calib_image, subgraph_name, model_input_shape,
                     tidl_import_tool, tidl_calib_tool, artifacts_folder, conv2d_kernel_type):

    print('Check if TIDL supports ' + model_file)
    raw_image = "./raw_calib_image.bin"
    
    # Check model type - can only recognize TF/ONNX/TFlite models
    if model_file.endswith('.pb'):
        print ('Importing TF model to TIDL')

        # Optimize TF model - assuming provided model is a frozen model
        opt_model_file = model_file.replace(".pb", "_opt.pb")
        result = tidl_optimize_tf_model(model_file, opt_model_file)

        if(result == 0):
            return 0, None

        model_type = MODEL_TYPE_TF
        model = opt_model_file
    
    elif model_file.endswith('.tflite'):
        print ('Importing TF lite model to TIDL')
        model_type = MODEL_TYPE_TFLITE
        model = model_file

    elif model_file.endswith('.onnx'):
        print ('Importing ONNX model to TIDL')
        model_type = MODEL_TYPE_ONNX
        model = model_file

    else:
        print ('This model is not supported by TIDL')
        return 0, None

    # Pre-process calibration image to raw data saved in binary format
    if model_file.endswith('.pb') or model_file.endswith('.tflite'):
        tidl_image_preprocess_tf(calib_image, raw_image, model_input_shape)
    else:
        tidl_image_preprocess_onnx(calib_image, raw_image, model_input_shape)

    # Try to import the model: return 1 if import succeeds and 0 if import fails
    return tidl_import(model_type, model, raw_image, model_input_shape,
                       tidl_import_tool, tidl_calib_tool, subgraph_name, 
                       artifacts_folder, conv2d_kernel_type)

################################################################################
# Function tidl_optimize_tf_model:
#   Optimize a given TF frozen model and write the optimized model into a file.
# Input: 
#       input_model:   file name of the frozen model
#       output_model:  file name of the optimized model
# Output: 
#       optimized model written to file "output_model"
################################################################################
def tidl_optimize_tf_model(input_model, output_model):
  try: 
    with tf.gfile.GFile(input_model, "rb") as f:
      input_graph_def = tf.GraphDef()
      data = f.read()
      input_graph_def.ParseFromString(data)

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(input_graph_def)

    for op in graph.get_operations(): 
      if op.type == "Placeholder":
          input_name = op.name.replace("import/","")

    for op in graph.get_operations(): 
      if op.type == "Softmax":
          output_name = op.name.replace("import/","")

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
              input_graph_def,
              [input_name], 
              [output_name],
              dtypes.float32.as_datatype_enum,
              False)

    #out_model = input_model
    f = tf.gfile.GFile(output_model, "w")
    f.write(output_graph_def.SerializeToString())
  except: 
    print("Model optimization failed!")
    return 0

  return 1

################################################################################
# Function tidl_image_preprocess_tf:
#   Preprocess image for TIDL import of TF and TF lite models:
#      - convert coded image to raw data 
#      - process raw data according to TF specification:
#        1.	Crop the image to 0.875 of original size
#        2.	Resize the cropped image to desired HeightxWidth
#        3.	Normalize the image to [-128,127] 
#      - write raw data to a file.  
################################################################################
def tidl_image_preprocess_tf(input_image, output_image, output_dim):
    print('Pre-process image for TF or TF lite')

    # read image and convert to RGB format - openCV reads image as BGR format
    image_BGR = cv2.imread(filename = input_image)  
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

    output_image_size = output_dim[0:2] # get image height and weight

    # crop and resize image 
    orig_H = image.shape[0]
    orig_W = image.shape[1]
    factor = 0.875
    crop_W = orig_W * factor
    crop_H = orig_H * factor
    half_W = orig_W/2
    half_H = orig_H/2
    start_x = half_H - crop_H/2
    start_y = half_W - crop_W/2
    end_x = start_x + crop_H
    end_y = start_y + crop_W
    x0 = round(start_x)
    x1 = round(end_x)
    y0 = round(start_y)
    y1 = round(end_y)
    cropped_image = image[x0:x1,y0:y1]
    resized_image = cv2.resize(cropped_image, output_image_size, interpolation = cv2.INTER_AREA)

    # serialize data to be written to a file
    r,g,b = cv2.split(resized_image)
    total_per_plane = output_image_size[0]*output_image_size[1];
    rr = r.reshape(1,total_per_plane)
    rg = g.reshape(1,total_per_plane)
    rb = b.reshape(1,total_per_plane)
    y = np.hstack((rr,rg,rb))

    # subtract all pixels by 128 -> convert to int8 
    mean = np.full(y.shape,128)
    y = y.astype(np.int32)
    mean = mean.astype(np.int32)
    y_sub_mean = cv2.subtract(y,mean)
    np.clip(y_sub_mean,-128,127)
    y_sub_mean.astype('int8').tofile(output_image)

################################################################################
# Function tidl_image_preprocess_onnx:
#   Preprocess image for TIDL import of ONNX modles:
#      - convert coded image to raw data with range [0, 255]
#      - write raw data to a file.  
#   TIDL will preprocess data according to ONNX specification. 
################################################################################
def tidl_image_preprocess_onnx(input_image, output_image, output_dim):
    print('Pre-process image for ONNX')

    # read image and convert to RGB format - openCV reads image as BGR format
    image_BGR = cv2.imread(filename = input_image)   
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

    output_image_size = output_dim[0:2] # get image height and weight
    resized_image = cv2.resize(image, output_image_size, interpolation = cv2.INTER_AREA)

    r,g,b = cv2.split(resized_image)
    total_per_plane = output_image_size[0]*output_image_size[1];
    rr = r.reshape(1,total_per_plane)
    rg = g.reshape(1,total_per_plane)
    rb = b.reshape(1,total_per_plane)

    y = np.hstack((rr, rg, rb))
    y.astype('uint8').tofile(output_image);


################################################################################
# Function tidl_import_tf_model:
#   Run TIDL import of the optimized TF model
# Input:
#       model_type:        type of model (TF: 1, ONNX: 2, TF lite: 3)
#       model:             file name of the model
#       raw_image:         file name of the raw image for TIDL calibration
#       input_shape:       shape of the input tensor of the model 
#       tidl_import_tool:  file name of TIDL import tool executable
#       tidl_calib_tool:   file name of TIDL calibration tool executable
#       subgraph_name:     base name of TIDL network and params binary files 
# Output: 
#       "subgraph_name"_net.bin:    TIDL network binary file
#       "subgraph_name"_params.bin: TIDL parameters binary file
# Return:
#       1 if the model can run on TIDL 
#       0 if the model cannot run on TIDL
################################################################################
def tidl_import(model_type, model, raw_image, input_shape, tidl_import_tool, 
                tidl_calib_tool, subgraph_name, artifacts_folder, conv2d_kernel_type=None):

    # TIDL net and params binary files
    tidl_net_bin_file    = "./" + artifacts_folder + "/" + subgraph_name + '_net.bin'
    tidl_params_bin_file = "./" + artifacts_folder + "/" + subgraph_name + '_params.bin'

    # Generate import config file
    import_config_file = './tidl_import_config.txt'
    with open(import_config_file,'w') as config_file:
        config_file.write("randParams         = 0\n") 
        config_file.write("modelType          = {}\n".format(model_type))
        config_file.write("quantizationStyle  = 1\n") 
        config_file.write("quantRoundAdd      = 50\n")
        config_file.write("numParamBits       = 12\n")
        config_file.write("inputNetFile       = {}\n".format(model))
        config_file.write("inputParamsFile    = NA\n")
        config_file.write("outputNetFile      = {}\n".format(tidl_net_bin_file))
        config_file.write("outputParamsFile   = {}\n".format(tidl_params_bin_file))
        config_file.write("inElementType      = 1\n")
        config_file.write("sampleInData       = {}\n".format(raw_image))
        config_file.write("tidlStatsTool      = {}\n".format(tidl_calib_tool))
        config_file.write("inWidth            = {}\n".format(input_shape[0]))
        config_file.write("inHeight           = {}\n".format(input_shape[1]))
        config_file.write("inNumChannels      = {}\n".format(input_shape[2]))
        if conv2d_kernel_type:
            config_file.write("conv2dKernelType   = {}\n".format(conv2d_kernel_type))
        if model_type == MODEL_TYPE_ONNX:
            config_file.write("rawSampleInData = 0\n")
            config_file.write("preProcType     = 256\n")
            config_file.write("inMean  = 123.675 116.28 103.53\n")
            config_file.write("inScale = 0.017125 0.017507 0.017429\n")
        else:
            config_file.write("rawSampleInData   = 1\n")

    # Run TIDL import tool
    try: 
        result = subprocess.run([tidl_import_tool, import_config_file], stdout=subprocess.PIPE)
        console_out = result.stdout.decode('utf-8')
    except:
        print("TIDL import crashed")
        return 0, None
  
    # Check TIDL import result
    if console_out.find('error')==-1 and console_out.find('ERROR')==-1:
        print("TIDL import succeeded")
        search_for_last_node_dim = os.popen("lastnode=`ls -1 ./tempDir/trace_dump*.y | cut -d'_' -f3 | sort -n | tail -1`; ls -1 ./tempDir/trace_dump_${lastnode}_*.y | cut -d'_' -f4 | cut -d'.' -f1")
        last_node_dim = search_for_last_node_dim.read().rstrip()
        return 1, last_node_dim
    else:
        print("TIDL import failed")
        return 0, None
