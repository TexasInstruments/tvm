/*
 *
 * Copyright (C) 2019 Texas Instruments Incorporated - http://www.ti.com/
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the  
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/
#ifndef TIDL_IMPORT_H
#define TIDL_IMPORT_H

#include <stdint.h>
//#include "dlpack.h"
//#include "itidl_ti.h"

#define TIDL_IMPORT_SUCCESS           1
#define TIDL_IMPORT_FAILURE           0

/*
 * TIDL import configuration parameters for network conversion
 *
 * Definition is temporary and will be finalized when Relay IR import is done.
 *   - May only need a subset of parameters listed below.
*/
typedef struct tidlImpConfig  
{
  int32_t  numParamBits;    /* Number of bits used to quantize the parameters (weights).
                               It can take values from 4 to 12. Default is 8. */
  int32_t  quantRoundAdd;   /* quantRoundAdd/100 will be added when rounding a 
                               floating point number to integer. It can take any 
                               value from 0 to 100. Default is 50.            */
//  int32_t  inQuantFactor;   /* Input quantization factor. This parameter will be 
//                               removed as it is hard coded for each framework 
//                               (Caffe/TF/ONNX).                               */
  int32_t  inElementType;   /* Flag to indicate whether input is signed or unsigned: 
                               - 0: input is 8-bit unsigned 
                               - 1: input is 8-bit signed 
                               Default is 1.                                  */
  int32_t  inNumChannels;   /* Number of channels of input data.
                               Default is -1. User must set it. Otherwise, error
                               will be returned.                              */
  int32_t  inHeight;        /* Height of input data.
                               Default is -1. User must set it. Otherwise, error
                               will be returned.                              */
  int32_t  inWidth;         /* Width of input data.
                               Default is -1. User must set it. Otherwise, error
                               will be returned.                              */
} tidlImpConfig;

/*
 * TIDL import configuration parameters for calibration 
 *
 * Definition is temporary and will be finalized when Relay IR import is done.
 *   - May only need a subset of parameters listed below.
*/
typedef struct{
  int32_t  rawSampleInData; /* Flag to indicate the type of input data:
                               - 0: input is encoded and to be preprocessed according to preProcType
                               - 1: input is RAW data and preProcType is ignored.
                               Default is 0.                                  */
  int32_t  preProcType;     /* Type of preprocessing needed for input data. 
                               Default is 0.
Preprocessing according to rawSampleInData and preProcType is explained below:
rawSampleInData   preProcType              image pre-processing
     0               0          1. Resize the original image (WxH) to (256x256) with scale factors (0,0) and INTER_AREA using OpenCV function resize().
                                2. Crop the resized image to ROI (128-W/2, 128-H/2, W, H) defined by cv::Rect.
     0               1          Resize and crop as preProcType 0, and then subtract pixels by (104, 117, 123) per plane.
     0               2          1. Change color space from BGR to RGB for the original image (WxH).
                                2. Crop new image to ROI (H/16, W/16, 7H/8, 7W/8) defined by cv::Rect.
                                3. Resize the cropped image to (WxH) with scale factors (0,0) and INTER_AREA using OpenCV function resize().
                                4. Subtract pixels by (104, 117, 123) per plane.
     0               3          1. Change color space from BGR to RGB for the original image (WxH).
                                2. Resize the original image (WxH) to (32x32) with scale factors (0,0) and INTER_AREA using OpenCV function resize().
                                3. Crop the resized image to ROI (16-W/2, 16-H/2, W, H) defined by cv::Rect.
     0               4          No pre-processing is performed on the original image.
     0               5          1. Change color space from BGR to RGB for the original image (WxH).
                                2. Crop new image to ROI (0, 0, H, W) defined by cv::Rect.
                                3. Resize the cropped image to (WxH) with scale factors (0,0) and INTER_AREA using OpenCV function resize().
                                4. Subtract pixels by (128, 128, 128) per plane.
     0               6          Normalize the original image in the range of [0, 255] (ONNX preprocessing):
                                Subtract pixels by (123.68 116.28, 103.53) per plane.
                                Divide pixels by (58.395, 57.12, 57.375) per plane.
     0               7-255      Configuration error. No pre-processing to be done.
     0               256        Take inMean and inScale from config file and do the normalization on RAW image:
                                Subtract pixels by (inMean[0], inMean[1], inMean[2]) per plane.
                                Multiply pixels by (inScale[0], inScale[1], inScale[2]) per plane.
     0               >256       Configuration error. No pre-processing to be done.
     1               N/A        Raw image. No pre-processing to be done, and preProcType is ignored.

                               */
  int32_t  numSampleInData; /* Number of frames in input data. 
                               Default is 1.                                  */
}tidlCalibConfig;

/* 
 * Function tidlImpConvertRelayIr()
 *   TIDL import library function to convert Relay IR to TIDL internal 
 *   representation. 
 *
 * Input parameters: 
 *       relayIrAst    - Relay IR AST data structure
 *       config        - TIDL import configuration 
 *
 * Output parameters: 
 *       tidlNetFile   - output file name for the network topology in TIDL format
 *       tidlPramsFile - output file name for the network weights in TIDL format
*/
extern int tidlImpConvertRelayIr(void *relayIrAst, tidlImpConfig *config,
                          char *tidlNetFile, char *tidlParamsFile);

/*
 * Function tidlImpCalibRelayIr()
 *   TIDL import library function to calibrate dynamic quantization for a converted 
 *   TIDL internal representation.
 *
 * Input parameters: 
 *       relayIrInTensor  - input tensor of TIDL subgraph
 *       tidlCalibTool    - TIDL calibration executable file name 
 *       config           - TIDL import calibration configuration 
 *       tidlNetFile      - file name for the converted TIDL network
 *       tidlParamsFile   - file name for the weights in converted TIDL network
 * Output parameters:     
 *       tidlNetFile      - file name for TIDL network after calibration
*/
int tidlImpCalibRelayIr(void *relayIrInTensor, char *tidlCalibTool, 
                        tidlCalibConfig *config,
                        char *tidlNetFile, char *tidlParamsFile);
#endif
