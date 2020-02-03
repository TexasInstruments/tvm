/*
 *
 * Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
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


/**
----------------------------------------------------------------------------
@file    itidl_ti.h
@brief     This file defines the ivision interface for TI deep learning libary
@version 0.1 (Oct 2016) : Initial Code [ADK]
@version 0.5 (Jan 2017) : Cleaned up [ADK]
----------------------------------------------------------------------------
*/

/** @ingroup    iTI_DL */
/*@{*/
#ifndef ITIDL_H
#define ITIDL_H

#define TIDL_IMPORT_TOOL

#ifdef  TIDL_IMPORT_TOOL 
#define USE_IVISION (0)
#else
#define USE_IVISION (1)
#endif

#if USE_IVISION
#include <ivision.h>
#include <ti/xdais/ires.h>
#include <stdint.h>


/**
* @brief The unique const funtion table for the TI deep learning algorithm.
*/
extern const IVISION_Fxns TIDL_VISION_FXNS;

/**
* @brief IRES interface of the TI deep learning algorithm.
*/
extern const IRES_Fxns TIDL_IRES;

#endif 

#define TIDL_NUM_MAX_LAYERS     (256)
#define TIDL_NUM_IN_BUFS        (16)
#define TIDL_NUM_OUT_BUFS       (16)
#define TIDL_STRING_SIZE        (256)
#define TIDL_MAX_PAD_SIZE       (4)
/* Fix for JIRA issue TIDL - 303 */
#define TIDL_MAX_DATA_BUFS      (256)
#define TIDL_MAX_ALG_IN_BUFS    (16)
#define TIDL_MAX_ALG_OUT_BUFS   (16)

typedef float float32_tidl;

/**
 @enum    eTIDL_LayerType
 @brief   This enumerator defines the different types of layers used 
          in deeep learing algorithms
*/
typedef enum
{
  TIDL_DataLayer             = 0,
  TIDL_ConvolutionLayer      = 1,
  TIDL_PoolingLayer          = 2,
  TIDL_ReLULayer             = 3,
  TIDL_PReLULayer            = 4,
  TIDL_EltWiseLayer          = 5,
  TIDL_InnerProductLayer     = 6,
  TIDL_SoftMaxLayer          = 7,
  TIDL_BatchNormLayer        = 8,
  TIDL_BiasLayer             = 9,
  TIDL_ScaleLayer            = 10,
  TIDL_Deconv2DLayer         = 11,
  TIDL_ConcatLayer           = 12,
  TIDL_SplitLayer            = 13,
  TIDL_SliceLayer            = 14,
  TIDL_CropLayer             = 15,
  TIDL_FlattenLayer          = 16,
  TIDL_DropOutLayer          = 17,
  TIDL_ArgMaxLayer           = 18,
  TIDL_DetectionOutputLayer  = 19,
  TIDL_UnSuportedLayer       = 20
}eTIDL_LayerType;

/**
 @enum    eTIDL_ElementType
 @brief   This enumerator defines the different types of element type
          used by TIDL library
*/
typedef enum
{
  TIDL_UnsignedChar          = 0,
  TIDL_SignedChar            = 1,
  TIDL_UnsignedShort         = 2,
  TIDL_SignedShort           = 3,
  TIDL_UnsignedWord          = 4,     
  TIDL_SignedWord            = 5       
}eTIDL_ElementType;

/**
 @enum    eTIDL_quantStyle
 @brief   This enumerator defines the different types of quantization Styles
          supported by TIDL library
*/
typedef enum
{
  TIDL_quantStyleFixed   = 0, /*!< In this case, TIDL expects the Quantization 
                                   information via interface. This is 
                                   supported by caffe-jacinto training framwork
                                   - Currently not suported for user */
  TIDL_quantStyleDynamic = 1, /*!< IDynamic quantization does not depend  
                                   on training.Quantization of kernel 
                                   weights happen during model import 
                                   using tool */
  TIDL_quantStyleCustom  = 10 /*!< Currently not suported for user */
}eTIDL_quantStyle;

/**
 @enum    eTIDL_optimiseExtMem
 @brief   This enumerator defines the different types of optimizations 
          supported by TIDL library
*/
typedef enum
{
  TIDL_optimiseExtMemL0 = 0, /*!< In this case, no optmization in output 
                                    memory, output buffer will not re-use 
                                    memory from previous layers, So, 
                                    algorithm uses more external memory.
                                    This is Currently used for debugging */
  TIDL_optimiseExtMemL1 = 1, /*!< In this case, output buffer will  
                                    re-use memory from previous layers when
                                    they have same width and height,  
                                    So, algorithm uses less external memory 
                                    compare to L0 case  */
  TIDL_optimiseExtMemL2 = 2  /*!< In this case, output buffer will  
                                    re-use memory from previous layers when
                                    they have different width and height,  
                                    So, algorithm uses less external memory 
                                    compare to L1 case  */
}eTIDL_optimiseExtMem;

/**
 @enum    eTIDL_kernelType
 @brief   This enumerator defines the different types of optimizations 
          in kernel types supported by TIDL library
*/
typedef enum
{
  TIDL_sparse = 0,          /*!< In this case, only non zero coeffs are 
                                   considered for convolution    */
  TIDL_dense  = 1           /*!< In this case, all coeffs in the kernel
                                   considered for convolution    */
}eTIDL_kernelType;

/**
 @enum    eTIDL_strideOffsetMethod
 @brief   This enumerator defines the different logic to choose offset for 
          kernels/windows with stride grater than 1
*/
typedef enum
{
  TIDL_strideOffsetTopLeft   = 0,
  TIDL_strideOffsetCenter    = 1,
  TIDL_strideOffsetMax       = 2
}eTIDL_strideOffsetMethod;

/**
 @enum    eTIDL_PoolType
 @brief   This enumerator defines the different types of spatial pooling
          supported by TIDL library
*/
typedef enum
{
  TIDL_MaxPooling            = 0,
  TIDL_AveragePooling        = 1
}eTIDL_PoolType;

/**
 @enum    eTIDL_reluType
 @brief   This enumerator defines the different RelU types 
          supported by TIDL library
*/
typedef enum
{
  TIDL_RelU         = 0,
  TIDL_PRelU        = 1,
  TIDL_RelU6        = 2
}eTIDL_reluType;

/**
 @enum    eTIDL_EltWiseType
 @brief   This enumerator defines the different types of eltWise layer
          operations. Currently only TIDL_EltWiseSum is supported by TIDL 
          library.
*/
typedef enum
{
    TIDL_EltWiseProduct        = 0,
    TIDL_EltWiseSum            = 1,
    TIDL_EltWiseMax            = 2    
}eTIDL_EltWiseType;

/**
 @enum    eTIDL_DataDimIndex
 @brief   This enumerator defines the indices of dimension array of layer data
          buffer in TIDL library
*/
typedef enum
{
  TIDL_DIM_BATCH          = 0,
  TIDL_DIM_NUMCH          = 1,
  TIDL_DIM_HEIGHT         = 2,
  TIDL_DIM_WIDTH          = 3,
  TIDL_DIM_MAX            = 4       
}eTIDL_DataDimIndex;

/**
 @enum    eTIDL_PitchDimIndex
 @brief   This enumerator defines the indices of picth array of layer data
          buffer in TIDL library
*/
typedef enum
{
  TIDL_ROI_PITCH         = 0,
  TIDL_CHANNEL_PITCH     = 1,
  TIDL_LINE_PITCH        = 2,
  TIDL_PITCH_MAX         = (TIDL_DIM_MAX - 1)
}eTIDL_PitchDimIndex;

/**
 @enum    eTIDL_ErrorType
 @brief   This enumerator defines error numbers that have to be set when layer
          parameters deviate from expected range of values.
*/
typedef enum
{
    /* Convolution layer error types */
    TIDL_E_CONVOLUTION                         = -1000,
    TIDL_E_CONV_INVALID_INPUT_WIDTH            = (TIDL_E_CONVOLUTION - 0),
    TIDL_E_CONV_INVALID_INPUT_HEIGHT           = (TIDL_E_CONVOLUTION - 1),
    TIDL_E_CONV_INVALID_OUTPUT_WIDTH           = (TIDL_E_CONVOLUTION - 2),
    TIDL_E_CONV_INVALID_OUTPUT_HEIGHT          = (TIDL_E_CONVOLUTION - 3),
    TIDL_E_CONV_INVALID_NUM_IN_CHANNELS        = (TIDL_E_CONVOLUTION - 4),
    TIDL_E_CONV_INVALID_NUM_OUT_CHANNELS       = (TIDL_E_CONVOLUTION - 5),
    TIDL_E_CONV_INVALID_KER_WIDTH              = (TIDL_E_CONVOLUTION - 6),
    TIDL_E_CONV_INVALID_KER_HEIGHT             = (TIDL_E_CONVOLUTION - 7),
    TIDL_E_CONV_INVALID_KER_TYPE               = (TIDL_E_CONVOLUTION - 8),
    TIDL_E_CONV_INVALID_STRIDE_WIDTH           = (TIDL_E_CONVOLUTION - 9),
    TIDL_E_CONV_INVALID_STRIDE_HEIGHT          = (TIDL_E_CONVOLUTION - 10),
    TIDL_E_CONV_NEGATIVE_OUTPUT_SHIFT          = (TIDL_E_CONVOLUTION - 11),

    /* Convolution pooling error types */
    TIDL_E_CONV_POOL_INVALID_POOL_TYPE         = (TIDL_E_CONVOLUTION - 12),
    TIDL_E_CONV_POOL_INVALID_KER_WIDTH         = (TIDL_E_CONVOLUTION - 13),
    TIDL_E_CONV_POOL_INVALID_KER_HEIGHT        = (TIDL_E_CONVOLUTION - 14),
    TIDL_E_CONV_POOL_INVALID_STRIDE_WIDTH      = (TIDL_E_CONVOLUTION - 15),
    TIDL_E_CONV_POOL_INVALID_STRIDE_HEIGHT     = (TIDL_E_CONVOLUTION - 16),

    /* Deconvolution layer error types */
    TIDL_E_DECONV_INVALID_INPUT_WIDTH          = (TIDL_E_CONVOLUTION - 17),
    TIDL_E_DECONV_INVALID_INPUT_HEIGHT         = (TIDL_E_CONVOLUTION - 18),
    TIDL_E_DECONV_INVALID_OUTPUT_WIDTH         = (TIDL_E_CONVOLUTION - 19),
    TIDL_E_DECONV_INVALID_OUTPUT_HEIGHT        = (TIDL_E_CONVOLUTION - 20),
    TIDL_E_DECONV_INVALID_NUM_IN_CHANNELS      = (TIDL_E_CONVOLUTION - 21),
    TIDL_E_DECONV_INVALID_NUM_OUT_CHANNELS     = (TIDL_E_CONVOLUTION - 22),
    TIDL_E_DECONV_INVALID_NUM_GROUPS           = (TIDL_E_CONVOLUTION - 23),
    TIDL_E_DECONV_INVALID_STRIDE               = (TIDL_E_CONVOLUTION - 24),

    /* Eltwise layer error types */
    TIDL_E_ELTWISE                             = -1030,
    TIDL_E_ELTWISE_INVALID_INPUT_WIDTH         = (TIDL_E_ELTWISE - 0),
    TIDL_E_ELTWISE_INVALID_INPUT_HEIGHT        = (TIDL_E_ELTWISE - 1),
    TIDL_E_ELTWISE_INVALID_OUTPUT_WIDTH        = (TIDL_E_ELTWISE - 2),
    TIDL_E_ELTWISE_INVALID_OUTPUT_HEIGHT       = (TIDL_E_ELTWISE - 3),
    TIDL_E_ELTWISE_INVALID_ELTWISE_TYPE        = (TIDL_E_ELTWISE - 4),
    TIDL_E_ELTWISE_INVALID_NUM_CHANNELS        = (TIDL_E_ELTWISE - 5),
    TIDL_E_ELTWISE_INVALID_INPUT_TYPE          = (TIDL_E_ELTWISE - 6),

    /* Pooling error types */
    TIDL_E_POOLING                             = -1040,
    TIDL_E_POOL_INVALID_INPUT_WIDTH            = (TIDL_E_POOLING - 0),
    TIDL_E_POOL_INVALID_INPUT_HEIGHT           = (TIDL_E_POOLING - 1),
    TIDL_E_POOL_INVALID_OUTPUT_WIDTH           = (TIDL_E_POOLING - 2),
    TIDL_E_POOL_INVALID_OUTPUT_HEIGHT          = (TIDL_E_POOLING - 3),
    TIDL_E_POOL_INVALID_POOL_TYPE              = (TIDL_E_POOLING - 4),
    TIDL_E_POOL_INVALID_NUM_CHANNELS           = (TIDL_E_POOLING - 5),
    TIDL_E_POOL_INVALID_KER_WIDTH              = (TIDL_E_POOLING - 6),
    TIDL_E_POOL_INVALID_KER_HEIGHT             = (TIDL_E_POOLING - 7),
    TIDL_E_POOL_INVALID_STRIDE_WIDTH           = (TIDL_E_POOLING - 8),
    TIDL_E_POOL_INVALID_STRIDE_HEIGHT          = (TIDL_E_POOLING - 9),
    TIDL_E_POOL_INVALID_INPUT_SIZE             = (TIDL_E_POOLING - 10),

    /* Inner product error types */
    TIDL_E_INNER_PRODUCT                       = -1060,
    TIDL_E_IP_INVALID_NUM_IN_NODES             = (TIDL_E_INNER_PRODUCT - 0),
    TIDL_E_IP_INVALID_NUM_OUT_NODES            = (TIDL_E_INNER_PRODUCT - 1),
    TIDL_E_IP_NEGATIVE_OUTPUT_SHIFT            = (TIDL_E_INNER_PRODUCT - 2),

    /* Argmax error types */
    TIDL_E_ARGMAX                              = -1070,
    TIDL_E_ARGMAX_INVALID_NUM_CHANNELS         = (TIDL_E_ARGMAX - 0),

    /* Bias error types */
    TIDL_E_BN                                 = -1080,
    TIDL_E_BN_INVALID_INPUT_WIDTH             = (TIDL_E_BN - 0),
    TIDL_E_BN_INVALID_INPUT_HEIGHT            = (TIDL_E_BN - 1),
    TIDL_E_BN_INVALID_OUTPUT_WIDTH            = (TIDL_E_BN - 2),
    TIDL_E_BN_INVALID_OUTPUT_HEIGHT           = (TIDL_E_BN - 3),
    TIDL_E_BN_INVALID_NUM_CHANNELS            = (TIDL_E_BN - 4),
    TIDL_E_BN_INVALID_ENABLE_RELU             = (TIDL_E_BN - 5),
    TIDL_E_BN_NEGATIVE_OUTPUT_SHIFT           = (TIDL_E_BN - 6),

    /* Crop layer error types */
    TIDL_E_CROP                               = -1090,
    TIDL_E_CROP_INVALID_INPUT_WIDTH           = (TIDL_E_CROP - 0),
    TIDL_E_CROP_INVALID_INPUT_HEIGHT          = (TIDL_E_CROP - 1),
    TIDL_E_CROP_INVALID_OUTPUT_WIDTH          = (TIDL_E_CROP - 2),
    TIDL_E_CROP_INVALID_OUTPUT_HEIGHT         = (TIDL_E_CROP - 3),
    TIDL_E_CROP_INVALID_NUM_CHANNELS          = (TIDL_E_CROP - 4),
    TIDL_E_CROP_INVALID_OFFSET_WIDTH          = (TIDL_E_CROP - 5),
    TIDL_E_CROP_INVALID_OFFSET_HEIGHT         = (TIDL_E_CROP - 6),

    /* Flatten layer error types */
    TIDL_E_FLATTEN                            = -1100,
    TIDL_E_FLATTEN_INVALID_INPUT_WIDTH        = (TIDL_E_FLATTEN - 0),
    TIDL_E_FLATTEN_INVALID_INPUT_HEIGHT       = (TIDL_E_FLATTEN - 1),
    TIDL_E_FLATTEN_INVALID_OUTPUT_WIDTH       = (TIDL_E_FLATTEN - 2),
    TIDL_E_FLATTEN_INVALID_OUTPUT_HEIGHT      = (TIDL_E_FLATTEN - 3),

    /* SoftMax error types */
    TIDL_E_SOFTMAX                             = -1110,
    TIDL_E_SOFTMAX_INVALID_NUM_CHANNELS        = (TIDL_E_SOFTMAX - 0),

    /* Error types common to layers */
    TIDL_E_COMMON                              = -1120,
    TIDL_E_UNSUPPORTED_LAYER                   = (TIDL_E_COMMON - 0),

    TIDL_E_MAX                                 = -1130
}eTIDL_ErrorType;

/**
 @struct  sBuffer_t
 @brief   This structure define the parmeters of data or kerner buffer
           memeory in TIDL
 @param  ptr
          For 32-bit platform, ptr is the address pointing to the actual buffer.
          For 64-bit platform, ptr can be either the index to buffer of pointers,  
          or the lower 32-bit of the address if upper 32-bit is the same for all
          pointers. 
 @remark This is to make sure the code is portable across 32-bit and 64-bit platforms. 
         
 @param  bufSize
          Size of the buffer in bytes
*/
typedef struct 
{
#ifdef PLATFORM_64BIT 
  uint32_t ptr;
#else
  void* ptr;
#endif
  int32_t bufSize;
  int32_t reserved[2];
}sBuffer_t;

/**
 @struct  sTIDL_DataParams_t
 @brief   This structure define the parmeters of data or kerner buffer
          used by TIDL layers (In,Out)
 @param  dataId
          Address pointing to the actual buffer
 @param  elementType
          Size of the buffer in bytes
 @param  numDim
          Address pointing to the actual buffer
 @param  dataQ
          Number of bits for fractional part if Quant Style is 1
          Q factor if Quant Style is 2
 @param  minValue
          Minimum value of 32-bit accumulator for all the values in 
					that layer	
 @param  maxValue
          Maximum value of 32-bit accumulator for all the values in 
					that layer	
 @param  pitch
          Pitch for each dimention
 @param  dimValues
          Size of the buffer in bytes

*/
typedef struct {
  int32_t dataId;
  int32_t elementType;    
  int32_t numDim;
  int32_t dataQ;
  int32_t minValue;
  int32_t maxValue;
  int32_t pitch[TIDL_DIM_MAX-1];
  int32_t dimValues[TIDL_DIM_MAX];
}sTIDL_DataParams_t;



/**
 @struct  sTIDL_DataLayerParams_t
 @brief   This structure define the parmeters of Data layer
          in TIDL
 @param  numChannels
          Number of channel in the In or Out data buffer
 @param  dataQ
          Q value of the data
*/
typedef struct {
  int32_t   numChannels;
  int32_t   dataQ;
}sTIDL_DataLayerParams_t;


/**
 @struct  sTIDL_ReLUParams_t
 @brief   This structure define the parmeters of ReLU activation layer
           in TIDL
 @param  slope
          Buffer containing Slope vales for PRelU           
 @param  numChannels
          Number of channels channels to be processed
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
 @param  slopeQ
          Q value of slope values for PRelU
 @param  zeroSlopeValue
          value of slope added for dynamic quantSytle                     
 @param  reluType
          value indicates different types of ReLU supported                     
*/
typedef struct {
  sBuffer_t slope;
  int32_t   numChannels;
  int32_t   inDataQ;
  int32_t   outDataQ;
  int32_t   slopeQ;
  int32_t   zeroSlopeValue;  
  int32_t   reluType;
}sTIDL_ReLUParams_t;

/**
 @struct  sTIDL_ArgMaxParams_t
 @brief   This structure define the parmeters Arg max layer
           in TIDL
 @param  numChannels
          Number of channels channels to be processed
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
*/
typedef struct {
  int32_t   numChannels;
  int32_t   inDataQ;
  int32_t   outDataQ;
}sTIDL_ArgMaxParams_t;

/**
 @struct  sTIDL_PoolingParams_t
 @brief   This structure define the parmeters spatial Pooling layer
           in TIDL
 @param  numChannels
          Number of channels channels to be processed
 @param  poolingType
          Type of the Pooling as defined in @sa eTIDL_PoolType
 @param  kernelW
          Kernel width
 @param  kernelH
          Kernel height
 @param  strideW
          Stride in horizontal direction
 @param  strideH
          Stride in vertical direction
 @param  padW
          Horizontal Padding requirement in number of elements
 @param  padH
          Vertical Padding requirement in number of elements
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
*/
typedef struct {
  int32_t   numChannels;
  int32_t   poolingType;
  int32_t   kernelW;
  int32_t   kernelH;
  int32_t   strideW;
  int32_t   strideH;
  int32_t   padW;
  int32_t   padH;
  int32_t   inDataQ;
  int32_t   outDataQ;
}sTIDL_PoolingParams_t;

/**
 @struct  sTIDL_ConvParams_t
 @brief   This structure define the parmeters Convoltuion Layer
           in TIDL
 @param  weights
          Buffer containing the kernel parameters
 @param  Bias
          Buffer containing the Bias parameters
 @param  convolutionType
          Type of the convolution, Reserved for future use
 @param  numInChannels
          Number of input channels channels to be processed
 @param  numOutChannels
          Number of output channels to be processed
 @param  numGroups
          Number of groups in the convolutions
 @param  kernelW
          Kernel width
 @param  kernelH. 
          Kernel height
 @param  strideW
          Stride in horizontal direction
 @param  strideH
          Stride in vertical direction
 @param  dilationW
          Dialation in horizontal direction
 @param  dilationH
          Dialation in vertical direction
 @param  padW
          Horizontal Padding requirement in number of elements
 @param  padH
          Vertical Padding requirement in number of elements
 @param  weightsQ
          Q value of Kernel weights
 @param  zeroWeightValue
          value of weights added for dynamic quantSytle 
 @param  biasQ
          Q value kernel Bias
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
 @param  interDataQ
          Q value intermediate ouput data
 @param  enableBias
          Enable/Disable ouput bias 
 @param  enablePooling
          Enable/Disable 2x2 Spatial pooling
 @param  enableRelU
          Enable/Disable relU activation part of convolution
 @param  kernelType
          Defines the different types of optimizations 
          in kernel types supported by TIDL 
 @param  poolParams
          Used only if enablePooling is true
 @param  reluParams
          Used only if enableRelU is true
*/
typedef struct {
  sBuffer_t weights;
  sBuffer_t bias;
  int32_t   convolutionType;
  int32_t   numInChannels;
  int32_t   numOutChannels;
  int32_t   numGroups;
  int32_t   kernelW;
  int32_t   kernelH;
  int32_t   strideW;
  int32_t   strideH;
  int32_t   dilationW;
  int32_t   dilationH;
  int32_t   padW;
  int32_t   padH;
  int32_t   weightsQ;
  int32_t   zeroWeightValue;
  int32_t   biasQ;
  int32_t   inDataQ;
  int32_t   outDataQ;
  int32_t   interDataQ;
  int32_t   enableBias;
  int32_t   enablePooling;
  int32_t   enableRelU;
  int32_t   kernelType;  
  sTIDL_PoolingParams_t poolParams;
  sTIDL_ReLUParams_t    reluParams;
}sTIDL_ConvParams_t;

/**
 @struct  sTIDL_DetectOutputParams_t
 @brief   This structure define the parmeters of Detection Output Layer
           in TIDL
 @param  priorBox
          Buffer containing the data required to form prior Bboxs
 @param  priorBoxSize
          Siz of the priorBox buffer required to form prior Bboxs 
 @param  numClasses
          number of classes to be detected in the detection Output
 @param  backgroundLabelId
          To indicate whether or not to ignore background class
 @param  codeType
          Indicates the coding type to be used for decoding Bboxs
 @param  confThreshold
          Value to indicates threshold above which objects to be 
          considered for detection
 @param  nmsThreshold
          Threshold Value used for finding overlap between the 
          bboxs in the NMS
 @param  eta. 
          Value used to update the adaptive Threshold in the NMS 
 @param  topK
          Number of top k objects to keep for class after applying NMS
 @param  keepTopK
          Number of top k objects to Keep in the final output
 @param  shareLocation
          Indicate whether same size Boxes used for all classes or not,  
          it is not supported in ti_dl 
 @param  varianceEncoded
          Flag to indicate the variance used in decoding bboxes is 
          encoded along with locations are not
*/
typedef struct {
  sBuffer_t priorBox;  
  int32_t  priorBoxSize;  
  int32_t  numClasses; 
  int32_t  backgroundLabelId;
  int32_t  codeType;
  float32_tidl  confThreshold;
  float32_tidl  nmsThreshold;
  float32_tidl  eta;
  int32_t  topK;
  int32_t  keepTopK;
  int32_t  shareLocation;
  int32_t  varianceEncoded;  
}sTIDL_DetectOutputParams_t;


/**
 @struct  sTIDL_ConcatParams_t
 @brief   This structure define the parmeters of PriorBox layer
           in TIDL
 @param  priorBox
          Buffer containing the priorBox parameters and variance
*/
typedef struct {
  int32_t  axis;
  int32_t  outDataQ;  
}sTIDL_ConcatParams_t;


/**
 @struct  sTIDL_BatchNormParams_t
 @brief   This structure define the parmeters of Batch Norm layer
           in TIDL
 @param  weights
          Buffer containing the kernel parameters           
 @param  Bias
          Buffer containing the Bias parameters
 @param  numChannels
          Number of channels channels to be processed
 @param  biasQ
          Q value of Bias parameter
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
 @param  weightsQ
          Q value of Kernel weights
 @param  zeroWeightValue
          value of weights added for dynamic quantSytle           
 @param  enableRelU
          Enable/Disable relU on the output data
 @param  reluParams
          Used only if enableRelU is true          
*/
typedef struct {
  sBuffer_t weights;
  sBuffer_t bias;
  int32_t   numChannels;
  int32_t   biasQ;
  int32_t   inDataQ;
  int32_t   outDataQ;
  int32_t   weightsQ;
  int32_t   zeroWeightValue;  
  int32_t   enableRelU;
  sTIDL_ReLUParams_t    reluParams;  
}sTIDL_BatchNormParams_t;

/**
 @struct  sTIDL_BiasParams_t
 @brief   This structure define the parmeters of Bias layer
           in TIDL
 @param  Bias
          Buffer containing the Bias parameters
 @param  numChannels
          Number of channels channels to be processed
 @param  biasQ
          Q value of Bias parameter
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
*/
typedef struct {
  sBuffer_t bias;
  int32_t   numChannels;
  int32_t   biasQ;
  int32_t   inDataQ;
  int32_t   outDataQ;
}sTIDL_BiasParams_t;



/**
 @struct  sTIDL_InnerProductParams_t
 @brief   This structure define the parmeters of Inner Product 
          (Fully connected) layer in TIDL
 @param  weights
          Buffer containing the kernel parameters
 @param  Bias
          Buffer containing the Bias parameters
 @param  numInNodes
          Number of elememnts in the flattend input
 @param  numOutNodes
          Number of elememnts in the output
 @param  weightsQ
          Q value of Kernel weights
 @param  zeroWeightValue
          value of weights added for dynamic quantSytle 
 @param  biasQ
          Q value kernel Bias
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
 @param  interDataQ
          Q value intermediate ouput data
 @param  enableRelU
          Enable/Disable relU activation part of convolution
 @param  reluParams
          Used only if enableRelU is true
*/
typedef struct {
  sBuffer_t weights;
  sBuffer_t bias;
  int32_t   activationType;
  int32_t   numInNodes;
  int32_t   numOutNodes;
  int32_t   weightsQ;
  int32_t   zeroWeightValue;
  int32_t   biasQ;
  int32_t   inDataQ;
  int32_t   outDataQ;
  int32_t   interDataQ;
  int32_t   enableRelU;
  sTIDL_ReLUParams_t    reluParams;
}sTIDL_InnerProductParams_t;

/**
 @struct  sTIDL_EltWiseParams_t
 @brief   This structure define the parmeters of Elementwise layer
           in TIDL
 @param Bias
         Buffer containing the Bias parameters
 @param numChannels
         Number of channels channels to be processed
 @param eltWiseType
         Type of the element wise opration. Currely only Add op is supported
 @param numInData
         Number of input data buffers on which  Elementwise operation 
         shall ne performed
 @param bufSize
          size of the buffer in bytes
 @param  biasQ
          Q value kernel Bias
 @param  inDataQ
          Q value of the in data buffers
 @param  outDataQ
          Q value expected for out data
 @param  enableRelU
          Enable/Disable relU activation part of convolution
 @param  reluParams
          Used only if enableRelU is true
*/
typedef struct {
  sBuffer_t bias;
  int32_t   numChannels;
  int32_t   eltWiseType;
  int32_t   numInData;
  int32_t   biasQ;
  int32_t   inDataQ[TIDL_NUM_IN_BUFS];
  int32_t   outDataQ;
  int32_t   enableRelU;
  sTIDL_ReLUParams_t    reluParams;
}sTIDL_EltWiseParams_t;

/**
 @struct  sTIDL_SoftMaxParams_t
 @brief   This structure define the parmeters Soft max layer
           in TIDL
 @param  numChannels
          Number of channels channels to be processed
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
*/
typedef struct {
  int32_t   numChannels;
  int32_t   inDataQ;
  int32_t   outDataQ;
}sTIDL_SoftMaxParams_t;

/**
 @struct  sTIDL_CropParams_t
 @brief   This structure define the parmeters Crop layer
           in TIDL
 @param  numChannels
          Number of channels channels to be processed
 @param  inDataQ
          Q value of the in data
 @param  outDataQ
          Q value expected for out data
*/
typedef struct {
  int32_t   numChannels;
  int32_t   inDataQ;
  int32_t   outDataQ;
	int32_t		offsetW;
	int32_t		offsetH;
}sTIDL_CropParams_t;

/**
 @struct  sTIDL_LayerParams_t
 @brief   This union define the layer specific parmeters of all the 
          suported layers in TIDL
*/
/* CHECK_MISRA("-18.4")  -> Disable rule 18.4  */
typedef union {
  sTIDL_ConvParams_t              convParams;
  sTIDL_ReLUParams_t              reluParams;
  sTIDL_EltWiseParams_t           eltWiseParams;
  sTIDL_PoolingParams_t           poolParams;
  sTIDL_InnerProductParams_t      innerProductParams;
  sTIDL_DataLayerParams_t         dataLayerParams;
  sTIDL_ArgMaxParams_t            argMaxParams;
  sTIDL_SoftMaxParams_t           softMaxParams;
  sTIDL_CropParams_t              cropParams;
  sTIDL_ConcatParams_t            concatParams; 
  sTIDL_DetectOutputParams_t      detectOutParams;
  sTIDL_BiasParams_t              biasParams;   
  sTIDL_BatchNormParams_t         batchNormParams;   
}sTIDL_LayerParams_t;
/*RESET_MISRA("18.4")  -> Reset rule 18.4 */

/**
 @struct  sTIDL_Layer_t
 @brief   This structure define the common layer parmeters
           in TIDL
 @param  layerType
          Layer Type
 @param  numInBufs
          Number of input data buffers for the layer
 @param  numOutBufs
          Number of output data buffers for the layer
 @param  inData
          Input data buffers details
 @param  outData
          output data buffers details
 @param  coreID
          Processing core ID (EVE or DSP)
 @param  layersGroupId
          Group of layers in the net are processed together. This unique number
          identify the currently processing group
 @param  weightsElementSizeInBits
          Size of compute layer weight parameters in bytes
 @param  strideOffsetMethod
          Offset selection method for stride. Refer eTIDL_strideOffsetMethod
*/
typedef struct {
  sTIDL_LayerParams_t layerParams;
  int32_t layerType;
  int32_t numInBufs;
  int32_t numOutBufs;
  sTIDL_DataParams_t inData[TIDL_NUM_IN_BUFS]; 
  sTIDL_DataParams_t outData[TIDL_NUM_OUT_BUFS];
  int32_t coreID;
  int32_t layersGroupId; 
  int32_t weightsElementSizeInBits;
  int32_t strideOffsetMethod;
}sTIDL_Layer_t;

/**
 @struct  sTIDL_Network_t
 @brief   This structure define the parmeters CNN/Deep learning net
           in TIDL
 @param  numLayers
          Number of layers in the network inclusing the input and output data 
          Layers
 @param  weightsElementSize
          Size of compute layer weight parameters in bytes
 @param  slopeElementSize
          Size of PRelU layer weight/slope parameters in bytes
 @param  biasElementSize
          Size of compute layer Bias parameters in bytes
 @param  dataElementSize
          Size of compute layer input and adat buffers in bytes
 @param  interElementSize
          Size of compute layer intermeadiate datas in bytes
 @param  quantizationStyle 
          Variable to indicate different types of quantization Styles
*/
typedef struct {
  int32_t numLayers;
  int32_t weightsElementSize;
  int32_t slopeElementSize;
  int32_t biasElementSize;
  int32_t dataElementSize;
  int32_t interElementSize;
  int32_t quantizationStyle;
  int32_t reserved;
  sTIDL_Layer_t TIDLLayers[TIDL_NUM_MAX_LAYERS];
}sTIDL_Network_t;

#if USE_IVISION

/**
  @struct TIDL_CreateParams
  @brief  This structure contains all the parameters which TI DL
          library at create time
  @param  visionParams
          Common parmeters for all ivison based modules
  @param  net
          This structure define the parmeters CNN/Deep learning net
          in TIDL
  @param  currCoreId
          Id of the core (EVE/DSP) on which TIDL is running
  @param  currLayersGroupId
          Id of the each layer (EVE/DSP) on which that layer is 
          running
  @param  l1MemSize
          Value indicates the available l1MemSize for TIDL
  @param  l2MemSize
          Value indicates the available l2MemSize for TIDL
  @param  l3MemSize
          Value indicates the available l3MemSize for TIDL
  @param  quantHistoryParam1
          weights used for previously processed inference 
          during application boot time
  @param  quantHistoryParam2
          weights used for previously processed inference during 
          application execution (After initial few frames)
  @param  quantMargin
          margin added to the average in percentage
  @param  optimiseExtMem
          Value indicates  type of memory optimization to be 
          used by TIDL library
*/
typedef struct
{
  IVISION_Params visionParams;
  sTIDL_Network_t net;
  int32_t currCoreId;
  int32_t currLayersGroupId;
  int32_t l1MemSize;
  int32_t l2MemSize;
  int32_t l3MemSize;
  int32_t quantHistoryParam1;
  int32_t quantHistoryParam2;
  int32_t quantMargin;
  int32_t optimiseExtMem;
  
} TIDL_CreateParams;




/**
  @struct TIDL_InArgs
  @brief  This structure contains all the parameters which controls
          the applet at create time
  @param  iVisionInArgs
          Common inArgs for all ivison based modules
  @param  numInBufs
          Number of input buffers 
  @param  dataQ
          Scale factors for input tensors in Q8 format. 
          The scale factors for input data layers are fixed and used from net structure.
          Only the intermediate layer scale factors needs to be passed here
*/

typedef struct
{
  IVISION_InArgs iVisionInArgs;
  int32_t numInBufs;
  int32_t dataQ[TIDL_NUM_IN_BUFS];
} TIDL_InArgs;

/**
  @struct TIDL_outArgs
  @brief  This structure contains all the parameters which controls
          the applet at create time
  @param  iVisionOutArgs
          Common outArgs for all ivison based modules
  @param  numOutBufs
          Number of output buffers 
  @param  dataQ
          Scale factors for output tensors in Q8 format. 
*/
typedef struct
{
  IVISION_OutArgs iVisionOutArgs;
  int32_t numOutBufs;
  int32_t dataQ[TIDL_NUM_OUT_BUFS];
} TIDL_outArgs;
#endif

/*@}*/
/* iTI_DL */

#endif /*ITIDL_H */
/*==========================================================================*/
/*     END of the FILE                                                      */
/*==========================================================================*/


