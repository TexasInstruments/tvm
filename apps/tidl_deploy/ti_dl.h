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


#ifndef TIDL_H_
#define TIDL_H_ 1

#include "itidl_ti.h"

typedef enum
{
  TIDL_ConstDataLayer       = TIDL_UnSuportedLayer+1, // 21
  // These layers don't have implmentation in TIDL lib, but are expected to be
  // fused with other layers. If they can't be fused, then the network is not 
  // able to run with TIDL. 
  TIDL_ShuffleChannelLayer   , //22
  TIDL_ResizeLayer           , //23
  TIDL_PriorBoxLayer         , //24
  TIDL_PermuteLayer          , //25
  TIDL_ReshapeLayer          , //26
  TIDL_ShapeLayer            , //27
  TIDL_SqueezeLayer          , //28
  TIDL_PadLayer              , //29
  TIDL_TransposeLayer        , //30
}eTIDL_PCLayerType;

extern const char * TIDL_LayerString[];
#define TIDL_NUM_MAX_PC_LAYERS (1024)


typedef struct {
  /** Buffer containing Dim values for output tensor */
  int32_t   outDims[TIDL_DIM_MAX];
}sTIDL_ReshapeParams_t;

typedef struct {
  /** Buffer containing Axis values, to be squeezed if 1*/
  int32_t   axis[TIDL_DIM_MAX];
}sTIDL_SqueezeParams_t;

typedef struct {
  int32_t   padTensor[TIDL_DIM_MAX*2];
}sTIDL_PadParams_t;

/**
@struct  sTIDL_ShuffleLayerParams_t
@brief   This structure define the parmeters of Shuffle layer
in TIDL
*/
typedef struct {
  /** Num channel in the each out buffer /tesnor */
  int32_t   sliceNumChs[TIDL_NUM_OUT_BUFS];

}sTIDL_SliceLayerParams_t;

/**
@struct  sTIDL_LayerPCParams_t
@brief   This structure holds the parmeters of the layers that are not supported
         in TIDL lib yet.
*/
typedef union {
  sTIDL_ReshapeParams_t      reshapeParams;
  sTIDL_SqueezeParams_t      squeezeParams;
  sTIDL_PadParams_t          padParams;
  sTIDL_SliceLayerParams_t   sliceParams;
}sTIDL_LayerPCParams_t;

/**
@struct  sBufferPc_t
@brief   This structure is used to store parameters during import, and is not
         written to output binary files.
*/
typedef struct
{
  void* ptr;
  int32_t bufSize;
  int32_t reserved[2];
}sBufferPc_t;

typedef struct {
    sTIDL_LayerParams_t layerParams;
    sTIDL_LayerPCParams_t layerPCParams;
    int32_t layerType;
    int32_t numInBufs;
    int32_t numOutBufs;
    int64_t numMacs;
    int8_t  name[TIDL_STRING_SIZE];
    int8_t  inDataNames[TIDL_NUM_IN_BUFS][TIDL_STRING_SIZE];   
    int8_t  outDataNames[TIDL_NUM_OUT_BUFS][TIDL_STRING_SIZE];
    int32_t outConsumerCnt[TIDL_NUM_OUT_BUFS];
    int32_t outConsumerLinked[TIDL_NUM_OUT_BUFS];
    sTIDL_DataParams_t inData[TIDL_NUM_IN_BUFS];
    sTIDL_DataParams_t outData[TIDL_NUM_OUT_BUFS]; 
    sBufferPc_t weights;
    sBufferPc_t bias;
    sBufferPc_t slope; 
    sBufferPc_t priorBox;  
    int32_t weightsElementSizeInBits;  //kernel weights in bits
    /** Offset selection method for stride. @ref eTIDL_strideOffsetMethod */
    int32_t strideOffsetMethod;

}sTIDL_LayerPC_t;

typedef struct {
  int32_t numLayers;
  sTIDL_LayerPC_t TIDLPCLayers[TIDL_NUM_MAX_PC_LAYERS];
}sTIDL_OrgNetwork_t;


#endif  /* TI_DL_H_ */

/* =========================================================================*/
/*  End of file:  ti_od_cnn.h                                               */
/* =========================================================================*/
