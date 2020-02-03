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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "tidl_import.h"
#include "itidl_ti.h"
#include "ti_dl.h"

#define TIDL_IMPORT_ENABLE_DBG_PRINT
#ifdef TIDL_IMPORT_ENABLE_DBG_PRINT
#define TIDL_IMPORT_DBG_PRINT(dbg_msg) printf(dbg_msg)
#define TIDL_IMPORT_DBG_PRINT2(dbg_msg, var) printf(dbg_msg, var)
#else
#define TIDL_IMPORT_DBG_PRINT(dbg_msg)
#define TIDL_IMPORT_DBG_PRINT2(dbg_msg, var)
#endif

typedef struct tidlImportState
{
  int layerIndex;
  int gloab_data_format;
  int dataIndex;
  int numErrs;
  int numUnsupportedLayers;
} tidlImportState_t;

sTIDL_OrgNetwork_t      orgTIDLNetStructure;
sTIDL_OrgNetwork_t      tempTIDLNetStructure;
sTIDL_Network_t         tIDLNetStructure;
tidlImportState_t       tidlImpState;

static int totalMemAllocation = 0;
void * my_malloc(int size)
{
  void *ptr;
  totalMemAllocation += size;
  ptr = malloc(size);
  assert(ptr != NULL);

  return ptr;
}

int tidlImpConvertRelayIr(void *relayIrAst, tidlImpConfig *config,
                          char *tidlNetFile, char *tidlParamsFile)
{
  printf("Hello TIDL import library Relay IR conversion!\n");

  return 0;
}

int tidlImpCalibRelayIr(void *relayIrInTensor, char *tidlCalibTool, 
                        tidlCalibConfig *config,
                        char *tidlNetFile, char *tidlParamsFile)
{
  printf("Hello TIDL import library Relay IR calibration!\n");

  return 0;
}

typedef struct Conv2dParams
{
    int num_in_channels;
    int num_out_channels;
    int num_groups;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int kernel_h;
    int kernel_w;
    char *kernel_layout;
    void *weights_array;
    char *weights_type;
} Conv2dParams;

typedef struct InOutNodes
{
  int   this_node;
  int   num_in_nodes;
  int   num_out_nodes;
  void *in_nodes;
  void *out_nodes;
} InOutNodes;

void tidlImportInit()
{
  int i;

  printf("Initializing TIDL import.\n");

  tidlImpState.gloab_data_format = 1; //TIDL_DATA_FORMAT_NCHW;
  tidlImpState.layerIndex = 0;
  tidlImpState.dataIndex  = 0;
  tidlImpState.numErrs = 0;
  tidlImpState.numUnsupportedLayers = 0;

  for(i=0; i<TIDL_NUM_MAX_LAYERS; i++)
  {
    /* Set default values of numInBufs and numOutBufs which may be changed by
       tidl_tfMapFunc below for certain layers. */
    orgTIDLNetStructure.TIDLPCLayers[i].numInBufs =  1;
    orgTIDLNetStructure.TIDLPCLayers[i].numOutBufs = 1;
  }
}


void tidlImportConv2d(Conv2dParams * conv2dInfo, tidlImpConfig *config)
{
  int i, num_weights, size;
  float * weights;
  sTIDL_LayerPC_t *layer;
  sTIDL_ConvParams_t *convParams;

  TIDL_IMPORT_DBG_PRINT("Importing conv2d layer... \n");
  printf("Layer index is: %d\n", tidlImpState.layerIndex);
  layer = &orgTIDLNetStructure.TIDLPCLayers[tidlImpState.layerIndex];
  layer->numOutBufs = 1;

  layer->layerType = TIDL_ConvolutionLayer;
  layer->outData[0].dataId = tidlImpState.dataIndex++;
  layer->outData[0].elementType = TIDL_SignedChar;

  convParams = &layer->layerParams.convParams;
  convParams->numInChannels   = conv2dInfo->num_in_channels;
  convParams->numOutChannels  = conv2dInfo->num_out_channels;
  convParams->kernelW         = conv2dInfo->kernel_w;
  convParams->kernelH         = conv2dInfo->kernel_h;
  convParams->numGroups       = conv2dInfo->num_groups;
  convParams->dilationW       = conv2dInfo->dilation_w;
  convParams->dilationH       = conv2dInfo->dilation_h;
  convParams->strideW         = conv2dInfo->stride_w;
  convParams->strideH         = conv2dInfo->stride_h;
  convParams->padW            = conv2dInfo->pad_w;
  convParams->padH            = conv2dInfo->pad_h;

  convParams->enableBias      = 0;
  convParams->enableRelU      = 0;
  convParams->enablePooling   = 0;

  num_weights =  conv2dInfo->num_in_channels * conv2dInfo->num_out_channels
               * conv2dInfo->kernel_h * conv2dInfo->kernel_w;
  printf("Number of weights: %d\n",num_weights);
  printf("Weights type: %s\n", conv2dInfo->weights_type);
  if(strcmp(conv2dInfo->weights_type, "float32") == 0) {
    size = sizeof(float)*num_weights;
    printf("float32, size is %d\n", size);
  }
  else if(strcmp(conv2dInfo->weights_type, "int32") == 0) {
    size = sizeof(int32_t)*num_weights;
  }

  layer->weights.ptr = my_malloc(size);
  memcpy(layer->weights.ptr, conv2dInfo->weights_array, size);


//  printf("\n =============== TIDL import conv2d ===================\n");
//  printf("Stride accross width: %d\n",    test_conv2dParams.stride_w);
//  printf("Stride accross height: %d\n",   test_conv2dParams.stride_h);
//  printf("Dilation accross width: %d\n",  test_conv2dParams.dilation_w);
//  printf("Dilation accross height: %d\n", test_conv2dParams.dilation_h);
//  printf("Kernel width: %d\n",            test_conv2dParams.kernel_w);
//  printf("Kernel height: %d\n",           test_conv2dParams.kernel_h);
//  printf("Weights array type: %s\n",      conv2dInfo->weights_type);
//  printf("First 10 weights: \n");
//  for(i=0; i<10; i++) 
//  {
//    float * weights = (float *)conv2dInfo->weights_array;
//    printf("%f\t", weights[i]);
//  }

  //tidlImpState.dataIndex++;
  //tidlImpState.layerIndex++;
  //printf("Number of layers imported to TIDL: %d\n", tidlImpState.layerIndex);
}

// It seems Python calling C has to have at least 2 arguments -- to find out
void tidlImportSetInOutNodes(InOutNodes *inOutNodes, tidlImpConfig *config)
{
  int i;
  int *in_nodes;
  char str[10];
  printf("Setup input and output nodes for node %d.\n", inOutNodes->this_node);

  printf("Number of input nodes: %d\n", inOutNodes->num_in_nodes);
  printf("Number of output nodes: %d\n", inOutNodes->num_out_nodes);
  in_nodes = (int *)inOutNodes->in_nodes;
  if(inOutNodes->num_in_nodes == 0) {
    printf("Number of input nodes is 0. This is the first node.\n");
  }
  else {
    for(i=0; i<inOutNodes->num_in_nodes; i++)
    {
      sprintf(str, "%d", in_nodes[i]);
      printf("Input node %d: node %s\n", i, str);
    }
  }


  tidlImpState.layerIndex++;
  printf("Number of layers imported to TIDL: %d\n", tidlImpState.layerIndex);
}

////////////////////// testing and prototyping code ////////////////////////////
Conv2dParams test_conv2dParams;

void tidlInitConv2dParams()
{
  printf("Initialize conv2d params.\n");

  test_conv2dParams.stride_w = 0;
  test_conv2dParams.stride_h = 0;
  test_conv2dParams.dilation_w = 0;
  test_conv2dParams.dilation_h = 0;
  test_conv2dParams.kernel_w = 0;
  test_conv2dParams.kernel_h = 0;
}

void tidlSetConv2dParams(Conv2dParams * conv2dInfo, tidlImpConfig *config)
{
  int i;
  char str[10];

  printf("\n =============== TIDL import conv2d ===================\n");
  printf("Stride accross width: %d\n",    conv2dInfo->stride_w);
  printf("Stride accross height: %d\n",   conv2dInfo->stride_h);
  printf("Dilation accross width: %d\n",  conv2dInfo->dilation_w);
  printf("Dilation accross height: %d\n", conv2dInfo->dilation_h);
  printf("Kernel width: %d\n",            conv2dInfo->kernel_w);
  printf("Kernel height: %d\n",           conv2dInfo->kernel_h);
  printf("Weights array type: %s\n",      conv2dInfo->weights_type);
  printf("First 10 weights: \n");
  for(i=0; i<10; i++) 
  {
    float * weights = (float *)conv2dInfo->weights_array;
    printf("%f\t", weights[i]);
  }

  sprintf(str, "%d", 100);
  printf("\nConvert integer 100 to string: %s", str);
}
