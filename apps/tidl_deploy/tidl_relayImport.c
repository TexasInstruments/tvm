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
#include "tidl_import.h"
#include "itidl_ti.h"
#include "ti_dl.h"
#include "tidl_import_utils.h"

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

sTIDL_OrgNetwork_t      orgTIDLNetStructure;
sTIDL_OrgNetwork_t      tempTIDLNetStructure;
sTIDL_Network_t         tIDLNetStructure;
tidlImportState_t       tidlImpState;
//tidlImpConfig           tidlConfigParams;
tidlImpConfig           gParams;    // to do: cleanup here

int32_t gloab_data_format = TIDL_DATA_FORMAT_UNDEFINED;

#define GET_DATA_INDEX (tidlImpState.dataIndex++)
#define GET_LAYER_PTR  (&orgTIDLNetStructure.TIDLPCLayers[tidlImpState.layerIndex])

sTIDL_tfOutRehapeMap_t sTIDL_relayOutRehapeTable[] =
{
};

// Python to C has to have 2 arguments - to figure out why
void tidlImportInit(tidlImpConfig * cfg, void * ptr_unused)
{
  int i;

  printf("Initializing TIDL import.\n");

  gParams.numParamBits  = cfg->numParamBits;
  gParams.quantRoundAdd = cfg->quantRoundAdd;
  gParams.inQuantFactor = cfg->inQuantFactor;
  gParams.inElementType = cfg->inElementType;
  gParams.inNumChannels = cfg->inNumChannels;
  gParams.inHeight      = cfg->inHeight;
  gParams.inWidth       = cfg->inWidth;

  printf("numParamBits  = %d\n", gParams.numParamBits );
  printf("quantRoundAdd = %d\n", gParams.quantRoundAdd);
  printf("inQuantFactor = %d\n", gParams.inQuantFactor);
  printf("inElementType = %d\n", gParams.inElementType);
  printf("inNumChannels = %d\n", gParams.inNumChannels);
  printf("inHeight      = %d\n", gParams.inHeight     );
  printf("inWidth       = %d\n", gParams.inWidth      );
  
  tidlImpState.gloab_data_format = 1; //TIDL_DATA_FORMAT_NCHW;
  tidlImpState.layerIndex = 0;
  tidlImpState.dataIndex  = 0;
  tidlImpState.numErrs = 0;
  tidlImpState.numUnsupportedLayers = 0;

  for(i=0; i<TIDL_NUM_MAX_LAYERS; i++)
  {
    /* Set default values of numInBufs and numOutBufs which may be changed by
       tidl_tfMapFunc below for certain layers. */
    orgTIDLNetStructure.TIDLPCLayers[i].layerType =  TIDL_UnSuportedLayer;
    orgTIDLNetStructure.TIDLPCLayers[i].numInBufs =  1;
    orgTIDLNetStructure.TIDLPCLayers[i].numOutBufs = 1;
  }
}


void tidlImportConv2d(Conv2dParams * conv2dInfo, void * ptr_unused)
{
  int i, num_weights;
  size_t size;
  float * weights;
  sTIDL_LayerPC_t *layer;
  sTIDL_ConvParams_t *convParams;

  TIDL_IMPORT_DBG_PRINT("----- Importing conv2d layer ----- \n");
  printf("Layer index is: %d\n", tidlImpState.layerIndex);
  layer = GET_LAYER_PTR;
  layer->numOutBufs = 1;

  layer->layerType = TIDL_ConvolutionLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
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
    size = sizeof(float)*(size_t)num_weights;
    printf("float32, size is %ld\n", size);
  }
  //else if(strcmp(conv2dInfo->weights_type, "int8") == 0) {
  //  size = sizeof(int8_t)*num_weights;
  //}
  else {
    // No action is needed as Python wrapper already verifies supported data types
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

  //printf("Number of layers imported to TIDL: %d\n", tidlImpState.layerIndex);
}

/*==============================================================================
 * Link this node with other nodes that have been imported so far
 *
 * Equivalent to following 4 functions in TF import:
 *   - tidl_tfLayerFillTensorNames()
 *   - tidl_tfLayerUpdateConsumerCount()
 *   - tidl_linkInputTensors()
 *   - tidl_linkOutputTensors()
==============================================================================*/
// It seems Python calling C has to have at least 2 arguments -- to find out
void tidlImportLinkNodes(InOutNodes *inOutNodes, tidlImpConfig *config)
{
  sTIDL_LayerPC_t *layer;
  int i;
  int32_t *in_nodes;
  char str[10];

  printf("----- Fill tensor names for layer %d -----\n", inOutNodes->this_node);
  //printf("Number of input nodes: %d\n", inOutNodes->num_in_nodes);
  //printf("Number of output nodes: %d\n", inOutNodes->num_out_nodes);

  layer = GET_LAYER_PTR;
  
  // change node index to layer name
  sprintf(str, "%d", inOutNodes->this_node);
  strcpy((char*)layer->name, str);

  // fill in input node names
  if(inOutNodes->num_in_nodes > 0) {
    in_nodes = (int32_t *)inOutNodes->in_nodes;
    for(i=0; i<inOutNodes->num_in_nodes; i++)
    {
      // input data name is the name of the input node 
      sprintf(str, "%d", in_nodes[i]);
      strcpy((char*)layer->inDataNames[i], str);
      printf("Layer %d's input node %d name: %s\n", inOutNodes->this_node, i, str);
    }
  }
  else {
    printf("Number of input nodes is 0. This is the first node.\n");
    if(tidlImpState.layerIndex > 0) {
      printf("Error! This should be the first node.\n");
      exit(0);
    }
    layer->layerType = TIDL_DataLayer;
    layer->numInBufs = -1;
    layer->outData[0].dataId = GET_DATA_INDEX;
    
    // a TIDL input data layer needs to be added as input to this operator
    // - maybe during initialization time. 
    // - TF models have an operator "Placeholder" which is converted to input data layer,
    //   but Relay IR doesn't have such an operator.     
  }

  // fill in output node names
  if(inOutNodes->num_out_nodes > 0) {
    // output data name is the name of this node 
    sprintf(str, "%d", inOutNodes->this_node);
    strcpy((char*)layer->outDataNames[0], str);
    printf("Layer %d's output node 0 name: %s\n", inOutNodes->this_node, str);
    layer->outConsumerLinked[0] = 0; // initialized to 0
    for(i=1; i<layer->numOutBufs; i++)
    {
      char numberStr[10];
      strcpy((char*)layer->outDataNames[i], str);
      strcat((char*)layer->outDataNames[i], "_");
      sprintf(numberStr, "%d", i);
      strcat((char*)layer->outDataNames[i], numberStr);
      printf("Layer %d's output node %d name: %s\n", inOutNodes->this_node, i, layer->outDataNames[i]);
      layer->outConsumerLinked[i] = 0; // initialized to 0
    }
    layer->outConsumerCnt[0] = inOutNodes->num_out_nodes;
  }
  else {
    printf("Number of output nodes is 0. This is the last node.\n");
    layer->outConsumerCnt[0] = 0;   
    // a TIDL output data layer needs to be added as output to this operator
    // - probably should be done similarly to what TF import does, at the end of import
  }

  tidl_linkInputTensors(&orgTIDLNetStructure,  tidlImpState.layerIndex);
  tidl_linkOutputTensors(&orgTIDLNetStructure, tidlImpState.layerIndex);

  printf("Layer %d's numInBufs: %d\n", tidlImpState.layerIndex, orgTIDLNetStructure.TIDLPCLayers[tidlImpState.layerIndex].numInBufs);
  tidlImpState.layerIndex++;
  printf("Number of layers imported to TIDL: %d\n", tidlImpState.layerIndex);
}

int tidlImportOptimize()
{
  int32_t importStatus, i;

  printf("----- Optimize TIDL -----\n");
  printf("number of layers: %d\n", tidlImpState.layerIndex);
  //for(i=0; i<tidlImpState.layerIndex; i++)
  //{
  //  printf("Layer %d, numInBufs = %d\n", i, orgTIDLNetStructure.TIDLPCLayers[i].numInBufs);
  //}

  importStatus = tidl_sortLayersInProcOrder(&orgTIDLNetStructure, &tempTIDLNetStructure, tidlImpState.layerIndex);
  tidlImpState.layerIndex = orgTIDLNetStructure.numLayers;
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\nImport error: This model's topology is not supported.\n");
    //numErrs++;
    return -1;
  }

  tidl_fillInDataLayerShape(&orgTIDLNetStructure, &gParams, tidlImpState.layerIndex);
  tidl_sortDataIds(&orgTIDLNetStructure, tidlImpState.layerIndex);
  
  printf("Updating out data shapes.\n");
  //tidl_updateOutDataShape(&orgTIDLNetStructure, 0, tidlImpState.layerIndex, (sTIDL_tfOutRehapeMap_t *)&sTIDL_relayOutRehapeTable);

}

void tidlImportPad(int size, void *padTensor)
{
  sTIDL_LayerPC_t *layer;

/*   int i;
  int32_t *pad_tensor = (int32_t *)padTensor;
  printf("Padding tensor: [");
  for(i=0; i<size; i++)
  {
    printf("%d ", pad_tensor[i]);
  }
  printf("]\n");
 */
  TIDL_IMPORT_DBG_PRINT("----- Importing pad layer ----- \n");

  layer = GET_LAYER_PTR;
  layer->layerType = TIDL_PadLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;

  memcpy((void*)layer->layerPCParams.padParams.padTensor, padTensor, size*sizeof(int));

/*   printf("Padding tensor after import: [");
  for(i=0; i<size; i++)
  {
    printf("%d ", layer->layerPCParams.padParams.padTensor[i]);
  }
  printf("]\n");
 */
  //return TIDL_IMPORT_NO_ERR;
}

void tidlImportAdd()
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing add layer ----- \n");

  layer = GET_LAYER_PTR;
  layer->layerType = TIDL_EltWiseLayer;
  layer->layerParams.eltWiseParams.eltWiseType = TIDL_EltWiseSum;
  layer->layerParams.eltWiseParams.numInData = 2;
  layer->outData[0].dataId = GET_DATA_INDEX;
  layer->numInBufs = 2;
}

void tidlImportBiasAdd(int numParams, char *dtype, void *biasParams)
{
  int i;
  size_t size;
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing biasAdd layer ----- \n");
  layer = GET_LAYER_PTR;

  if(strcmp(dtype, "float32") == 0) {
    //printf("BiasAdd params are float32, number of params is %d\n", numParams);
    //for(i=0;i<numParams;i++)
    //{
    //  float * params = (float *)biasParams;
    //  printf("%f, ", params[i]);
    //}
    //printf("\n");
    size = (size_t)numParams*sizeof(float);
    layer->bias.ptr = (float *)my_malloc(size);
    memcpy(layer->bias.ptr, biasParams, size);
  }
  else {
    // No action is needed as Python wrapper already verifies supported data types
  }

  layer->layerType = TIDL_BiasLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
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
