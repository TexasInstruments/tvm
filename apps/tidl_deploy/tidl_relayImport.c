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
#include <math.h>
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

typedef struct BatchNormParams
{
  int num_params;
  char *params_dtype;
  float *gama;
  float *beta;
  float *mean;
  float *var;
  float epsilon;
  int center_enable;
  int scale_enable;
} BatchNormParams;

typedef struct PoolingParams
{
  int kernelH;
  int kernelW;
  int strideH;
  int strideW;
  int padH;
  int padW;
} PoolingParams;

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

extern sTIDL_outRehapeMap_t sTIDL_outRehapeTable[];

// Python to C has to have 2 arguments - to figure out why
void tidlImportInit(tidlImpConfig * cfg, char * layout)
{
  int i;

  printf("Initializing TIDL import...\n");

  gParams.numParamBits  = cfg->numParamBits;
  gParams.quantRoundAdd = cfg->quantRoundAdd;
  gParams.inQuantFactor = cfg->inQuantFactor;
  gParams.inElementType = cfg->inElementType;
  gParams.inNumChannels = cfg->inNumChannels;
  gParams.inHeight      = cfg->inHeight;
  gParams.inWidth       = cfg->inWidth;
  gParams.quantizationStyle = TIDL_quantStyleDynamic;
  gParams.modelType     = 1; // to clean up
  for(i = 0; i < TIDL_NUM_MAX_LAYERS; i++)
  {
    gParams.layersGroupId[i] = 1;
    // By default, conv2d kernel type is automatically set instead of being read from config file
    gParams.conv2dKernelType[i] = -1;
  }

  printf("numParamBits  = %d\n", gParams.numParamBits );
  printf("quantRoundAdd = %d\n", gParams.quantRoundAdd);
  printf("inQuantFactor = %d\n", gParams.inQuantFactor);
  printf("inElementType = %d\n", gParams.inElementType);
  printf("inNumChannels = %d\n", gParams.inNumChannels);
  printf("inHeight      = %d\n", gParams.inHeight     );
  printf("inWidth       = %d\n", gParams.inWidth      );
  
  if(strcmp(layout, "NCHW") == 0) {
    //tidlImpState.gloab_data_format = TIDL_DATA_FORMAT_NCHW;
    gloab_data_format = TIDL_DATA_FORMAT_NCHW;
  }
  else {
    gloab_data_format = TIDL_DATA_FORMAT_NHWC;
  }

  tidlImpState.numErrs = 0;
  tidlImpState.numUnsupportedLayers = 0;

  // Create a Data layer: 
  //  - TF models have an operator "Placeholder" which is converted to input data 
  //    layer, but Relay IR doesn't have such an operator.     
  orgTIDLNetStructure.TIDLPCLayers[0].layerType = TIDL_DataLayer;
  orgTIDLNetStructure.TIDLPCLayers[0].numInBufs  = -1;
  orgTIDLNetStructure.TIDLPCLayers[0].numOutBufs = 1;
  //orgTIDLNetStructure.TIDLPCLayers[0].outConsumerCnt[0] = 1;
  strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[0].outDataNames[0], "InputData");
  orgTIDLNetStructure.TIDLPCLayers[0].weightsElementSizeInBits = (int32_t)NUM_WHGT_BITS;
  orgTIDLNetStructure.TIDLPCLayers[0].strideOffsetMethod = (int32_t)TIDL_strideOffsetCenter;

  tidlImpState.layerIndex = 1;
  tidlImpState.dataIndex  = 1;

  // Initialize rest of the layers
  for(i=1; i<TIDL_NUM_MAX_LAYERS; i++)
  {
    /* Set default values of numInBufs and numOutBufs which may be changed by
       tidl_tfMapFunc below for certain layers. */
    orgTIDLNetStructure.TIDLPCLayers[i].layerType =  TIDL_UnSuportedLayer;
    orgTIDLNetStructure.TIDLPCLayers[i].numInBufs =  1;
    orgTIDLNetStructure.TIDLPCLayers[i].numOutBufs = 1;
    gParams.conv2dKernelType[i] = -1;
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
    // To add error handling
  }

  // Allocate memory to store weights. To be freed after writing weights to file.
  layer->weights.ptr     = my_malloc(size);
  layer->weights.bufSize = num_weights;
  memcpy(layer->weights.ptr, conv2dInfo->weights_array, size);

  printf("TIDL conv2d parameters");
  printf("Stride accross width: %d\n",    conv2dInfo->stride_w);
  printf("Stride accross height: %d\n",   conv2dInfo->stride_h);
  printf("Padding accross width: %d\n",    conv2dInfo->pad_w);
  printf("Padding accross height: %d\n",   conv2dInfo->pad_h);
  printf("Dilation accross width: %d\n",  conv2dInfo->dilation_w);
  printf("Dilation accross height: %d\n", conv2dInfo->dilation_h);
  printf("Kernel width: %d\n",            conv2dInfo->kernel_w);
  printf("Kernel height: %d\n",           conv2dInfo->kernel_h);
  printf("Weights array type: %s\n",      conv2dInfo->weights_type);
  printf("Weights array size: %ld\n",     size);

  {
    printf("First 10 weights: \n");
    float * weights = (float *)layer->weights.ptr;
    for(i=0; i<9; i++) 
    {
      printf("%f, ", weights[i]);
    }
    printf("%f\n", weights[i]);
  }
}

void TIDL_tfBNToScaleBias(
  float    * scale,
  float    * bias,
  uint32_t  numCh,
  float * mean,
  float * var,
  float * gamma,
  float * beta,
  float eps
  )

{
  uint32_t j;
  for (j = 0; j < numCh; j++)
  {
    double m = mean[j];
    double v = var[j];
    double s = gamma[j];
    double b = beta[j];
    double inv_var = pow((eps + v), -0.5);
    scale[j] = (s)*inv_var;
    bias[j]  = (((-m)*s)*inv_var) + b;
  }
}

void tidlImportBatchNorm(BatchNormParams * bn_params, void * ptr_unused)
{
  int i;
  size_t params_size;
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Batch Norm layer ----- \n");

  layer = GET_LAYER_PTR;

  layer->layerType = TIDL_BatchNormLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;

  if(strcmp(bn_params->params_dtype, "float32") == 0) {
    params_size = sizeof(float)*bn_params->num_params;
  }
  // To add support of quantized model
  //else if(strcmp(bn_params->params_dtype, "int8") == 0) {
  //  size = sizeof(int8_t)*bn_params->num_params;
  //}
  else {
    // To add error handling
  }

  layer->weights.ptr     = my_malloc(params_size);
  layer->weights.bufSize = bn_params->num_params;
  layer->bias.ptr        = my_malloc(params_size);
  layer->bias.bufSize    = bn_params->num_params;

  TIDL_tfBNToScaleBias((float *)layer->weights.ptr, (float *)layer->bias.ptr, 
                       bn_params->num_params, bn_params->mean, bn_params->var, 
                       bn_params->gama, bn_params->beta, bn_params->epsilon);

/*
  printf("Number of BN parameters: %d\n", bn_params->num_params); 
  printf("BN parameter data type: %s\n", bn_params->params_dtype); 
  
  for(i=0;i<bn_params->num_params;i++)
  {
    printf("%f, ", bn_params->gama[i]);
  }
  printf("\n");
  for(i=0;i<bn_params->num_params;i++)
  {
    printf("%f, ", bn_params->beta[i]);
  }
  printf("\n");
  for(i=0;i<bn_params->num_params;i++)
  {
    printf("%f, ", bn_params->mean[i]);
  }
  printf("\n");
  for(i=0;i<bn_params->num_params;i++)
  {
    printf("%f, ", bn_params->var[i]);
  }
  printf("\n");
  
  printf("BN epsilon: %f\n", bn_params->epsilon); 
  printf("BN center: %d, scale: %d\n", bn_params->center_enable, bn_params->scale_enable); 
*/  
} /* tidlImportBatchNorm */

void tidlImportPooling(PoolingParams *params, char * pooling_type)
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Pooling layer ----- \n");

  layer = GET_LAYER_PTR;

  layer->layerType = TIDL_PoolingLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;

  layer->layerParams.poolParams.kernelH = params->kernelH;
  layer->layerParams.poolParams.kernelW = params->kernelW;
  layer->layerParams.poolParams.strideH = params->strideH;
  layer->layerParams.poolParams.strideW = params->strideW;
  layer->layerParams.poolParams.padH    = params->padH;
  layer->layerParams.poolParams.padW    = params->padW;

  if(strcmp(pooling_type, "avg_pool2d") == 0) {
    layer->layerParams.poolParams.poolingType = TIDL_AveragePooling;
  }
  else {
    layer->layerParams.poolParams.poolingType = TIDL_MaxPooling;
  }

  printf("Pooling type: %s\n", pooling_type);
  printf("Params: (%d, %d), (%d, %d), (%d, %d)\n", params->kernelH, params->kernelW,
          params->strideH, params->strideW, params->padH, params->padW);
  
  return;
}

void tidlImportSqueeze()
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Squeeze layer -----\n");

  layer = GET_LAYER_PTR;

  layer->layerType = TIDL_SqueezeLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
}

void tidlImportReshape()
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Reshape layer -----\n");

  layer = GET_LAYER_PTR;

  layer->layerType = TIDL_ReshapeLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
}

void tidlImportSoftmax()
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Softmax layer -----\n");

  layer = GET_LAYER_PTR;

  layer->layerType = TIDL_SoftMaxLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
}

void tidlImportPad(int size, void *padTensor)
{
  sTIDL_LayerPC_t *layer;

   int i;
  int32_t *pad_tensor = (int32_t *)padTensor;
  printf("Padding tensor: [");
  for(i=0; i<size; i++)
  {
    printf("%d ", pad_tensor[i]);
  }
  printf("]\n");

  TIDL_IMPORT_DBG_PRINT("----- Importing pad layer ----- \n");

  layer = GET_LAYER_PTR;
  layer->layerType = TIDL_PadLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;

  memcpy((void*)layer->layerPCParams.padParams.padTensor, padTensor, size*sizeof(int));

   printf("Padding tensor after import: [");
  for(i=0; i<size; i++)
  {
    printf("%d ", layer->layerPCParams.padParams.padTensor[i]);
  }
  printf("]\n");
 
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
    // Allocate memory to store weights. To be freed after writing weights to file.
    size = (size_t)numParams*sizeof(float);
    layer->bias.bufSize = numParams;
    layer->bias.ptr = (float *)my_malloc(size);
    memcpy(layer->bias.ptr, biasParams, size);
  }
  else {
    // To add "int8"
  }

  layer->layerType = TIDL_BiasLayer;
  layer->outData[0].dataId = GET_DATA_INDEX;
}

void tidlImportRelu(char * reluType)
{
  sTIDL_LayerPC_t *layer;

  TIDL_IMPORT_DBG_PRINT("----- Importing Relu layer ----- \n");

  layer = GET_LAYER_PTR;
  layer->layerType = TIDL_ReLULayer;
  layer->outData[0].dataId = GET_DATA_INDEX;

  if(strcmp(reluType, "Relu6") == 0) {
    layer->layerParams.reluParams.reluType = TIDL_RelU6;
    printf("Relu6\n");
  }
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
void tidlImportLinkNodes(InOutNodes *inOutNodes, void *ptr_unused)
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
    printf("Number of input nodes is 0. This is the first layer after input data layer.\n");
    if(tidlImpState.layerIndex > 1) {
      printf("Error! This should be the first node.\n");
      exit(0);
    }
    
    // Connect to first layer which should be a data layer
    printf("Frist layer is %s\n", (char*)orgTIDLNetStructure.TIDLPCLayers[0].outDataNames[0]);
    strcpy((char*)layer->inDataNames[0], (char*)orgTIDLNetStructure.TIDLPCLayers[0].outDataNames[0]);
    orgTIDLNetStructure.TIDLPCLayers[0].outConsumerCnt[0] = 1;
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
    printf("outConsumerCnt[0] = %d\n", layer->outConsumerCnt[0]);
  }
  else {
    printf("Number of output nodes is 0. This is the last node.\n");
    layer->outConsumerCnt[0] = 0;   
    // a TIDL output data layer needs to be added as output to this operator
    // - probably should be done similarly to what TF import does, at the end of import
  }

  layer->weightsElementSizeInBits = (int32_t)NUM_WHGT_BITS;
  layer->strideOffsetMethod = (int32_t)TIDL_strideOffsetCenter;

  tidl_linkInputTensors(&orgTIDLNetStructure,  tidlImpState.layerIndex);
  tidl_linkOutputTensors(&orgTIDLNetStructure, tidlImpState.layerIndex);

  printf("Layer %d's numInBufs: %d\n", tidlImpState.layerIndex, orgTIDLNetStructure.TIDLPCLayers[tidlImpState.layerIndex].numInBufs);
  tidlImpState.layerIndex++;
  printf("Number of layers imported to TIDL: %d\n", tidlImpState.layerIndex);
} // tidlImportLinkNodes()


void tidl_printLayerparams(sTIDL_Network_t  *tIDLNetStructure, int32_t numLayers)
{
  int i;
  FILE *fp_params;
  
  fp_params = fopen("layer_params.txt", "w");

  fprintf(fp_params, "Network parameters: %d, %d, %d, %d, %d, %d, %d\n", 
                     tIDLNetStructure->numLayers,
                     tIDLNetStructure->weightsElementSize,
                     tIDLNetStructure->slopeElementSize,
                     tIDLNetStructure->biasElementSize,
                     tIDLNetStructure->dataElementSize,
                     tIDLNetStructure->interElementSize,
                     tIDLNetStructure->quantizationStyle);

  for(i=0; i<numLayers; i++)
  {
    fprintf(fp_params, "Layer %d parameters: %d, %d, %d, %d, %d, %d, %d\n", i, 
                       tIDLNetStructure->TIDLLayers[i].layerType,
                       tIDLNetStructure->TIDLLayers[i].numInBufs,
                       tIDLNetStructure->TIDLLayers[i].numOutBufs,
                       tIDLNetStructure->TIDLLayers[i].coreID,
                       tIDLNetStructure->TIDLLayers[i].layersGroupId,
                       tIDLNetStructure->TIDLLayers[i].weightsElementSizeInBits,
                       tIDLNetStructure->TIDLLayers[i].strideOffsetMethod);

    fprintf(fp_params, "inData parameters: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", 
                       tIDLNetStructure->TIDLLayers[i].inData[0].dataId,
                       tIDLNetStructure->TIDLLayers[i].inData[0].elementType,    
                       tIDLNetStructure->TIDLLayers[i].inData[0].numDim,
                       tIDLNetStructure->TIDLLayers[i].inData[0].dataQ,
                       tIDLNetStructure->TIDLLayers[i].inData[0].minValue,
                       tIDLNetStructure->TIDLLayers[i].inData[0].maxValue,
                       tIDLNetStructure->TIDLLayers[i].inData[0].pitch[0],
                       tIDLNetStructure->TIDLLayers[i].inData[0].pitch[1],
                       tIDLNetStructure->TIDLLayers[i].inData[0].pitch[2],
                       tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[0],
                       tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[1],
                       tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[2],
                       tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[3]);
    fprintf(fp_params, "outData parameters: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", 
                       tIDLNetStructure->TIDLLayers[i].outData[0].dataId,
                       tIDLNetStructure->TIDLLayers[i].outData[0].elementType,    
                       tIDLNetStructure->TIDLLayers[i].outData[0].numDim,
                       tIDLNetStructure->TIDLLayers[i].outData[0].dataQ,
                       tIDLNetStructure->TIDLLayers[i].outData[0].minValue,
                       tIDLNetStructure->TIDLLayers[i].outData[0].maxValue,
                       tIDLNetStructure->TIDLLayers[i].outData[0].pitch[0],
                       tIDLNetStructure->TIDLLayers[i].outData[0].pitch[1],
                       tIDLNetStructure->TIDLLayers[i].outData[0].pitch[2],
                       tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[0],
                       tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[1],
                       tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[2],
                       tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[3]);

    if(tIDLNetStructure->TIDLLayers[i].layerType == TIDL_DataLayer)
    {
      sTIDL_DataLayerParams_t * dataParams = &tIDLNetStructure->TIDLLayers[i].layerParams.dataLayerParams;
      fprintf(fp_params, "Layer %d, data layer parameters: %d, %d\n", i, 
                         dataParams->numChannels,dataParams->dataQ);
    }
    
    if(tIDLNetStructure->TIDLLayers[i].layerType == TIDL_InnerProductLayer)
    {
      fprintf(fp_params, "Layer %d, Inner Product layer, input dimension values:  %d, %d, %d, %d\n", i, 
         tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[0],
         tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[1],   
         tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[2],   
         tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[3]);
      fprintf(fp_params, "                               output dimension values: %d, %d, %d, %d\n", 
         tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[0],
         tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[1],   
         tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[2],   
         tIDLNetStructure->TIDLLayers[i].outData[0].dimValues[3]);
         
      //printf("Transpose input: \n");
      //tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[3] = tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[1];
      //tIDLNetStructure->TIDLLayers[i].inData[0].dimValues[1] = 1;
    }

    if(tIDLNetStructure->TIDLLayers[i].layerType == TIDL_SoftMaxLayer)
    {
      sTIDL_SoftMaxParams_t *softmaxParams = &tIDLNetStructure->TIDLLayers[i].layerParams.softMaxParams;
      fprintf(fp_params, "Layer %d, SoftMax layer parameters: %d, %d, %d\n", i,
                         softmaxParams->numChannels, softmaxParams->inDataQ, softmaxParams->outDataQ);
    }
    
    if(tIDLNetStructure->TIDLLayers[i].layerType == TIDL_PoolingLayer)
    {
      sTIDL_PoolingParams_t *poolingParams = &tIDLNetStructure->TIDLLayers[i].layerParams.poolParams;
      fprintf(fp_params, "Layer %d, Pooling layer parameters: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", i,
                          poolingParams->numChannels,
                          poolingParams->poolingType,
                          poolingParams->kernelW,
                          poolingParams->kernelH,
                          poolingParams->strideW,
                          poolingParams->strideH,
                          poolingParams->padW,
                          poolingParams->padH,
                          poolingParams->inDataQ,
                          poolingParams->outDataQ);
    }
    if(tIDLNetStructure->TIDLLayers[i].layerType == TIDL_ConvolutionLayer)
    {
      sTIDL_ConvParams_t *convParams = &tIDLNetStructure->TIDLLayers[i].layerParams.convParams;
      //if(i==1 || i==4 || i==8 || i==12 || i==24) 
      //{
      //  convParams->padW = convParams->padH = 1; 
      //}
      fprintf(fp_params, "Layer %d, conv2d layer parameters: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", i, 
             convParams->convolutionType,
             convParams->numInChannels,
             convParams->numOutChannels,
             convParams->numGroups    ,
             convParams->kernelW      ,
             convParams->kernelH      ,
             convParams->strideW      ,
             convParams->strideH      ,
             convParams->dilationW    ,
             convParams->dilationH    ,
             convParams->padW         ,
             convParams->padH         ,
             convParams->weightsQ     ,
             convParams->zeroWeightValue,
             convParams->biasQ        ,
             convParams->inDataQ      ,
             convParams->outDataQ     ,
             convParams->interDataQ   ,
             convParams->enableBias   ,
             convParams->enablePooling,
             convParams->enableRelU   ,
             convParams->kernelType);
    }
  }

  fclose(fp_params);
}


int tidlImportOptimize(int graphId)
{
  int32_t importStatus, i, numErrs, numUnsupportedLayers, tiLayerIndex;
  FILE    *fpNetFile;
  FILE    *fpParamsFile;
  char    str[30];

  numErrs = 0;
  printf("TIDL import graph %d\n", graphId);
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
  tidl_updateOutDataShape(&orgTIDLNetStructure, 0, tidlImpState.layerIndex, (sTIDL_outRehapeMap_t *)&sTIDL_outRehapeTable);
  printf("Out data shapes updated.\n");

  importStatus = tidl_mergeBiasLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Bias layer cannot be merged into other layers.\n");
    numErrs++;
  }

  importStatus = tidl_mergeBNLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Batch Norm layer cannot be merged into other layers.\n");
    numErrs++;
  }

  importStatus = tidl_mergeReluLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Relu layer cannot be merged into other layers.\n");
    numErrs++;
  }

  importStatus = tidl_mergePadLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Pad layer cannot be merged into other layers.\n");
    numErrs++;
  }

  importStatus = tidl_mergePoolLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Pad layer cannot be merged into other layers.\n");
    numErrs++;
  }

  tidl_removeMergedLayersFromNet(&orgTIDLNetStructure, &tempTIDLNetStructure, tidlImpState.layerIndex);
  tidlImpState.layerIndex = orgTIDLNetStructure.numLayers;
  tidl_sortDataIds(&orgTIDLNetStructure, tidlImpState.layerIndex);
  tidl_findFlattenLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  tidl_removeMergedLayersFromNet(&orgTIDLNetStructure, &tempTIDLNetStructure, tidlImpState.layerIndex);
  tidlImpState.layerIndex = orgTIDLNetStructure.numLayers;
  tidl_sortDataIds(&orgTIDLNetStructure, tidlImpState.layerIndex);

  importStatus = tidl_convertConv2DToIpLayer(&orgTIDLNetStructure, tidlImpState.layerIndex, (sTIDL_outRehapeMap_t *)&sTIDL_outRehapeTable);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Conv2D layer cannot be converted into Inner Product layer.\n");
    numErrs++;
  }

  importStatus = tidl_mergeFlattenLayer(&orgTIDLNetStructure, tidlImpState.layerIndex);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\n Import error: Flatten layer cannot be merged.\n");
    numErrs++;
  }

  importStatus = tidl_mergeReshapeLayer(&orgTIDLNetStructure, tidlImpState.layerIndex, (sTIDL_outRehapeMap_t *)&sTIDL_outRehapeTable);
  if(importStatus != TIDL_IMPORT_NO_ERR)
  {
    printf("\nImport error: Reshape layer cannot be merged.\n");
    numErrs++;
  }

  tidl_removeMergedLayersFromNet(&orgTIDLNetStructure, &tempTIDLNetStructure, tidlImpState.layerIndex);
  tidlImpState.layerIndex = orgTIDLNetStructure.numLayers;
  tidl_sortDataIds(&orgTIDLNetStructure, tidlImpState.layerIndex);

  tidl_importEltWiseParams(&orgTIDLNetStructure, tidlImpState.layerIndex);

  /* Quantize and write out layer params */
  sprintf(str, "tidl_subgraph%d_params.bin", graphId);
  fpParamsFile = fopen(str, "wb+");
  TIDL_importQuantWriteLayerParams(&orgTIDLNetStructure, tidlImpState.layerIndex, fpParamsFile);
  fclose(fpParamsFile);

  tiLayerIndex = tidl_copyPCNetToDeviceNet(&orgTIDLNetStructure, &tIDLNetStructure, tidlImpState.layerIndex, NUM_WHGT_BITS);

  /* Have a final check if there are any layers which
     - are supported only by merging with other TIDL layers,
     - but are not able to be merged. */
  numUnsupportedLayers = tidl_countUnsupportedLayers(&tIDLNetStructure, tiLayerIndex);
  if(numUnsupportedLayers > 0)
  {
    printf("\nImport error: This TensorFlow model has operators that are supported"
           " by TIDL only if they can be merged with TIDL layers. But these operators"
           " cannot be merged with any TIDL layer.\n"
           "Please check TIDL User's Guide for supported TensorFlow operators.\n");
    numErrs++;
  }

  if(numErrs > 0)
  {
    /* Stop the import process here to prevent potential crash if continuing */
    return TIDL_IMPORT_FAILURE;
  }

  /* Function to set Conv2dKernelType in layer params based on the "conv2dKernelType"
     parameter from import config file.
  */
  TIDL_setConv2dKernelType(&tIDLNetStructure, tiLayerIndex);

  tidl_addOutDataLayer(&tIDLNetStructure, tiLayerIndex);

  tidl_printLayerparams(&tIDLNetStructure, tiLayerIndex);

  sprintf(str, "tidl_subgraph%d_net.bin", graphId);
  fpNetFile = fopen(str, "wb+");
  fwrite(&tIDLNetStructure,1,sizeof(tIDLNetStructure),fpNetFile);

  fclose(fpNetFile);

  return TIDL_IMPORT_SUCCESS;
} // tidlImportOptimize()


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
