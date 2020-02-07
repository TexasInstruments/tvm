
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <assert.h>

#include "ti_dl.h"
#include "tidl_import_utils.h"

extern tidlImpConfig gParams;    // to do: cleanup here
#define QUAN_STYLE2_ROUND ((gParams.quantRoundAdd*1.0 / 100))

extern int32_t gloab_data_format;

static int totalMemAllocation = 0;
void * my_malloc(size_t size)
{
  void *ptr;
  totalMemAllocation += size;
  ptr = malloc(size);
  assert(ptr != NULL);

  return ptr;
}
void my_free(void *ptr)
{
  //fprintf(fpAlloc, "Free: Ptr: %0x\n",ptr);
  //fflush(fpAlloc);
  free(ptr);
}

/*==============================================================================
* Function purpose: for current layer, search from all mapped layers and find those
*   whose output is the input of current layer. Once a match is found, it will 
*   link the two layers: assign output data id of the found layer to input data 
*   id of current layer. 
*
* Note: - this function is a duplication of same function in tidl-import-utils.
*       - it will be removed when moving C code to tidl-import-utils.
==============================================================================*/
int32_t tidl_linkInputTensors(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i0, i1, i2;
  sTIDL_LayerPC_t *pCurrentLayer;
  sTIDL_LayerPC_t *pSearchLayer;

  pCurrentLayer = &pOrgTIDLNetStructure->TIDLPCLayers[layerIndex];
  for (i0 = 0; i0 < pCurrentLayer->numInBufs; i0++)
  {
    for (i1 = layerIndex - 1; i1 >= 0; i1--)
    {
      pSearchLayer = &pOrgTIDLNetStructure->TIDLPCLayers[i1];
      for (i2 = 0; i2 < pSearchLayer->numOutBufs; i2++)
      {
        if (pSearchLayer->outConsumerLinked[i2] < pSearchLayer->outConsumerCnt[i2])
        {
          if (strcmp((const char *)pCurrentLayer->inDataNames[i0], 
                     (const char *)pSearchLayer->outDataNames[i2]) == 0)
          {
            pCurrentLayer->inData[i0].dataId = pSearchLayer->outData[i2].dataId;
            pSearchLayer->outConsumerLinked[i2]++;
          }
        }
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

/*==============================================================================
* Function purpose: similar to tidl_linkInputTensors, except that it tries to 
*   match the output of current layer to the input of other layers. 
*
* Note: - this function is a duplication of same function in tidl-import-utils.
*       - it will be removed when moving C code to tidl-import-utils.
==============================================================================*/
int32_t tidl_linkOutputTensors(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i0, i1, i2;
  for (i0 = 0; i0 < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs; i0++)
  {
    for (i1 = layerIndex - 1; i1 >= 0; i1--)
    {
      for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs; i2++)
      {
        if (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outConsumerLinked[i0] < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outConsumerCnt[i0])
        {
          if (strcmp((const char *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[i0], 
                     (const char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[i2]) == 0)
          {
            pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[i2].dataId = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[i0].dataId;
            pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outConsumerLinked[i0]++;
          }
        }
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_isAllInsAvailable(sTIDL_LayerPC_t    *orgLayer, 
                               sTIDL_OrgNetwork_t *ptempTIDLNetStructure, 
                               int32_t layerIndex)
{
  int32_t i0, i1, i2;
  int32_t status = 0;
  int32_t availableIns = 0;
  for (i0 = 0; i0 < orgLayer->numInBufs; i0++)
  {
    for (i1 = 0; i1 < layerIndex; i1++)
    {
      //printf("temp numOutBufs: %d\n", ptempTIDLNetStructure->TIDLPCLayers[i1].numOutBufs);
      for (i2 = 0; i2 < ptempTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
      {
        if (strcmp((const char *)ptempTIDLNetStructure->TIDLPCLayers[i1].outDataNames[i2], (const char *)orgLayer->inDataNames[i0]) == 0)
        {
          printf("i1 = %d, i2 = %d, i0 = %d\n", i1, i2, i0);
          availableIns++;
        }
      }
    }
  }
  /* Is shall be orgLayer->numInBufs <= availableIns, temprary fix to get caffe import working
     TODO : need rever back after migatration caffe to new import framework */

  if ((orgLayer->numInBufs <= availableIns) || (orgLayer->numInBufs == -1))
  {
    printf("numInBufs: %d, available: %d\n", orgLayer->numInBufs, availableIns);
    status = 1;
  }
  return(status);
}

extern sTIDL_OrgNetwork_t      orgTIDLNetStructure;

int32_t tidl_sortLayersInProcOrder(sTIDL_OrgNetwork_t *pOrgTIDLNetStructure, 
                                   sTIDL_OrgNetwork_t *ptempTIDLNetStructure, 
                                   int32_t numLayers)
{
  int32_t i, i0, i1, i2, status;
  int32_t newNetIdx, newNetIdxTemp;
  int32_t *isAddedToList = (int32_t *)my_malloc(numLayers*sizeof(int32_t));
  memset(isAddedToList, 0, sizeof(int32_t)*numLayers);

  status = TIDL_IMPORT_NO_ERR;

  /* Sort layers in processing order according to inputs and outputs of each layer */
  printf("Sorting layers in processing order...\n");
  printf("number of layers: %d\n", numLayers);
  //for(i=0; i<numLayers; i++)
  //{
  //  printf("Layer %d, numInBufs = %d\n", i, orgTIDLNetStructure.TIDLPCLayers[i].numInBufs);
  //}

  newNetIdx = 0;
  while (newNetIdx < numLayers)
  {
    /* For each layer, find the layer whose input is the output of this layer. */
    newNetIdxTemp = newNetIdx;
    for (i0 = 0; i0 < numLayers; i0++)
    {
      if (isAddedToList[i0] == 0)
      {
        printf("Number of inBuffs: %d\n", pOrgTIDLNetStructure->TIDLPCLayers[i0].numInBufs);
        if (tidl_isAllInsAvailable(&pOrgTIDLNetStructure->TIDLPCLayers[i0], 
                                   ptempTIDLNetStructure, newNetIdx))
        {
          printf("newNetIdx = %d, i0 = %d\n", newNetIdx, i0);
          ptempTIDLNetStructure->TIDLPCLayers[newNetIdx] = pOrgTIDLNetStructure->TIDLPCLayers[i0];
          newNetIdx++;
          isAddedToList[i0] = 1;
        }
      }
    }

    /* If no matching layer is found among all layers, this topology is not supported by TIDL */
    if(newNetIdx == newNetIdxTemp)
    {
      /* Return error: network topology not supported */
      printf("Error in sorting layers: matching layer cannot be found for index %d!\n", newNetIdxTemp);
      status = TIDL_IMPORT_ERR_NET_TOPOLOTY_NOT_SUPPORTED;
      break;
    }
  }

  my_free(isAddedToList);
  ptempTIDLNetStructure->numLayers = newNetIdx;
  printf("Number of layers after sorting is: %d\n", newNetIdx);

  // save network structure after sorting
  memset((void *)pOrgTIDLNetStructure, 0, sizeof(sTIDL_OrgNetwork_t));
  memcpy((void *)pOrgTIDLNetStructure, (void *)ptempTIDLNetStructure, sizeof(sTIDL_OrgNetwork_t));

  return status;
}

int32_t tidl_fillInDataLayerShape(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, tidlImpConfig * params, int32_t layerIndex)
{
  int32_t i, j, inDataIdx;
  int overWritefirstNode = 1;
  if ((params->inWidth == -1) || (params->inHeight == -1) || (params->inNumChannels == -1))
  {
    overWritefirstNode = 0;
  }
  inDataIdx = 0;
  for (i = 0; i < layerIndex; i++)
  {
    if ((pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_DataLayer) && (pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs > 0))
    {
      printf("Input data layer found at layer %d\n", i);
      pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[0] = 1;
      if (overWritefirstNode)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[1] = params->inNumChannels;
        pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[2] = params->inHeight;
        pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[3] = params->inWidth;
      }
      pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].elementType = params->inElementType;
      pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dataQ = params->inQuantFactor;

      inDataIdx++;
    }
  }
  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_upateAInDataId(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, int32_t oldId, int32_t currId)
{
  int32_t i0, i1, i2, i3, i4;
  for (i3 = 0; i3 < layerIndex; i3++)
  {
    for (i4 = 0; i4 < pOrgTIDLNetStructure->TIDLPCLayers[i3].numInBufs; i4++)
    {
      if (pOrgTIDLNetStructure->TIDLPCLayers[i3].inData[i4].dataId == oldId)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[i3].inData[i4].dataId = currId;
      }
    }
  }
  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_sortDataIds(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i0, i1, i2, i3, i4;
  int32_t maxDataId = 0;
  int32_t currId = 0;
  int32_t oldId = 0;

  printf("Sorting data ids ...\n");
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
    {
      maxDataId = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId >= maxDataId ? pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId : maxDataId;
    }
  }
  maxDataId = maxDataId + 1;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId += maxDataId;
    }
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs; i2++)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[i2].dataId += maxDataId;
    }
  }

  for (i1 = 0; i1 < layerIndex; i1++)
  {
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
    {
      oldId = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId;
      pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId = currId;
      tidl_upateAInDataId(pOrgTIDLNetStructure, layerIndex, oldId, currId);
      currId++;
    }
  }
  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_updateOutDataShape(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, 
                                int32_t startIdx, int32_t layerIndex, 
                                sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;

  printf("Updating out data shapes ...\n");

  for (i1 = startIdx; i1 < layerIndex; i1++)
  {
//    printf("In tidl_updateOutDataShape, i1 is %d\n", i1);
    status = sTIDL_tfOutRehapeTable[pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType].tidl_tfOutReshape(pOrgTIDLNetStructure, i1);
    if (status != -1)
    {
      for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
      {
        for (i3 = 0; i3 < layerIndex; i3++)
        {
          for (i4 = 0; i4 < pOrgTIDLNetStructure->TIDLPCLayers[i3].numInBufs; i4++)
          {
            if (pOrgTIDLNetStructure->TIDLPCLayers[i3].inData[i4].dataId == pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId)
            {
              pOrgTIDLNetStructure->TIDLPCLayers[i3].inData[i4] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2];
            }
          }

        }
      }
    }
  }
  return status;
}

int32_t tidl_getInLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, int32_t dataId)
{
  int32_t i1, i2;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs; i2++)
    {
      if (pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[i2].dataId == dataId)
      {
        return (i1);
      }
    }
  }
  return (-1);
}

int32_t tidl_getOutLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, int32_t dataId)
{
  int32_t i1, i2;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    for (i2 = 0; i2 < pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs; i2++)
    {
      if (pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[i2].dataId == dataId)
      {
        return (i1);
      }
    }
  }
  return (-1);
}

int32_t tidl_mergeBiasLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_BiasLayer)
    {
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      { /* Didn't find any layer whose output data ID is the same as this layer's input data ID.  */
        printf("Error in merging Bias layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_BIAS_NOT_MERGED;
      }
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if ((TIDLPCLayers->layerType == TIDL_ConvolutionLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
        if (TIDLPCLayers->layerParams.convParams.enableBias == 0)
        {
          TIDLPCLayers->layerParams.convParams.enableBias = 1;
          TIDLPCLayers->bias = pOrgTIDLNetStructure->TIDLPCLayers[i1].bias;
        }
        else
        {
          float * src = (float *)pOrgTIDLNetStructure->TIDLPCLayers[i1].bias.ptr;
          float * dst = (float *)TIDLPCLayers->bias.ptr;
          for (i2 = 0; i2 < TIDLPCLayers->bias.bufSize; i2++)
          {
            dst[i2] += src[i2];
          }
        }
      }
      else if((TIDLPCLayers->layerType == TIDL_InnerProductLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
        float * src = (float *)pOrgTIDLNetStructure->TIDLPCLayers[i1].bias.ptr;
        float * dst = (float *)TIDLPCLayers->bias.ptr;
        for (i2 = 0; i2 < TIDLPCLayers->bias.bufSize; i2++)
        {
          dst[i2] += src[i2];
        }
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergeBNLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_BatchNormLayer)
    {
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      {
        printf("Error in merging BatchNorm layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if ((TIDLPCLayers->layerType == TIDL_ConvolutionLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;

        if (TIDLPCLayers->layerParams.convParams.enableBias == 0)
        {
          TIDLPCLayers->layerParams.convParams.enableBias = 1;
          TIDLPCLayers->bias.ptr = my_malloc(sizeof(float)*TIDLPCLayers->outData[0].dimValues[1]);
          TIDLPCLayers->bias.bufSize = TIDLPCLayers->outData[0].dimValues[1];
          float * dst = (float *)TIDLPCLayers->bias.ptr;
          for (i2 = 0; i2 < TIDLPCLayers->bias.bufSize; i2++)
          {
            dst[i2] = 0;
          }
        }
        /* Merge BN scale and Bias to Conv2d */
        float * weights = (float *)TIDLPCLayers->weights.ptr;
        float * bias = (float *)TIDLPCLayers->bias.ptr;

        float * scale = (float *)pOrgTIDLNetStructure->TIDLPCLayers[i1].weights.ptr;
        float * bias2 = (float *)pOrgTIDLNetStructure->TIDLPCLayers[i1].bias.ptr;
        int32_t weightsSize = (TIDLPCLayers->weights.bufSize / TIDLPCLayers->bias.bufSize);
        for (i2 = 0; i2 < TIDLPCLayers->bias.bufSize; i2++)
        {
          for (i3 = 0; i3 < weightsSize; i3++)
          {
            weights[i2*weightsSize + i3] *= scale[i2];
          }
          bias[i2] = bias[i2] * scale[i2] + bias2[i2];
        }
        my_free(scale);
        my_free(bias2);
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergeReluLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  int32_t merged;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_ReLULayer)
    {
      merged = 0;
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      {
        printf("Error in merging ReLU layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if ((TIDLPCLayers->layerType == TIDL_ConvolutionLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->layerParams.convParams.enableRelU = 1;
        TIDLPCLayers->layerParams.convParams.reluParams.reluType = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.reluParams.reluType;
        merged = 1;
      }
      if ((TIDLPCLayers->layerType == TIDL_EltWiseLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->layerParams.eltWiseParams.enableRelU = 1;
        TIDLPCLayers->layerParams.eltWiseParams.reluParams.reluType = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.reluParams.reluType;
        merged = 1;
      }
      if ((TIDLPCLayers->layerType == TIDL_BatchNormLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->layerParams.batchNormParams.enableRelU = 1;
        TIDLPCLayers->layerParams.batchNormParams.reluParams.reluType = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.reluParams.reluType;
        merged = 1;
      }
      if ((TIDLPCLayers->layerType == TIDL_InnerProductLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1))
      {
        TIDLPCLayers->layerParams.innerProductParams.enableRelU = 1;
        TIDLPCLayers->layerParams.innerProductParams.reluParams.reluType = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.reluParams.reluType;
        merged = 1;
      }	  
      if (merged == 1)
      {
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergePadLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  int32_t padW, padH;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_PadLayer)
    {
      int32_t  inIdx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (inIdx == -1)
      {
        printf("Error in merging Pad layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
      int32_t  outIdx = tidl_getOutLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0].dataId);
      if (outIdx == -1)
      {
        printf("Error in merging Pad layer: could not find output layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        return TIDL_IMPORT_ERR_OUTPUT_LAYER_NOT_FOUND;
      }

      sTIDL_LayerPC_t *TIDLPCLayersIn  = &pOrgTIDLNetStructure->TIDLPCLayers[inIdx];
      sTIDL_LayerPC_t *TIDLPCLayersOut = &pOrgTIDLNetStructure->TIDLPCLayers[outIdx];

      if (gloab_data_format == 0)
      {
        padW = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerPCParams.padParams.padTensor[2 * 2 + 0];
        padH = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerPCParams.padParams.padTensor[1 * 2 + 0];
      }
      else
      {
        padW = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerPCParams.padParams.padTensor[3 * 2 + 0];
        padH = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerPCParams.padParams.padTensor[2 * 2 + 0];;
      }

      if ((TIDLPCLayersOut->layerType == TIDL_ConvolutionLayer) &&
        (pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0] == 1) &&
        (TIDLPCLayersIn->outConsumerCnt[0] == 1) &&
        (TIDLPCLayersOut->layerParams.convParams.strideW > 1) && 
        (TIDLPCLayersOut->layerParams.convParams.strideH > 1) &&
        ((TIDLPCLayersOut->layerParams.convParams.kernelW/2) == padW) &&
        ((TIDLPCLayersOut->layerParams.convParams.kernelH/2) == padH))
      {
        TIDLPCLayersIn->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;

        TIDLPCLayersOut->layerParams.convParams.padW = padW;
        TIDLPCLayersOut->layerParams.convParams.padH = padH;
        TIDLPCLayersOut->strideOffsetMethod = TIDL_strideOffsetTopLeft;

          //TIDLPCLayersIn->outData[0]        = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        TIDLPCLayersOut->inData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0];
        strcpy((char *)TIDLPCLayersOut->inDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
      }
      else
      {
        printf("TIDL limitation: Currently PAD layer is supported if the following layer is convolution with stride > 1.\n");
        return TIDL_IMPORT_ERR_PAD_NOT_MERGEABLE;
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergePoolLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if ((pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_PoolingLayer) && 
	    (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.poolParams.strideW == 2) &&
	    (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.poolParams.strideH == 2) &&
	    (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.poolParams.kernelW == 2) &&
	    (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.poolParams.kernelH == 2))
    {
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      {
        printf("Error in merging BatchNorm layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
	  
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if ((TIDLPCLayers->layerType == TIDL_ConvolutionLayer) &&
          (TIDLPCLayers->outConsumerCnt[0] == 1) &&
	      ((TIDLPCLayers->outData[0].dimValues[2] % 2) == 0) && 
          ((TIDLPCLayers->outData[0].dimValues[3] % 2) == 0))
      {		  
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;

        TIDLPCLayers->layerParams.convParams.enablePooling = 1; 	
        TIDLPCLayers->layerParams.convParams.poolParams.poolingType = pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.poolParams.poolingType;
        TIDLPCLayers->layerParams.convParams.poolParams.kernelW   = 2;
        TIDLPCLayers->layerParams.convParams.poolParams.kernelH   = 2;
        TIDLPCLayers->layerParams.convParams.poolParams.strideW   = 2;
        TIDLPCLayers->layerParams.convParams.poolParams.strideH   = 2;
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_removeMergedLayersFromNet(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, sTIDL_OrgNetwork_t  *ptempTIDLNetStructure, int32_t layerIndex)
{
  int32_t i0, i1, i2;
  int32_t newNetIdx = 0;

  for (i0 = 0; i0 < layerIndex; i0++)
  {
    if ((pOrgTIDLNetStructure->TIDLPCLayers[i0].numInBufs != -1) ||
      (pOrgTIDLNetStructure->TIDLPCLayers[i0].numOutBufs != -1))
    {
      // copy unmerged layers (both numInBufs and numOutBufs are set to -1 for merged layers) 
      ptempTIDLNetStructure->TIDLPCLayers[newNetIdx] = pOrgTIDLNetStructure->TIDLPCLayers[i0];
      newNetIdx++;
    }
  }
  ptempTIDLNetStructure->numLayers = newNetIdx;
  memset((void *)pOrgTIDLNetStructure, 0, sizeof(sTIDL_OrgNetwork_t));
  memcpy((void *)pOrgTIDLNetStructure, (void *)ptempTIDLNetStructure, sizeof(sTIDL_OrgNetwork_t));
  return TIDL_IMPORT_NO_ERR;
}


int32_t tidl_convertConv2DToIpLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable)
{
  int32_t i1, i2, i3, i4;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_ConvolutionLayer)
    {
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[i1];
      sTIDL_ConvParams_t *convParams = &pOrgTIDLNetStructure->TIDLPCLayers[i1].layerParams.convParams;
      if ((convParams->kernelW == 1) && (convParams->kernelH == 1) && (TIDLPCLayers->inData[0].dimValues[2] == 1) && (TIDLPCLayers->inData[0].dimValues[3] == 1))
      {
        int32_t  inIdx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
        if (inIdx == -1)
        {
          printf("Error in converting Conv2d layer to InnerProduct layer: could not find input layer: %s!\n",
                 pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
          return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
        }
        sTIDL_LayerPC_t *TIDLPCLayersIn = &pOrgTIDLNetStructure->TIDLPCLayers[inIdx];

        if ((TIDLPCLayersIn->layerType == TIDL_PoolingLayer) && (TIDLPCLayersIn->layerParams.poolParams.poolingType == TIDL_AveragePooling) && (TIDLPCLayersIn->outConsumerCnt[0] == 1))
        {
          sTIDL_LayerPC_t *TIDLPCLayersOut;
          int32_t  outIdx = tidl_getOutLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0].dataId);
          if (outIdx != -1)
          {
            TIDLPCLayersOut = &pOrgTIDLNetStructure->TIDLPCLayers[outIdx];
          }
          if ((outIdx == -1) || (TIDLPCLayersOut->layerType == TIDL_InnerProductLayer) ||
            (TIDLPCLayersOut->layerType == TIDL_SoftMaxLayer) || (TIDLPCLayersOut->layerType == TIDL_FlattenLayer) || (TIDLPCLayersOut->layerType == TIDL_ReshapeLayer))
          {
            TIDLPCLayersIn->layerParams.poolParams.kernelW = 0;
            TIDLPCLayersIn->layerParams.poolParams.kernelH = 0;

            sTIDL_LayerPC_t TIDLPCLayerstemp = pOrgTIDLNetStructure->TIDLPCLayers[i1];
            TIDLPCLayers->layerType = TIDL_InnerProductLayer;
            TIDLPCLayers->inData[0].dimValues[3] = TIDLPCLayers->inData[0].dimValues[1] * TIDLPCLayers->inData[0].dimValues[2] * TIDLPCLayers->inData[0].dimValues[3];
            TIDLPCLayers->inData[0].dimValues[2] = 1;
            TIDLPCLayers->inData[0].dimValues[1] = 1;
            TIDLPCLayersIn->outData[0] = TIDLPCLayers->inData[0];

            TIDLPCLayers->outData[0].dimValues[3] = TIDLPCLayers->outData[0].dimValues[1] * TIDLPCLayers->outData[0].dimValues[2] * TIDLPCLayers->outData[0].dimValues[3];
            TIDLPCLayers->outData[0].dimValues[2] = 1;
            TIDLPCLayers->outData[0].dimValues[1] = 1;
            TIDLPCLayersOut->inData[0] = TIDLPCLayers->outData[0];
            tidl_updateOutDataShape(pOrgTIDLNetStructure, outIdx, layerIndex, sTIDL_tfOutRehapeTable);

            TIDLPCLayers->layerParams.innerProductParams.numInNodes = TIDLPCLayers->inData[0].dimValues[3];
            TIDLPCLayers->layerParams.innerProductParams.numOutNodes = TIDLPCLayers->outData[0].dimValues[3];
            TIDLPCLayers->layerParams.innerProductParams.enableRelU = TIDLPCLayerstemp.layerParams.convParams.enableRelU;
          }
        }
      }
    }
  }
  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergeFlattenLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_FlattenLayer)
    {
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      {
        printf("Error in merging Flatten layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if (((TIDLPCLayers->layerType == TIDL_InnerProductLayer) &&
        (TIDLPCLayers->outConsumerCnt[0] == 1)) || 
		((TIDLPCLayers->layerType == TIDL_PoolingLayer) && (TIDLPCLayers->layerParams.poolParams.poolingType == TIDL_AveragePooling)))
      {
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

int32_t tidl_mergeReshapeLayer(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable)
{
  int32_t i1, i2, i3, i4;
  int32_t status = 0;
  for (i1 = 0; i1 < layerIndex; i1++)
  {
    if (pOrgTIDLNetStructure->TIDLPCLayers[i1].layerType == TIDL_ReshapeLayer)
    {
      int32_t  idx = tidl_getInLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].inData[0].dataId);
      if (idx == -1)
      {
        printf("Error in merging Reshape layer: could not find input layer: %s!\n",
               pOrgTIDLNetStructure->TIDLPCLayers[i1].inDataNames[0]);
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
      sTIDL_LayerPC_t *TIDLPCLayers = &pOrgTIDLNetStructure->TIDLPCLayers[idx];
      if ((TIDLPCLayers->layerType == TIDL_InnerProductLayer) ||
        ((TIDLPCLayers->layerType == TIDL_PoolingLayer) && (TIDLPCLayers->layerParams.poolParams.poolingType == TIDL_AveragePooling) && (TIDLPCLayers->outConsumerCnt[0] == 1)))
      {
        if (TIDLPCLayers->layerType == TIDL_PoolingLayer)
        {
          TIDLPCLayers->layerParams.poolParams.kernelW = 0;
          TIDLPCLayers->layerParams.poolParams.kernelH = 0;
        }
        TIDLPCLayers->numMacs += pOrgTIDLNetStructure->TIDLPCLayers[i1].numMacs;
        TIDLPCLayers->outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0];
        strcpy((char *)TIDLPCLayers->outDataNames[0], (char *)pOrgTIDLNetStructure->TIDLPCLayers[i1].outDataNames[0]);
        TIDLPCLayers->outConsumerCnt[0] = pOrgTIDLNetStructure->TIDLPCLayers[i1].outConsumerCnt[0];
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numInBufs = -1;
        pOrgTIDLNetStructure->TIDLPCLayers[i1].numOutBufs = -1;
        
        TIDLPCLayers->outData[0].dimValues[3] = TIDLPCLayers->outData[0].dimValues[1] * TIDLPCLayers->outData[0].dimValues[2] * TIDLPCLayers->outData[0].dimValues[3];
        TIDLPCLayers->outData[0].dimValues[2] = 1;
        TIDLPCLayers->outData[0].dimValues[1] = 1;
        sTIDL_LayerPC_t *TIDLPCLayersOut;
        int32_t  outIdx = tidl_getOutLayer(pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[i1].outData[0].dataId);
        if (outIdx != -1)
        {
          TIDLPCLayersOut = &pOrgTIDLNetStructure->TIDLPCLayers[outIdx];
          TIDLPCLayersOut->inData[0] = TIDLPCLayers->outData[0];
          tidl_updateOutDataShape(pOrgTIDLNetStructure, outIdx, layerIndex, sTIDL_tfOutRehapeTable);
        }
      }
      else {
        printf("TIDL limitation: Reshape layer cannot be merged with layers other than InnerProduct or AveragePooling layer!\n");
        return TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND;
      }
    }
  }

  return TIDL_IMPORT_NO_ERR;
}

void tidl_importEltWiseParams(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex)
{
    int32_t i;
    
    for(i=0; i<layerIndex; i++)
    {
      if(pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_EltWiseLayer) {
        pOrgTIDLNetStructure->TIDLPCLayers[i].layerParams.eltWiseParams.numChannels 
            = pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[1];
      }
    }
}

void TIDL_findRange(float * data, int32_t dataSize, float * minOut, float * maxOut, float scale)
{
  float min = FLT_MAX;
  float max = FLT_MIN;
  int32_t i;
  for (i = 0; i < dataSize; i++)
  {
    min = ((data[i] * scale) < min) ? (data[i] * scale) : min;
    max = ((data[i] * scale) > max) ? (data[i] * scale) : max;
  }
  *minOut = (min < *minOut) ? min : *minOut;
  *maxOut = (max > *maxOut) ? max : *maxOut;
}

int32_t TIDL_normalize(float data, float min, float max)
{
  int32_t param;
  if (max == min)
  {
    if (min)
    {
      min = min * 0.5;
    }
    else
    {
      min = -1;
    }
  }
  float absRange = abs(max - min);
  float quantPrec = ((1.0*(1 << NUM_BIAS_BITS)) / absRange);
  if ((quantPrec * 256) > INT32_MAX)
  {
    quantPrec = INT32_MAX / 256;
  }
  if(data  > 0)
  {
    param = (data *  quantPrec + QUAN_STYLE2_ROUND);
  }
  else
  {
    param = (data *  quantPrec - QUAN_STYLE2_ROUND);
  }


  return param;
}

int64_t TIDL_roundSat(int64_t val, uint8_t bits, int64_t min, int64_t max)
{
  if (bits > 0)
  {
    val += (1U << (bits - 1U));
    val >>= bits;
  }
  val = val < min ? min : val;
  val = val > max ? max : val;

  return val;
}

void TIDL_convertSbuff(sBuffer_t * sBuffDst, sBufferPc_t *sBuffSrc)
{
  RESET_PTR(sBuffDst->ptr);  // ptr is not used by TIDL lib 
  sBuffDst->bufSize = sBuffSrc->bufSize;  
}


int32_t TIDL_QuantizeUnsignedMax(uint8_t * params, float * data, int32_t dataSize, float min, float max, int32_t numBits, int32_t weightsElementSizeInBits, int32_t * zeroWeightValue)
{
  int32_t i;
  if (max == min)
  {
    if (min)
    {
      min = min*0.5;
    }
    else
    {
      min = -1;
    }
  }
  float absRange = abs(max - min);

  float quantPrec = ((1.0*(1 << numBits)) / absRange);
  float pData;
  int32_t param;

  int32_t offset;
  if ((quantPrec * 256) > INT32_MAX)
  {
    quantPrec = INT32_MAX / 256;
  }
  if(min  > 0)
  {
    offset = (min *  quantPrec + QUAN_STYLE2_ROUND);
  }
  else
  {
    offset = (min *  quantPrec - QUAN_STYLE2_ROUND);
  }

  //Convert float params to 8-bit or 16-bit
  if(weightsElementSizeInBits <= 8)
  {
    for(i = 0; i < dataSize; i++)
    {
      pData = data[i]; 
      if(pData  > 0)
      {
        param = (pData *  quantPrec + QUAN_STYLE2_ROUND);
      }
      else
      {
        param = (pData *  quantPrec - QUAN_STYLE2_ROUND);
      }
      param = param - offset;

      params[i] = param > ((1 << weightsElementSizeInBits) - 1) ? ((1 << weightsElementSizeInBits) - 1) : param;
    }
  }
  else
  {
	uint16_t *params16 = (uint16_t *)params;

  for(i = 0; i < dataSize; i++)
  {
    pData = data[i]; 
    if(pData  > 0)
    {
      param = (pData *  quantPrec + QUAN_STYLE2_ROUND);
    }
    else
    {
      param = (pData *  quantPrec - QUAN_STYLE2_ROUND);
    }
    param = param - offset;

      params16[i] = param > ((1 << weightsElementSizeInBits) - 1) ? ((1 << weightsElementSizeInBits) - 1) : param;
    }
  }
  *zeroWeightValue = -offset;
  return ((int32_t)(quantPrec*256));
}

// Quantize and write parameters out to file
void TIDL_importQuantWriteLayerParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
                                      int32_t              numLayers,
                                      FILE                 *fp1
                                     )
{
  int32_t zeroWeightValue, i;
  sTIDL_LayerPC_t *pTIDLPCLayers;

  for(i=0; i<numLayers; i++)
  {
    pTIDLPCLayers = &(pOrgTIDLNetStructure->TIDLPCLayers[i]);

    if((pTIDLPCLayers->layerType == TIDL_ConvolutionLayer)  ||
       (pTIDLPCLayers->layerType == TIDL_InnerProductLayer) ||
       (pTIDLPCLayers->layerType == TIDL_BatchNormLayer))
    {
      float min = FLT_MAX;
      float max = FLT_MIN;
      int32_t weightsElementSizeInBits = pTIDLPCLayers->weightsElementSizeInBits;

      if(pTIDLPCLayers->layerType == TIDL_ConvolutionLayer)
      {
        float *  data     = (float *)pTIDLPCLayers->weights.ptr;
        uint32_t dataSize = pTIDLPCLayers->weights.bufSize;
        uint8_t * params  = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits-1)/8 + 1));
        TIDL_findRange(data, dataSize, &min , &max, 1.0);

        pTIDLPCLayers->layerParams.convParams.weightsQ = 
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data,dataSize, min , max, 
                                   NUM_WHGT_BITS, weightsElementSizeInBits, &zeroWeightValue);
        pTIDLPCLayers->layerParams.convParams.zeroWeightValue = zeroWeightValue;

        if(weightsElementSizeInBits > 8)
          fwrite(params,2,dataSize,fp1);
        else
          fwrite(params,1,dataSize,fp1);

        free(params);
        free(data);

        // bufSize is not needed by calibration - below is from ti_dl\test\src\tidl_tb.c:
        //   dataSize = (conv2dPrms->numInChannels* conv2dPrms->numOutChannels* 
        //              conv2dPrms->kernelW*conv2dPrms->kernelH)/conv2dPrms->numGroups;
        //   FREAD((int8_t *)conv2dPrms->weights.ptr,1,2*dataSize, fp1);
        pTIDLPCLayers->weights.ptr = NULL;
        pTIDLPCLayers->weights.bufSize = 0;

        if(pTIDLPCLayers->layerParams.convParams.enableBias)
        {
          min = FLT_MAX;
          max = FLT_MIN;
          {
            float * biasData = (float *)pTIDLPCLayers->bias.ptr;
            uint32_t biasDataSize = pTIDLPCLayers->bias.bufSize;
            TIDL_findRange(biasData, biasDataSize, &min, &max, 1.0);
          }

          data = (float *)pTIDLPCLayers->bias.ptr;
          dataSize = pTIDLPCLayers->bias.bufSize;
          max = abs(min) >  abs(max) ? abs(min) : abs(max);
          pTIDLPCLayers->layerParams.convParams.biasQ =
            TIDL_QuantizeUnsignedMax(0, 0, 0, 0, max, NUM_BIAS_BITS, (NUM_BIAS_BYTES * 8), &zeroWeightValue);

          int16_t * params = (int16_t *)malloc(dataSize*NUM_BIAS_BYTES);
          for (int idx = 0; idx < dataSize; idx++) 
          {
            int32_t biasParam = TIDL_normalize(data[idx], 0 , max);
            params[idx] = (int16_t)TIDL_roundSat(biasParam,0,SHRT_MIN,SHRT_MAX);
          }
          fwrite(params, NUM_BIAS_BYTES,dataSize,fp1);
          free(params);
          free(data);
          pTIDLPCLayers->bias.ptr = NULL;
          pTIDLPCLayers->bias.bufSize = 0;
        }
        if (pTIDLPCLayers->layerParams.convParams.biasQ == 0)
        {
          pTIDLPCLayers->layerParams.convParams.biasQ = 1;
        }
      }
      else if(pTIDLPCLayers->layerType == TIDL_InnerProductLayer)
      {
        float *  data     = (float *)pTIDLPCLayers->weights.ptr;
        uint32_t dataSize = pTIDLPCLayers->weights.bufSize;
        uint8_t * params = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits-1)/8 + 1));
        TIDL_findRange(data, dataSize, &min , &max, 1.0);
        pTIDLPCLayers->layerParams.innerProductParams.weightsQ = 
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data,dataSize, min , max,  NUM_WHGT_BITS, weightsElementSizeInBits, &zeroWeightValue);
        pTIDLPCLayers->layerParams.innerProductParams.zeroWeightValue = zeroWeightValue;

        if(weightsElementSizeInBits > 8)
          fwrite(params,2,dataSize,fp1);
        else
          fwrite(params,1,dataSize,fp1);

        free(params);
        free(data);
        pTIDLPCLayers->weights.ptr = NULL;
        pTIDLPCLayers->weights.bufSize = 0;

        min = FLT_MAX;
        max = FLT_MIN;

        {
          float * biasData = (float *)(pTIDLPCLayers->bias.ptr);
          uint32_t biasDataSize = pTIDLPCLayers->bias.bufSize;
          TIDL_findRange(biasData, biasDataSize, &min, &max, 1.0 );
        }

        max = abs(min) >  abs(max) ? abs(min) : abs(max);
        pTIDLPCLayers->layerParams.innerProductParams.biasQ =
          TIDL_QuantizeUnsignedMax(0, 0, 0, 0, max, NUM_BIAS_BITS, (NUM_BIAS_BYTES * 8), &zeroWeightValue);

        data     = (float *)pTIDLPCLayers->bias.ptr;
        dataSize = pTIDLPCLayers->bias.bufSize;
        {
          int16_t *params = (int16_t *)malloc(dataSize*NUM_BIAS_BYTES);
          for (int idx = 0; idx < dataSize; idx++) 
          {
            int32_t biasParam = TIDL_normalize(data[idx], 0 , max);
            params[idx] = (int16_t)TIDL_roundSat(biasParam,0,SHRT_MIN,SHRT_MAX);
          }
          fwrite(params, NUM_BIAS_BYTES,dataSize,fp1);
          free(params);
        }
        free(data);
        pTIDLPCLayers->bias.ptr = NULL;
        pTIDLPCLayers->bias.bufSize = 0;

        if (pTIDLPCLayers->layerParams.innerProductParams.biasQ == 0)
        {
          pTIDLPCLayers->layerParams.innerProductParams.biasQ = 1;
        }
      }
      else if (pTIDLPCLayers->layerType == TIDL_BatchNormLayer)
      {
        float *  data = (float *)pTIDLPCLayers->weights.ptr;
        uint32_t dataSize = pTIDLPCLayers->weights.bufSize;
        uint8_t * params = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits - 1) / 8 + 1));
        TIDL_findRange(data, dataSize, &min, &max, 1.0);

        pTIDLPCLayers->layerParams.batchNormParams.weightsQ =
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data, dataSize, min, max, NUM_WHGT_BITS, weightsElementSizeInBits, &zeroWeightValue);
        pTIDLPCLayers->layerParams.batchNormParams.zeroWeightValue = zeroWeightValue;

        if (weightsElementSizeInBits > 8)
          fwrite(params, 2, dataSize, fp1);
        else
          fwrite(params, 1, dataSize, fp1);

        free(params);
        free(data);
        pTIDLPCLayers->weights.ptr = NULL;
        pTIDLPCLayers->weights.bufSize = 0;

        {
          min = FLT_MAX;
          max = FLT_MIN;
          {
            float * biasData = (float *)pTIDLPCLayers->bias.ptr;
            uint32_t biasDataSize = pTIDLPCLayers->bias.bufSize;
            TIDL_findRange(biasData, biasDataSize, &min, &max, 1.0);
          }

          data = (float *)pTIDLPCLayers->bias.ptr;
          dataSize = pTIDLPCLayers->bias.bufSize;
          max = abs(min) > abs(max) ? abs(min) : abs(max);
          pTIDLPCLayers->layerParams.batchNormParams.biasQ =
            TIDL_QuantizeUnsignedMax(0, 0, 0, 0, max, NUM_BIAS_BITS, (NUM_BIAS_BYTES * 8), &zeroWeightValue);

          int16_t * params = (int16_t *)malloc(dataSize*NUM_BIAS_BYTES);
          for (int idx = 0; idx < dataSize; idx++)
          {
            int32_t biasParam = TIDL_normalize(data[idx], 0, max);
            params[idx] = (int16_t)TIDL_roundSat(biasParam, 0, SHRT_MIN, SHRT_MAX);
          }
          fwrite(params, NUM_BIAS_BYTES, dataSize, fp1);
          free(params);
          free(data);
          if (pTIDLPCLayers->layerParams.batchNormParams.biasQ == 0)
          {
            pTIDLPCLayers->layerParams.batchNormParams.biasQ = 1;
          }
          pTIDLPCLayers->bias.ptr = NULL;
          pTIDLPCLayers->bias.bufSize = 0;
        }

        if (pTIDLPCLayers->layerParams.batchNormParams.reluParams.reluType == TIDL_PRelU)
        {
          float * slopeData = (float *)pTIDLPCLayers->slope.ptr;
          uint32_t slopeDataSize = pTIDLPCLayers->slope.bufSize;
          uint8_t * params = (uint8_t *)malloc(slopeDataSize * ((weightsElementSizeInBits - 1) / 8 + 1));
          float min = FLT_MAX;
          float max = FLT_MIN;
          TIDL_findRange(slopeData, slopeDataSize, &min, &max, (1.0));
          pTIDLPCLayers->layerParams.batchNormParams.reluParams.slopeQ =
            TIDL_QuantizeUnsignedMax((uint8_t *)params, slopeData, slopeDataSize, min, max, NUM_WHGT_BITS, weightsElementSizeInBits, &zeroWeightValue);
          pTIDLPCLayers->layerParams.batchNormParams.reluParams.slopeQ /= 256;
          pTIDLPCLayers->layerParams.batchNormParams.reluParams.zeroSlopeValue = zeroWeightValue;

          if (weightsElementSizeInBits > 8)
          {
            fwrite(params, 2, slopeDataSize, fp1);
          }
          else
          {
            fwrite(params, 1, slopeDataSize, fp1);
          }
          free(params);
          free(slopeData);
          pTIDLPCLayers->slope.ptr = NULL;
          pTIDLPCLayers->slope.bufSize = 0;
          if (pTIDLPCLayers->layerParams.batchNormParams.reluParams.slopeQ == 0)
          {
            pTIDLPCLayers->layerParams.batchNormParams.reluParams.slopeQ = 1;
          }
        }
      }
    }

  }  // for(i=0; i<numLayers; i++)
} //TIDL_importQuantWriteLayerParams

const char * TIDL_LayerString[] = 
{
"TIDL_DataLayer            ",
"TIDL_ConvolutionLayer     ",
"TIDL_PoolingLayer         ",
"TIDL_ReLULayer            ",
"TIDL_PReLULayer           ",
"TIDL_EltWiseLayer         ",
"TIDL_InnerProductLayer    ",
"TIDL_SoftMaxLayer         ",
"TIDL_BatchNormLayer       ",
"TIDL_BiasLayer            ",
"TIDL_ScaleLayer           ",
"TIDL_Deconv2DLayer        ",
"TIDL_ConcatLayer          ",
"TIDL_SplitLayer           ",
"TIDL_SliceLayer           ",
"TIDL_CropLayer            ",
"TIDL_FlattenLayer         ",
"TIDL_DropOutLayer         ",
"TIDL_ArgMaxLayer          ",
"TIDL_DetectionOutputLayer ",
"TIDL_UnSuportedLayer",
"TIDL_ConstDataLayer",
"TIDL_ShuffleChannelLayer",
"TIDL_ResizeLayer",
"TIDL_PriorBoxLayer",
"TIDL_PermuteLayer",
"TIDL_ReshapeLayer",
"TIDL_ShapeLayer",
"TIDL_SqueezeLayer",
"TIDL_PadLayer",
"TIDL_TransposeLayer",
};

int32_t tidl_copyPCNetToDeviceNet(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, sTIDL_Network_t  *tIDLNetStructure, int32_t layerIndex, int weightsElementSizeInBits)
{
  int32_t i, j;
  int64_t                    totalMacs = 0;
  int32_t tiLayerIndex = 0;
  printf("\nNum of Layer Detected : %3d \n", layerIndex);
  FILE *layerInfoFile = fopen(LAYER_INFO_FILENAME, "w");

  tIDLNetStructure->dataElementSize = 1;
  tIDLNetStructure->biasElementSize = 2;
  tIDLNetStructure->weightsElementSize = ((weightsElementSizeInBits-1)/8 + 1);
  tIDLNetStructure->slopeElementSize = tIDLNetStructure->weightsElementSize;
  tIDLNetStructure->interElementSize = 1;
  tIDLNetStructure->quantizationStyle = gParams.quantizationStyle;

  printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  printf("%5s|%-30s|%-50s|%-6s|%-6s|%-6s|%-32s|%-10s|%-36s|%-36s|%-11s|\n", "Num", "TIDL Layer Name", "Out Data Name", "Group", "#Ins", "#Outs", "Inbuf Ids", "Outbuf Id", "In NCHW", "Out NCHW", "MACS");
  printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  for (i = 0; i < layerIndex; i++)
  {
    if ((pOrgTIDLNetStructure->TIDLPCLayers[i].layerType != TIDL_UnSuportedLayer) &&
      (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType != TIDL_ConstDataLayer))
    {
      tIDLNetStructure->TIDLLayers[tiLayerIndex].layerType = pOrgTIDLNetStructure->TIDLPCLayers[i].layerType;
      tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams = pOrgTIDLNetStructure->TIDLPCLayers[i].layerParams;
      tIDLNetStructure->TIDLLayers[tiLayerIndex].numInBufs = pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs;
      tIDLNetStructure->TIDLLayers[tiLayerIndex].numOutBufs = pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs;
      tIDLNetStructure->TIDLLayers[tiLayerIndex].weightsElementSizeInBits = pOrgTIDLNetStructure->TIDLPCLayers[i].weightsElementSizeInBits;
      if ((gParams.modelType == 2) || (gParams.modelType == 0))
      {
        tIDLNetStructure->TIDLLayers[tiLayerIndex].strideOffsetMethod = TIDL_strideOffsetTopLeft;
      }
      else
      {
        tIDLNetStructure->TIDLLayers[tiLayerIndex].strideOffsetMethod = pOrgTIDLNetStructure->TIDLPCLayers[i].strideOffsetMethod;
      }

      if (tIDLNetStructure->TIDLLayers[tiLayerIndex].layerType == TIDL_DataLayer)
      {
        tIDLNetStructure->TIDLLayers[tiLayerIndex].layersGroupId = 0;
      }
      else
      {
        tIDLNetStructure->TIDLLayers[tiLayerIndex].coreID             = gParams.layersGroupId[tiLayerIndex];
	    tIDLNetStructure->TIDLLayers[tiLayerIndex].layersGroupId      = gParams.layersGroupId[tiLayerIndex];
      }

      // Copy sBuff_t information from PC Network to device network
      if ((pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer) ||
        (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_Deconv2DLayer))
      {
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.convParams.weights, &pOrgTIDLNetStructure->TIDLPCLayers[i].weights);
        if (pOrgTIDLNetStructure->TIDLPCLayers[i].layerParams.convParams.enableBias)
        {
          TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.convParams.bias, &pOrgTIDLNetStructure->TIDLPCLayers[i].bias);
        }
      }
      else if (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_InnerProductLayer)
      {
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.innerProductParams.weights, &pOrgTIDLNetStructure->TIDLPCLayers[i].weights);
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.innerProductParams.bias, &pOrgTIDLNetStructure->TIDLPCLayers[i].bias);
      }
      else if (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_BatchNormLayer)
      {
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.batchNormParams.weights, &pOrgTIDLNetStructure->TIDLPCLayers[i].weights);
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.batchNormParams.bias, &pOrgTIDLNetStructure->TIDLPCLayers[i].bias);

        if (pOrgTIDLNetStructure->TIDLPCLayers[i].layerParams.batchNormParams.reluParams.reluType == TIDL_PRelU)
        {
          TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.batchNormParams.reluParams.slope, &pOrgTIDLNetStructure->TIDLPCLayers[i].slope);
        }
      }
      else if (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_DetectionOutputLayer)
      {
        TIDL_convertSbuff(&tIDLNetStructure->TIDLLayers[tiLayerIndex].layerParams.detectOutParams.priorBox, &pOrgTIDLNetStructure->TIDLPCLayers[i].priorBox);
      }

      printf("%5d|%-30s|", tiLayerIndex, TIDL_LayerString[pOrgTIDLNetStructure->TIDLPCLayers[i].layerType]);
      if (strlen((const char *)pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[0]) > 50)
      {
        printf("%-50s|", &pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[0][strlen((const char *)pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[0]) - 50]);
      }
      else
      {
        printf("%-50s|", pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[0]);
      }
      fprintf(layerInfoFile, "%d %s \n", pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dataId,
        pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[0]);

      printf("%6d|%6d|%6d|", tIDLNetStructure->TIDLLayers[tiLayerIndex].layersGroupId, pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs, pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs);

      for (j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs; j++)
      {
        printf("%3d ", pOrgTIDLNetStructure->TIDLPCLayers[i].inData[j].dataId);
        tIDLNetStructure->TIDLLayers[tiLayerIndex].inData[j] = pOrgTIDLNetStructure->TIDLPCLayers[i].inData[j];

      }
      for (j = (pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs > 0 ? pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs : 0); j < 8; j++)
      {
        printf("  x ");
      }
      printf("|");
      printf("%3d ", pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dataId);
      printf("      |");
      for (j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs; j++)
      {
        tIDLNetStructure->TIDLLayers[tiLayerIndex].outData[j] = pOrgTIDLNetStructure->TIDLPCLayers[i].outData[j];

      }
      for (j = 0; j < TIDL_DIM_MAX; j++)
      {
        printf("%8d ", pOrgTIDLNetStructure->TIDLPCLayers[i].inData[0].dimValues[j]);
      }
      printf("|");

      for (j = 0; j < TIDL_DIM_MAX; j++)
      {
        printf("%8d ", pOrgTIDLNetStructure->TIDLPCLayers[i].outData[0].dimValues[j]);
      }
      printf("|");
#ifdef PLATFORM_64BIT
      printf("%10ld |", pOrgTIDLNetStructure->TIDLPCLayers[i].numMacs);
#else
      printf("%10lld |", pOrgTIDLNetStructure->TIDLPCLayers[i].numMacs);
#endif
      totalMacs += pOrgTIDLNetStructure->TIDLPCLayers[i].numMacs;
      printf("\n");
      tiLayerIndex++;
    }

  }
  fclose(layerInfoFile);
  printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  printf("Total Giga Macs : %4.4f\n", ((float)totalMacs / 1000000000));
  printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  return tiLayerIndex;
}

/* Define strings for layers that can be merged into TIDL layers when requirements
   are met. The corresponding layer types are defined in ti_dl.h. */
const char * TIDL_Unsupported_Layers[] = 
{
"ShuffleChannel Layer" ,  // TIDL_ShuffleChannelLayer
"Resize Layer" ,          // TIDL_ResizeLayer
"PriorBox Layer" ,        // TIDL_PriorBoxLayer
"Permute Layer" ,         // TIDL_PermuteLayer
"Reshape Layer" ,         // TIDL_ReshapeLayer
"Shape Layer" ,           // TIDL_ShapeLayer
"Squeeze Layer" ,         // TIDL_SqueezeLayer
"Pad Layer" ,             // TIDL_PadLayer
"Transpose Layer",        // TIDL_TransposeLayer
};

/*==============================================================================
* Function tidl_countUnsupportedLayers()
*
*  For a given network model, it may have layers that TIDL implemented, and layers
*  that TIDL has not implemented yet. For layers not implemented by TIDL, some can
*  be coalesced into TIDL layers under certain conditions. These layers are defined
*  in ti_dl.h and also listed in table TIDL_Unsupported_Layers.
*
*  These layers, even though not implemented by TIDL, are still mapped to dummy
*  TIDL layers (defined in ti_dl.h) during the parsing stage of the import. Then
*  the import process will try to coalesce (merge) them into layers implemented
*  by TIDL (defined in itidl_ti.h). However, there are limitations on coalescing
*  these unsupported layers into TIDL layers. If these layers cannot be coalesced
*  into TIDL layers, it should be caught during the import process.
*
*  This function scans through a provided network structure and count how many
*  layers are not supported by TIDL and not able to be merged into TIDL layers.
*  It prints out each unsupported layer name and returns total number of unsupported
*  layers. 
==============================================================================*/
int32_t tidl_countUnsupportedLayers(sTIDL_Network_t *pNetStructure, 
                                    int32_t numLayers)
{
  int32_t i, layerType, numUnsupportedLayers;

  numUnsupportedLayers = 0;
  for(i=0; i<numLayers; i++)
  {
    layerType = pNetStructure->TIDLLayers[i].layerType;
    if(  (layerType == TIDL_ShuffleChannelLayer)
       ||(layerType == TIDL_ResizeLayer)
       ||(layerType == TIDL_PriorBoxLayer)
       ||(layerType == TIDL_PermuteLayer)
       ||(layerType == TIDL_ReshapeLayer)
       ||(layerType == TIDL_ShapeLayer)
       ||(layerType == TIDL_SqueezeLayer)
       ||(layerType == TIDL_PadLayer)
       ||(layerType == TIDL_TransposeLayer) )
    {
      printf("%s is not supported by TIDL and cannot be merged into any TIDL layer.\n", 
             TIDL_Unsupported_Layers[layerType-TIDL_ShuffleChannelLayer]);
      numUnsupportedLayers++;
    }
  }

  return numUnsupportedLayers;
} /* tidl_countUnsupportedLayers */

/*==============================================================================
 * Function TIDL_setConv2dKernelType():
 *   Set conv2d kernel type for each convolution layer:
 *     - If kernel type is not specified through import config file, it will be
 *       set to optimal type. This is the default. 
 *     - If kernel type is explicitly specified through import config file, the 
 *       specified type will be used if conditions are met. 
==============================================================================*/
void TIDL_setConv2dKernelType(sTIDL_Network_t *pTIDLNetStructure, int32_t tiLayerIndex)
{
  int layerIndex;
  int numLayersFixedType = 0;

  for (layerIndex = 0; layerIndex < tiLayerIndex; layerIndex++) 
  {
    if(pTIDLNetStructure->TIDLLayers[layerIndex].layerType ==  TIDL_ConvolutionLayer)
    {
      if(gParams.conv2dKernelType[layerIndex] == -1) // kernel type set to -1 by default
      {
        // Kernel type is not explicitly configured - set it to optimal type:
        //   use dense conv2d if:
        //      - kernel size is 1xN or 3x3,
        //      - stride is 1, and 
        //      - kernel layer size < 64.
		//   use sparse conv2d otherwise.
        if( (  ((pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelW == 1 )) ||
               ((pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelW == 3 ) &&
                (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelH == 3 )) 
            ) &&
            (  (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.strideW == 1 ) &&
               (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.strideH == 1 )
            ) &&
            (  (pTIDLNetStructure->TIDLLayers[layerIndex].outData[0].dimValues[TIDL_DIM_HEIGHT] < 64) ||
               (pTIDLNetStructure->TIDLLayers[layerIndex].outData[0].dimValues[TIDL_DIM_WIDTH]  < 64) 
            )
          )
        {
          pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelType = TIDL_dense;
        }
        else 
        {
          pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelType = TIDL_sparse;
        }
      }
      else
      {
        // Kernel type is explicitly configured - use the configured type:
		//    - if it is configured to sparse, then use sparse,
		//    - if it is configured to dense, then use dense if conditions are met.
        if(gParams.conv2dKernelType[layerIndex] == 0)
        {
          pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelType = TIDL_sparse;
        }
        else if(gParams.conv2dKernelType[layerIndex] == 1)
        {
          if((  ((pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelW == 1 )) ||
                ((pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelW == 3 ) &&
                 (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelH == 3 )) 
             ) &&
             (  (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.strideW == 1 ) &&
                (pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.strideH == 1 )
             )
            )
          {
            pTIDLNetStructure->TIDLLayers[layerIndex].layerParams.convParams.kernelType = TIDL_dense;
          }
        }
        numLayersFixedType++;
      }
    }
  }

  // If any layer is explicitly configured to have dense or sparse type through 
  // the import config file, print warning message. 
  if(numLayersFixedType >0)
  {
    printf("\nWarning - conv2D kernel type is explicitly specified by the user. " 
           " TIDL performance may not be optimal!\n");
  }
}

int32_t TIDL_isDataBufUsed(int32_t           dataId,
                           sTIDL_Network_t   *pTIDLNetStructure,
                           int32_t           numLayer)
{
  int32_t i, j;
  for (i = 0; i < numLayer; i++)
  {
    for (j = 0; j < pTIDLNetStructure->TIDLLayers[i].numInBufs; j++)
    {
      if (pTIDLNetStructure->TIDLLayers[i].inData[j].dataId == dataId)
      {
        return 1;
      }
    }
  }
  return 0;
}

int32_t tidl_addOutDataLayer(sTIDL_Network_t  *tIDLNetStructure, int32_t tiLayerIndex)
{
  int32_t i, j;

  if(tIDLNetStructure->TIDLLayers[tiLayerIndex-1].layerType == TIDL_DataLayer)
  { // if last layer is already DataLayer, overwrite it
    //printf("Last layer is already data layer.\n");
    tiLayerIndex -= 1;
  }

  tIDLNetStructure->TIDLLayers[tiLayerIndex].layerType = TIDL_DataLayer;
  tIDLNetStructure->TIDLLayers[tiLayerIndex].numInBufs = 0;
  tIDLNetStructure->TIDLLayers[tiLayerIndex].numOutBufs = -1;
  tIDLNetStructure->TIDLLayers[tiLayerIndex].coreID = 255;

  for (i = 0; i < tiLayerIndex; i++)
  {
    if (tIDLNetStructure->TIDLLayers[i].layerType != TIDL_DataLayer)
    {
      for (j = 0; j < tIDLNetStructure->TIDLLayers[i].numOutBufs; j++)
      {
        if (!TIDL_isDataBufUsed(tIDLNetStructure->TIDLLayers[i].outData[j].dataId, tIDLNetStructure, tiLayerIndex))
        {
          tIDLNetStructure->TIDLLayers[tiLayerIndex].inData[tIDLNetStructure->TIDLLayers[tiLayerIndex].numInBufs] = tIDLNetStructure->TIDLLayers[i].outData[j];
          tIDLNetStructure->TIDLLayers[tiLayerIndex].numInBufs++;
        }
      }
    }
  }
  tIDLNetStructure->numLayers = tiLayerIndex + 1;
  return TIDL_IMPORT_NO_ERR;
}

