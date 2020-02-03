#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ti_dl.h"

/*==============================================================================
* Function purpose: for current layer, search from all mapped layers and find those
*   whose output is the input of current layer. Once a match is found, it will 
*   link the two layers: assign output data id of the found layer to input data 
*   id of current layer. 
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
  return 0;
}

/*==============================================================================
* Function purpose: similar to tidl_linkInputTensors, except that it tries to 
*   match the output of current layer to the input of other layers. 
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
  return 0;
}

