/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


/* This file provides simplified API for using DmaUtils
   in TVM+TIDL generated code */

#include <stdio.h>
#include "tvm_tidl_dmautils.h"
#include "tidl_api_mem.h"

/* Include path is correctly decided based on SDK used for build as part of the Cmake files */
#include <dmautils_autoincrement_3d.h>

//#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT(...)

#ifndef HOST_EMULATION
void *getUDMADrvObjPtr()
{
  extern void *appUdmaGetObj(void);
  return appUdmaGetObj();
}
#endif

/**
 * @brief Allocate and initialize dmaUtilsContext, allocate TrMem for each
 *     channel
 *
 * @param num_channels number of DMA channels
 *     each DMA/double-buffered tensor needs a channel
 * @param pTrMem_chs returns allocated TrMem for each channel
 * @return allcoated and initialized dmaUtilsContext
*/
uint8_t *tvm_tidl_dmautils_init(int32_t num_channels, uint8_t *pTrMem_chs[])
{
  int32_t retVal = DMAUTILS_SOK;
  DmaUtilsAutoInc3d_InitParam initParams;
  DmaUtilsAutoInc3d_ChannelInitParam chInitParams[TVMTIDL_DMAUTILS_CHANNEL_MAX];

  /* data structure preferred to be placed in faster on-chip memory */
  /* size 0xa80 for 2 channels */
  uint8_t *dmaUtilsContext = tvm_tidl_l2_scratch_alloc(
                               DmaUtilsAutoInc3d_getContextSize(num_channels));
  for (int32_t ch = 0; ch < num_channels; ch++)
  {
    /* size 0x80 per channel */
    pTrMem_chs[ch] = tvm_tidl_l2_scratch_alloc(
                                             DmaUtilsAutoInc3d_getTrMemReq(1));
  }

  initParams.contextSize = DmaUtilsAutoInc3d_getContextSize(num_channels);
  initParams.numChannels = num_channels;
  initParams.traceLogLevel   = 1;  // TODO: change to 0 later
  initParams.udmaDrvHandle   = (void *) getUDMADrvObjPtr();
  initParams.DmaUtilsVprintf = vprintf;

  for (int32_t ch = 0; ch < num_channels; ch++)
  {
    chInitParams[ch].dmaQueNo = 0;
    chInitParams[ch].druOwner = DMAUTILSAUTOINC3D_DRUOWNER_DIRECT_TR;
  }

  retVal = DmaUtilsAutoInc3d_init(dmaUtilsContext, &initParams, chInitParams);
  if (retVal != DMAUTILS_SOK)
  {
    tvm_tidl_l2_scratch_reset();
    return NULL;
  }

  return dmaUtilsContext;
}

/**
 * @brief Configure channel for DmaUtilsAutoInc3d transfer
 *
 * @param dmaUtilsContext
 * @param num_channels number of DMA channels
 * @param pTrMem_chs allocated TrMem for each channel
 * @return retVal
*/
int32_t tvm_tidl_configure_channel(uint8_t *dmaUtilsContext,
    int32_t ch, uint8_t *pTrMem_chs[],
    uint8_t *srcPtr, uint8_t *dstPtr, tvmtidlDmaUtilsAutoInc3d_SyncType syncType,
    uint16_t sicnt0, uint16_t sicnt1, uint16_t sicnt2, uint16_t sicnt3,
                      int32_t sdim1,   int32_t sdim2,   int32_t sdim3,
    uint16_t dicnt0, uint16_t dicnt1, uint16_t dicnt2, uint16_t dicnt3,
                      int32_t ddim1,   int32_t ddim2,   int32_t ddim3)
{
  int32_t retVal = DMAUTILS_SOK;
  DmaUtilsAutoInc3d_TrPrepareParam trPrepParams;
  DmaUtilsAutoInc3d_TransferProp   xferProp;

  // DMA src into dst
  // 1. setup Xfer Prop
  xferProp.transferDim.sicnt0 = sicnt0;
  xferProp.transferDim.sicnt1 = sicnt1;
  xferProp.transferDim.sicnt2 = sicnt2;
  xferProp.transferDim.sicnt3 = sicnt3;
  xferProp.transferDim.sdim1  = sdim1;
  xferProp.transferDim.sdim2  = sdim2;
  xferProp.transferDim.sdim3  = sdim3;

  xferProp.transferDim.dicnt0 = dicnt0;
  xferProp.transferDim.dicnt1 = dicnt1;
  xferProp.transferDim.dicnt2 = dicnt2;
  xferProp.transferDim.dicnt3 = dicnt3;
  xferProp.transferDim.ddim1  = ddim1;
  xferProp.transferDim.ddim2  = ddim2;
  xferProp.transferDim.ddim3  = ddim3;

  xferProp.circProp.circSize1 = 0;
  xferProp.circProp.circSize2 = 0;
  xferProp.circProp.addrModeIcnt0 = (uint8_t)DMAUTILSAUTOINC3D_ADDR_LINEAR;
  xferProp.circProp.addrModeIcnt1 = (uint8_t)DMAUTILSAUTOINC3D_ADDR_LINEAR;
  xferProp.circProp.addrModeIcnt2 = (uint8_t)DMAUTILSAUTOINC3D_ADDR_LINEAR;
  xferProp.circProp.addrModeIcnt3 = (uint8_t)DMAUTILSAUTOINC3D_ADDR_LINEAR;
  xferProp.circProp.circDir       = (uint8_t)DMAUTILSAUTOINC3D_CIRCDIR_DST;

  xferProp.syncType               = (int32_t)syncType;
  xferProp.dmaDfmt                = (uint32_t)DMAUTILSAUTOINC3D_DFMT_NONE;

  // pointers could also be set in pTrMem_chs[0] before configure()
  xferProp.ioPointers.srcPtr = (uint8_t *) srcPtr;
  xferProp.ioPointers.dstPtr = (uint8_t *) dstPtr;

  // 2. prepareTr
  trPrepParams.numTRs    = 1;
  trPrepParams.trMemSize = DmaUtilsAutoInc3d_getTrMemReq(trPrepParams.numTRs);
  trPrepParams.channelId = ch;
  trPrepParams.trMem     = pTrMem_chs[ch];

  DEBUG_PRINT("calling DmaUtilsAutoInc3d_prepareTr...\n");
  retVal = DmaUtilsAutoInc3d_prepareTr(&trPrepParams, &xferProp);
  DEBUG_PRINT("DmaUtilsAutoInc3d_prepareTr, retVal=%d\n", retVal);

  // 3. configure
  // ((CSL_UdmapTR *) pTrMem_chs[ch])->addr  = srcPtr;
  // ((CSL_UdmapTR *) pTrMem_chs[ch])->daddr = dstPtr;
  uint32_t convertMask = DMAUTILSAUTOINC3D_ADDRCONVERTMASK_SRCADDR |
                            DMAUTILSAUTOINC3D_ADDRCONVERTMASK_DSTADDR;
  DmaUtilsAutoInc3d_convertTrVirtToPhyAddr(dmaUtilsContext, &trPrepParams,
                                           convertMask);
  DEBUG_PRINT("calling DmaUtilsAutoInc3d_configure...\n");
  retVal = DmaUtilsAutoInc3d_configure(dmaUtilsContext, ch, pTrMem_chs[ch], 1);
  DEBUG_PRINT("DmaUtilsAutoInc3d_configure IN, retVal=%d\n", retVal);

  return retVal;
}

/**
 * @brief De-init dma setup
 *
 * @param dmaUtilsContext
 * @param num_channels number of DMA channels
 * @param pTrMem_chs allocated TrMem for each channel
 * @return retVal
*/
int32_t tvm_tidl_dmautils_deinit(uint8_t *dmaUtilsContext,
    int32_t num_channels, uint8_t *pTrMem_chs[])
{
  int32_t retVal = DMAUTILS_SOK;

  for (int32_t ch = 0; ch < num_channels; ch++)
  {
    retVal = DmaUtilsAutoInc3d_deconfigure(dmaUtilsContext,
                                           ch, pTrMem_chs[ch], 1);
    DEBUG_PRINT("DmaUtilsAutoInc3d_deconfigure %d, retVal=%d\n", ch, retVal);
  }

  retVal = DmaUtilsAutoInc3d_deinit(dmaUtilsContext);
  DEBUG_PRINT("DmaUtilsAutoInc3d_deinit, retVal=%d\n", retVal);

  return retVal;
}

/**
 * @brief Trigger DMA transfer
 *
 * @param dmaUtilsContext
 * @param channelId DMA channel
 * @return retVal
*/
int32_t tvm_tidl_dmautils_trigger(uint8_t *dmaUtilsContext, int32_t channelId)
{
  return DmaUtilsAutoInc3d_trigger(dmaUtilsContext, channelId);
}

/**
 * @brief Wait for DMA transfer to finish
 *
 * @param dmaUtilsContext
 * @param channelId DMA channel
*/
void tvm_tidl_dmautils_wait(uint8_t *dmaUtilsContext, int32_t channelId)
{
  DmaUtilsAutoInc3d_wait(dmaUtilsContext, channelId);
}

