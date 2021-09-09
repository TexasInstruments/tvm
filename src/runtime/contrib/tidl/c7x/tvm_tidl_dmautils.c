/******************************************************************************
 * Copyright (c) 2021, Texas Instruments Incorporated - http://www.ti.com/
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions are met:
 *       * Redistributions of source code must retain the above copyright
 *         notice, this list of conditions and the following disclaimer.
 *       * Redistributions in binary form must reproduce the above copyright
 *         notice, this list of conditions and the following disclaimer in the
 *         documentation and/or other materials provided with the distribution.
 *       * Neither the name of Texas Instruments Incorporated nor the
 *         names of its contributors may be used to endorse or promote products
 *         derived from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 *   THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

/* This file provides simplified API for using DmaUtils
   in TVM+TIDL generated code */

#include <stdio.h>
#include "tvm_tidl_dmautils.h"

//#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT(...)

#define L2_ALIGN_SIZE (128U)
#define L2_ALIGN_CEIL(VAL, ALIGN) ((((VAL)+(ALIGN)-1)/(ALIGN)) * (ALIGN))

static uint8_t *p_l2_scratch;
static int32_t l2_scratch_avail_size;

void tvm_tidl_l2_scratch_reset()
{
  #if 1
  extern void* g_l2_mem_addr;
  p_l2_scratch = (uint8_t *) g_l2_mem_addr;
  #else
  p_l2_scratch = (uint8_t *) 0x64800000;
  #endif
  l2_scratch_avail_size = 448*1024;  // 448 KB
}

uint8_t *tvm_tidl_l2_scratch_alloc(int32_t size)
{
  if (size <= 0 || l2_scratch_avail_size < size)  return NULL;

  uint8_t *alloc_ptr = p_l2_scratch;
  int32_t aligned_alloc_size = L2_ALIGN_CEIL(size, L2_ALIGN_SIZE);
  p_l2_scratch          += aligned_alloc_size;
  l2_scratch_avail_size -= aligned_alloc_size;
  return alloc_ptr;
}

int32_t tvm_tidl_l2_scratch_avail_size()
{
  return l2_scratch_avail_size;
}

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
  int32_t retVal = UDMA_SOK;
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
  initParams.udmaDrvHandle   = (Udma_DrvHandle) getUDMADrvObjPtr();
  initParams.DmaUtilsVprintf = vprintf;

  for (int32_t ch = 0; ch < num_channels; ch++)
  {
    chInitParams[ch].dmaQueNo = 0;
    chInitParams[ch].druOwner = DMAUTILSAUTOINC3D_DRUOWNER_DIRECT_TR;
  }

  retVal = DmaUtilsAutoInc3d_init(dmaUtilsContext, &initParams, chInitParams);
  if (retVal != UDMA_SOK)
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
    uint8_t *srcPtr, uint8_t *dstPtr, DmaUtilsAutoInc3d_SyncType syncType,
    uint16_t sicnt0, uint16_t sicnt1, uint16_t sicnt2, uint16_t sicnt3,
                      int32_t sdim1,   int32_t sdim2,   int32_t sdim3,
    uint16_t dicnt0, uint16_t dicnt1, uint16_t dicnt2, uint16_t dicnt3,
                      int32_t ddim1,   int32_t ddim2,   int32_t ddim3)
{
  int32_t retVal = UDMA_SOK;
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
  int32_t retVal = UDMA_SOK;

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
