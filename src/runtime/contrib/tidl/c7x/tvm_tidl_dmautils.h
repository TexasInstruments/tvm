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

//#include <ti/csl/csl_dma.h>
#include <ti/drv/udma/dmautils/dmautils.h>
#include <ti/drv/udma/udma.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
  TVMTIDL_DMAUTILS_CHANNEL_0 = 0,
  TVMTIDL_DMAUTILS_CHANNEL_1,
  TVMTIDL_DMAUTILS_CHANNEL_MAX
} tvmtidlDmaUtilsChannel;

extern void* getUDMADrvObjPtr();
extern void tvm_tidl_l2_scratch_reset();
extern uint8_t* tvm_tidl_l2_scratch_alloc(int32_t size);
extern int32_t  tvm_tidl_l2_scratch_avail_size();

extern uint8_t* tvm_tidl_dmautils_init(int32_t num_channels,
                                       uint8_t *pTrMem_chs[]);

extern int32_t tvm_tidl_configure_channel(uint8_t *dmaUtilsContext,
    int32_t ch, uint8_t *pTrMem_chs[],
    uint8_t *srcPtr, uint8_t *dstPtr, DmaUtilsAutoInc3d_SyncType syncType,
    uint16_t sicnt0, uint16_t sicnt1, uint16_t sicnt2, uint16_t sicnt3,
                      int32_t sdim1,   int32_t sdim2,   int32_t sdim3,
    uint16_t dicnt0, uint16_t dicnt1, uint16_t dicnt2, uint16_t dicnt3,
                      int32_t ddim1,   int32_t ddim2,   int32_t ddim3);

extern int32_t tvm_tidl_dmautils_deinit(uint8_t *dmaUtilsContext,
    int32_t num_channels, uint8_t *pTrMem_chs[]);

#ifdef __cplusplus
}
#endif
