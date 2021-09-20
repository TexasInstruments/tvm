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

#ifndef _TVM_TIDL_DMAUTILS_H_
#define _TVM_TIDL_DMAUTILS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
  TVMTIDL_DMAUTILS_CHANNEL_0 = 0,
  TVMTIDL_DMAUTILS_CHANNEL_1,
  TVMTIDL_DMAUTILS_CHANNEL_MAX
} tvmtidlDmaUtilsChannel;

/* Same as DmaUtilsAutoInc3d_SyncType in ti/drv/udma/dmautils/dmautils.h */
typedef enum{
  TVMTIDL_DMAUTILSAUTOINC3D_SYNC_1D = 0,
  TVMTIDL_DMAUTILSAUTOINC3D_SYNC_2D = 1,
  TVMTIDL_DMAUTILSAUTOINC3D_SYNC_3D = 2,
  TVMTIDL_DMAUTILSAUTOINC3D_SYNC_4D = 3
}tvmtidlDmaUtilsAutoInc3d_SyncType;


extern void* getUDMADrvObjPtr();

extern uint8_t* tvm_tidl_dmautils_init(int32_t num_channels,
                                       uint8_t *pTrMem_chs[]);

extern int32_t tvm_tidl_configure_channel(uint8_t *dmaUtilsContext,
    int32_t ch, uint8_t *pTrMem_chs[],
    uint8_t *srcPtr, uint8_t *dstPtr, tvmtidlDmaUtilsAutoInc3d_SyncType syncType,
    uint16_t sicnt0, uint16_t sicnt1, uint16_t sicnt2, uint16_t sicnt3,
                      int32_t sdim1,   int32_t sdim2,   int32_t sdim3,
    uint16_t dicnt0, uint16_t dicnt1, uint16_t dicnt2, uint16_t dicnt3,
                      int32_t ddim1,   int32_t ddim2,   int32_t ddim3);

extern int32_t tvm_tidl_dmautils_deinit(uint8_t *dmaUtilsContext,
    int32_t num_channels, uint8_t *pTrMem_chs[]);

extern int32_t tvm_tidl_dmautils_trigger(uint8_t *dmaUtilsContext, int32_t channel);
extern void    tvm_tidl_dmautils_wait(uint8_t *dmaUtilsContext, int32_t channel);

#ifdef __cplusplus
}
#endif

#endif  // _TVM_TIDL_DMAUTILS_H_
