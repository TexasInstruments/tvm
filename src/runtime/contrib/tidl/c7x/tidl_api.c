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

/*------------------------------------------------------------------------------*/
// TIDL_API.C
//   Implement the TIDL API interface
/*------------------------------------------------------------------------------*/
#include "tidl_api.h"
#include "tidl_api_mem.h"
#include "itidl_ti.h"
#include "itidl_rt.h"
#include "ivision.h"

#include <stddef.h>        // itidl_ti.h uses NULL w/o this #include
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if (HOST_EMULATION)
   #define restrict __restrict__
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

// Copied from tidl_config.h
#define TIDL_FLOW_CTRL_DEFAULT  (0x00000000)
#define TIDL_FLOW_CTRL_REF_ONLY (0x00000001)
#define TIDL_FLOW_CTRL_REF_STAT (0x00000002)
#define TIDL_FLOW_CTRL_MMA_NATC (0x00000004)
#define TIDL_FLOW_CTRL_DSP_NATC (0x00000008)
#define TIDL_FLOW_CTRL_REF_COMP (0x00000010)

// When network statically linked into firmware, for some reason, TIDL
// updates the network at TIDL_free() call.  Make a copy of the network.
// Should not have this problem when migrating to dynamically loading
// TIDL networks.
// #define TIDL_COPY_NETWORK_BUF 1

// Represents a TIDL instance, including all its state
typedef struct
{
  IVISION_Handle     handle;      // IALG handle
  sTIDL_IOBufDesc_t* IOParams;    // Descriptors for in/out buffers
  sTIDL_Network_t*   network;     // The TIDL network
  uint32_t           network_size;// The TIDL network size in bytes
  int                numMemRec;   // Number of allocation pools
  IALG_MemRec*       memRec;      // IALG memory request/allocation pools
  IVISION_InBufs*    inBufs;      // Input tensor buffers
  IVISION_OutBufs*   outBufs;     // Output tensor buffers
  TIDL_InArgs*       inArgs;      // Input arguments
  TIDL_outArgs*      outArgs;     // Ouput arguments
  int                is_nchw;     // From relay.nfo. 1 => no layout conversion
                                  // required before TIDL invocation
  TIDL_CreateParams* createParams;// TIDL createParams, need to keep outside of TIDL
} TIDL_subgraph_instance;


// Extern functions in TIDL RT
EXTERN_C void TIDLRT_PrintMetaData(TIDL_outArgs *outArgs);

// Helper functions
static int32_t init_inbufs(TIDL_subgraph_instance *instance);
static void    free_inbufs(TIDL_subgraph_instance *instance);
static int32_t init_outbufs(TIDL_subgraph_instance *instance);
static void    free_outbufs(TIDL_subgraph_instance *instance);
static int32_t tidl_element_size(int32_t elementType);

static int32_t connect_input_output_tensors(TIDL_subgraph_instance *instance,
                                            DLTensor *in_tensors[],
                                            DLTensor *out_tensors[]);
static int32_t disconnect_input_output_tensors(TIDL_subgraph_instance *instance);

// Memory management
static void init_mem_regions();
static int32_t alloc_mem_records(IALG_MemRec * memRec,int32_t numMemRec);
static int32_t free_mem_records(IALG_MemRec * memRec,int32_t numMemRec);

static int32_t printTIDLLog(const char * format, va_list va_args_ptr)
{
    static char buf[1024];

    vsnprintf(buf, 1024, format, va_args_ptr);

    printf(buf);

    return 0;
}

//-------------------------------------------------------------------------
// Create an instance of TIDL
EXTERN_C void* init_tidl_subgraph(void *network,
                                  uint32_t network_size,
                                  void *IOParams,
                                  void* udmaDrvObjPtr,
                                  int   is_nchw,
                                  void* in_rt_info)
{
  int32_t status = IALG_EOK;

  // Setup L1/L2/L3/L4 system memory regions, for servicing memory requests.
  init_mem_regions();

  TIDL_subgraph_instance* instance =
      (TIDL_subgraph_instance *)tidl_malloc(sizeof(TIDL_subgraph_instance));
  if (instance == NULL)  return NULL;

#ifdef TIDL_COPY_NETWORK_BUF
  instance->network = (sTIDL_Network_t*) tidl_malloc(network_size);
  memcpy(instance->network, network, network_size);
  instance->network_size = network_size;
#else
  instance->network = (sTIDL_Network_t*)network;
  instance->network_size = 0;
#endif
  instance->IOParams = (sTIDL_IOBufDesc_t*)IOParams;

  // Setup TIDL construction parameters.
  // Mostly defaults from setDefaultParams()
  TIDL_CreateParams  *createParams = tidl_malloc(sizeof(TIDL_CreateParams));
  instance->createParams = createParams;
  TIDL_createParamsInit(createParams);

  tvm_tidl_rt_info *rt_info = (tvm_tidl_rt_info *) in_rt_info;
  extern int32_t TVM_lockInterrupts();
  extern void    TVM_unlockInterrupts(int32_t);

  createParams->visionParams.algParams.size   = sizeof(TIDL_CreateParams);
  createParams->visionParams.cacheWriteBack   = NULL;
  createParams->currLayersGroupId             = 1;
  createParams->isInbufsPaded                 = 1;
  createParams->optimiseExtMem                = TIDL_OptimiseExtMemL1;
  createParams->quantRangeExpansionFactor     = 1.0;
  createParams->quantRangeUpdateFactor        = 0.0;
  if (rt_info != NULL)
  {
    createParams->traceLogLevel                 = rt_info->tidl_trace_log_level;
    createParams->traceWriteLevel               = rt_info->tidl_trace_write_level;
    createParams->maxPreEmptDelay               = rt_info->max_preempt_delay;
    createParams->targetPriority                = rt_info->tvm_rt_target_priority;
    createParams->coreId                        = rt_info->tvm_rt_core_num - 1;
  }
  createParams->reservedCtrl                  = 0;
#if (HOST_EMULATION)
  createParams->flowCtrl                      = TIDL_FLOW_CTRL_REF_ONLY;
#else
  createParams->flowCtrl                      = TIDL_FLOW_CTRL_DEFAULT ;
#endif
  createParams->traceBaseName                 = NULL;
  createParams->udmaDrvObj                    = udmaDrvObjPtr;

  createParams->net                           = instance->network;

  createParams->pFxnLock                      = TVM_lockInterrupts;
  createParams->pFxnUnLock                    = TVM_unlockInterrupts;
  createParams->TIDLVprintf                   = printTIDLLog;
  //createParams.TIDLVprintf                   = vprintf;
  createParams->tracePtr                      = NULL;
  createParams->TIDLWriteBinToFile            = NULL;
  createParams->TIDLReadBinFromFile           = NULL;
  createParams->TIDL_CustomLayerProcess       = NULL;

  // Setup memRecs and solicit memory requests.
  // Each memRec is a pool of memory requested by TIDL.

  // First, find out how many memRecs we need.
  //   TIDL_numalloc
  int32_t numMemRec = TIDL_VISION_FXNS.ialg.algNumAlloc();
  IALG_MemRec *memRec =
      (IALG_MemRec *)tidl_malloc(numMemRec*sizeof(IALG_MemRec));
  instance->numMemRec = numMemRec;
  instance->memRec = memRec;
  if (memRec == NULL)  status = IALG_EFAIL;

  // Let TIDL fill in the requests.
  //   TIDL_alloc
  if (status == IALG_EOK)
  {
    status = TIDL_VISION_FXNS.ialg.algAlloc((IALG_Params *)(createParams),
                                             NULL, memRec);
    if (status != IALG_EOK)  printf("init_tidl_subgraph: algAlloc failed\n");
  }

  // Allocate the memory pools as requested.
  if (status == IALG_EOK)
  {
    status = alloc_mem_records(memRec, numMemRec);
    if (status != IALG_EOK)
      printf("init_tidl_subgraph: alloc_mem_records failed\n");
  }

  // Call IALG algInit API to instantiate TIDL and setup all its internal
  // data structures.
  //   TIDL_init
  if (status == IALG_EOK)
  {
    IALG_Handle handle = (IALG_Handle) memRec[0].base;
    status = TIDL_VISION_FXNS.ialg.algInit(handle, memRec, NULL,
				(IALG_Params *)(createParams));
    if (status != IALG_EOK)  printf("init_tidl_subgraph: algInit failed\n");
  }

  // Set the algorithm handle to the newly created TIDL instance.
  if (status == IALG_EOK)
  {
    instance->handle = (IVISION_Handle) memRec[0].base;
  }

  // Allocate IVISION_InBufs/OutBufs for input and output tensors.
  if (status == IALG_EOK)
  {
    status  = init_inbufs(instance);
    status |= init_outbufs(instance);
  }

  // Allocate TIDL_InArgs structure for passing arguments to TIDL_process.
  TIDL_InArgs *inArgs = NULL;
  if (status == IALG_EOK)
  {
    inArgs = (TIDL_InArgs *)tidl_malloc(sizeof(TIDL_InArgs));
    if (inArgs != NULL)
    {
      inArgs->iVisionInArgs.size = sizeof(TIDL_InArgs);
      inArgs->iVisionInArgs.subFrameInfo = 0;
      inArgs->enableLayerPerfTraces = (createParams->traceLogLevel > 0) ? 1 : 0;
    }
    else
    {
      printf("init_tidl_subgraph, InArgs alloc failed\n");
      status = IALG_EFAIL;
    }
  }
  instance->inArgs = inArgs;

  // Allocate TIDL_outArgs structure for returning values from TIDL_process.
  TIDL_outArgs *outArgs = NULL;
  if (status == IALG_EOK)
  {
    outArgs = (TIDL_outArgs *)tidl_malloc(sizeof(TIDL_outArgs));
    if (outArgs != NULL)
    {
      outArgs->iVisionOutArgs.size = sizeof(TIDL_outArgs);
    }
    else
    {
      printf("init_tidl_subgraph, OutArgs alloc failed\n");
      status = IALG_EFAIL;
    }
  }
  instance->outArgs = outArgs;

  instance->is_nchw = is_nchw;

  if (status != IALG_EOK)
  {
    // Cleanup Sequence
    free_tidl_subgraph(instance);
    return NULL;
  }

  return instance;
}

//-------------------------------------------------------------------------
// Invoke TIDL to perform its computation
EXTERN_C int32_t process_tidl_subgraph(void *instance_,
                           DLTensor *in_tensors[],
			   DLTensor *out_tensors[])
{
  TIDL_subgraph_instance* instance = (TIDL_subgraph_instance*)instance_;
  IVISION_Handle handle = instance->handle;
  int32_t status = IALG_EOK;

  // Call IALG activate API to give TIDL ownership of its memory.
  //   TIDL_activate
  handle->fxns->ialg.algActivate((IALG_Handle)(instance->handle));

  // With TIDL DataConv layers, subgraph directly use TVM tensors' data buffers
  connect_input_output_tensors(instance, in_tensors, out_tensors);

  // Call IALG process API to run the network. This is the TIDL interpreter.
  //   TIDL_process
  handle->fxns->algProcess(instance->handle,
                           instance->inBufs, instance->outBufs,
		           (IVISION_InArgs *)instance->inArgs,
	                   (IVISION_OutArgs *)instance->outArgs);
  if (status != IALG_EOK)
  {
    printf("process_tidl_subgraph: algProcess failed\n");
  }

  // With TIDL DataConv layers, subgraph directly use TVM tensors' data buffers
  disconnect_input_output_tensors(instance);

  // Call IALG deactivate API to release TIDL's ownership.
  //   TIDL_deactivate
  handle->fxns->ialg.algDeactivate((IALG_Handle)(instance->handle));

  // Dump layer perf info
  if (instance->inArgs->enableLayerPerfTraces > 0)
  {
    TIDLRT_PrintMetaData(instance->outArgs);
  }

  return status;
}

//-------------------------------------------------------------------------
// Free resources used by TIDL instance
EXTERN_C int32_t free_tidl_subgraph(void *instance_)
{
  TIDL_subgraph_instance *instance = (TIDL_subgraph_instance*)instance_;
  IVISION_Handle handle = instance->handle;
  int32_t status = IALG_EOK;

  tidl_free(instance->outArgs, sizeof(TIDL_outArgs));
  tidl_free(instance->inArgs, sizeof(TIDL_InArgs));
  free_outbufs(instance);
  free_inbufs(instance);

  // tidl_tb_algFree()
  if (handle != NULL)
    status = handle->fxns->ialg.algFree((IALG_Handle)(handle),instance->memRec);
  if (status != IALG_EOK)
  {
    printf("free_tidl_subgraph: algFree failed\n");
  }

  free_mem_records(instance->memRec, instance->numMemRec);
  tidl_free(instance->memRec, instance->numMemRec * sizeof(IALG_MemRec));
  tidl_free(instance->createParams, sizeof(TIDL_CreateParams));

  if (instance->network_size > 0)
    tidl_free(instance->network, instance->network_size);

  tidl_free(instance_, sizeof(TIDL_subgraph_instance));

  return status;
}

// Allocate and initialize IVISION buffers used for TIDL input tensors.
static int32_t init_inbufs(TIDL_subgraph_instance *instance)
{
  sTIDL_IOBufDesc_t *IOParams = instance->IOParams;
  int nBufs = IOParams->numInputBuf;
  IVISION_InBufs *Bufs =
     (IVISION_InBufs *)tidl_malloc(sizeof(IVISION_InBufs));
  IVISION_BufDesc **BufDescList =
     (IVISION_BufDesc **)tidl_malloc(sizeof(IVISION_BufDesc*) * nBufs);
  IVISION_BufDesc *BufDescs =
     (IVISION_BufDesc *)tidl_malloc(sizeof(IVISION_BufDesc) * nBufs);

  if (Bufs == NULL || BufDescList == NULL || BufDescs == NULL)
  {
    tidl_free(BufDescs, sizeof(IVISION_BufDesc) * nBufs);
    tidl_free(BufDescList, sizeof(IVISION_BufDesc*) * nBufs);
    tidl_free(Bufs, sizeof(IVISION_InBufs));
    instance->inBufs = NULL;
    printf("init_inbufs: memory alloc failed\n");
    return IALG_EFAIL;
  }

  Bufs->numBufs = nBufs;
  Bufs->bufDesc = BufDescList;
  instance->inBufs = Bufs;

  for(int i = 0; i < nBufs; ++i)
  {
    int32_t bufWidth;
    int32_t bufHeight;
    int32_t elementSizeBytes =
       tidl_element_size(IOParams->inElementType[i]);
    IVISION_BufDesc *BufDesc = &BufDescs[i];
    BufDescList[i] = BufDesc;
    BufDesc->bufferId = IOParams->inDataId[i];
    BufDesc->numPlanes = 1;
    BufDesc->reserved[0] = IOParams->inDataId[i];
    BufDesc->bufPlanes[0].frameROI.topLeft.x    = 0;
    BufDesc->bufPlanes[0].frameROI.topLeft.y    = 0;

    bufWidth  = IOParams->inWidth[i] + IOParams->inPadL[i]
                                     + IOParams->inPadR[i];
    bufHeight = IOParams->inNumChannels[i] *
        (IOParams->inHeight[i] + IOParams->inPadT[i] + IOParams->inPadB[i]);

    BufDesc->bufPlanes[0].width  = bufWidth;
    BufDesc->bufPlanes[0].height = bufHeight;
    BufDesc->bufPlanes[0].frameROI.width = IOParams->inWidth[i];
    BufDesc->bufPlanes[0].frameROI.height = IOParams->inHeight[i];

    // TIDL DataConv: subgraph directly use TVM tensors' data buffers later
    BufDesc->bufPlanes[0].buf = NULL;
  }
  return IALG_EOK;
}

static void free_inbufs(TIDL_subgraph_instance *instance)
{
  if (instance->inBufs == NULL)  return;

  sTIDL_IOBufDesc_t *IOParams = instance->IOParams;
  int nBufs = IOParams->numInputBuf;
  IVISION_InBufs *Bufs = instance->inBufs;
  IVISION_BufDesc **BufDescList = Bufs->bufDesc;
  IVISION_BufDesc *BufDescs = BufDescList[0];

  Bufs->numBufs = nBufs;
  Bufs->bufDesc = BufDescList;
  instance->inBufs = Bufs;

  for(int i = 0; i < nBufs; ++i)
  {
    IVISION_BufDesc *BufDesc = &BufDescs[i];
    if (BufDesc->bufPlanes[0].buf != NULL)
      break;
    tidl_free(BufDesc->bufPlanes[0].buf, BufDesc->reserved[1]);
  }
  tidl_free(BufDescs, sizeof(IVISION_BufDesc) * nBufs);
  tidl_free(BufDescList, sizeof(IVISION_BufDesc*) * nBufs);
  tidl_free(Bufs, sizeof(IVISION_InBufs));
  instance->inBufs = NULL;
}

//-------------------------------------------------------------------------
// Allocate and initialize IVISION buffers used for TIDL output tensors.
static int32_t init_outbufs(TIDL_subgraph_instance *instance)
{
  sTIDL_IOBufDesc_t *IOParams = instance->IOParams;
  int nBufs = IOParams->numOutputBuf;
  IVISION_OutBufs *Bufs =
     (IVISION_OutBufs *)tidl_malloc(sizeof(IVISION_OutBufs));
  IVISION_BufDesc **BufDescList =
     (IVISION_BufDesc **)tidl_malloc(sizeof(IVISION_BufDesc*) * nBufs);
  IVISION_BufDesc *BufDescs =
     (IVISION_BufDesc *)tidl_malloc(sizeof(IVISION_BufDesc) * nBufs);

  if (Bufs == NULL || BufDescList == NULL || BufDescs == NULL)
  {
    tidl_free(BufDescs, sizeof(IVISION_BufDesc) * nBufs);
    tidl_free(BufDescList, sizeof(IVISION_BufDesc*) * nBufs);
    tidl_free(Bufs, sizeof(IVISION_InBufs));
    instance->outBufs = NULL;
    printf("init_outbufs: memory alloc failed\n");
    return IALG_EFAIL;
  }

  Bufs->numBufs  = nBufs;
  Bufs->bufDesc  = BufDescList;
  instance->outBufs = Bufs;

  for(int i = 0; i < nBufs; ++i)
  {
    int32_t elementSizeBytes  =
       tidl_element_size(IOParams->outElementType[i]);

    IVISION_BufDesc *BufDesc = &BufDescs[i];
    BufDescList[i] = BufDesc;
    BufDesc->bufferId = IOParams->outDataId[i];
    BufDesc->reserved[0] = IOParams->outDataId[i];
    BufDesc->numPlanes = 1;
    BufDesc->bufPlanes[0].frameROI.topLeft.x    = 0;
    BufDesc->bufPlanes[0].frameROI.topLeft.y    = 0;

    int32_t imHeight      = IOParams->outHeight[i];
    int32_t imWidth       = IOParams->outWidth[i];

    BufDesc->bufPlanes[0].width = imWidth + IOParams->outPadL[i]
                                          + IOParams->outPadR[i];
    BufDesc->bufPlanes[0].height = IOParams->outNumChannels[i]*
              (imHeight + IOParams->outPadT[i] + IOParams->outPadB[i]);
    BufDesc->bufPlanes[0].frameROI.width = imWidth;
    BufDesc->bufPlanes[0].frameROI.height = imHeight;

    // TIDL DataConv: subgraph directly use TVM tensors' data buffers later
    BufDesc->bufPlanes[0].buf = NULL;
  }
  return IALG_EOK;
}

static void free_outbufs(TIDL_subgraph_instance *instance)
{
  if (instance->outBufs == NULL)  return;

  sTIDL_IOBufDesc_t *IOParams = instance->IOParams;
  int nBufs = IOParams->numOutputBuf;
  IVISION_OutBufs *Bufs = instance->outBufs;
  IVISION_BufDesc **BufDescList = Bufs->bufDesc;
  IVISION_BufDesc *BufDescs = BufDescList[0];

  for(int i = 0; i < nBufs; ++i)
  {
    IVISION_BufDesc *BufDesc = &BufDescs[i];
    if (BufDesc->bufPlanes[0].buf == NULL)
      break;
    tidl_free(BufDesc->bufPlanes[0].buf, BufDesc->reserved[1]);
  }
  tidl_free(BufDescs, sizeof(IVISION_BufDesc) * nBufs);
  tidl_free(BufDescList, sizeof(IVISION_BufDesc*) * nBufs);
  tidl_free(Bufs, sizeof(IVISION_OutBufs));
  instance->outBufs = NULL;
}

static int32_t connect_input_output_tensors(TIDL_subgraph_instance *instance,
                                            DLTensor *in_tensors[],
                                            DLTensor *out_tensors[])
{
  IVISION_BufDesc**  inBufDescList = instance->inBufs->bufDesc;
  IVISION_BufDesc**  outBufDescList = instance->outBufs->bufDesc;
  TIDL_InArgs*       inArgs = instance->inArgs;
  sTIDL_IOBufDesc_t* IOParams = instance->IOParams;

  for(int i = 0; i < IOParams->numInputBuf; i++)
  {
    IVISION_BufDesc* inBufDesc = inBufDescList[i];
    inArgs->scale[i] = 1.0;
    inBufDesc->bufPlanes[0].buf = in_tensors[i]->data;
  }

  for(int i = 0; i < IOParams->numOutputBuf; i++)
  {
    IVISION_BufDesc* outBufDesc = outBufDescList[i];

    /* Find tvm/tidlrt tensor that corresponds to this tidl output tensor */
    /* After imported into TIDL layers, TIDL performs transormations and
         topologically sorts all layers, always iterating from layer 0.
         While tvm/tidl_rt input tensor <i> is still TIDL input tensor <i>,
         tvm/tidl_rt output tensor <i> could become TIDL output tensor <j>.
       TIDL outDataName is encoded as tidl_<subgraph_id>_o<tidlrt_id>,
         from which we can extract the corresponding tvm output tensor index.
    */
    const char *out_name = (const char *) IOParams->outDataName[i];
    int32_t pos = strlen(out_name) - 1;
    while(out_name[pos] != 'o')  pos--;
    int32_t j = atoi(&out_name[pos+1]);

    outBufDesc->bufPlanes[0].buf = out_tensors[j]->data;
  }

  return 0;
}

static int32_t disconnect_input_output_tensors(TIDL_subgraph_instance *instance)
{
  IVISION_BufDesc**  inBufDescList = instance->inBufs->bufDesc;
  IVISION_BufDesc**  outBufDescList = instance->outBufs->bufDesc;
  sTIDL_IOBufDesc_t* IOParams = instance->IOParams;

  for(int i = 0; i < IOParams->numInputBuf; i++)
  {
    IVISION_BufDesc* inBufDesc = inBufDescList[i];
    inBufDesc->bufPlanes[0].buf = NULL;
  }

  for(int i = 0; i < IOParams->numOutputBuf; i++)
  {
    IVISION_BufDesc* outBufDesc = outBufDescList[i];
    outBufDesc->bufPlanes[0].buf = NULL;
  }

  return 0;
}

//-------------------------------------------------------------------------
// Helper function to get size of TIDL element types.
static int32_t tidl_element_size(int32_t elementType)
{
  switch (elementType)
  {
    case TIDL_SignedChar:
    case TIDL_UnsignedChar:  return 1;
    case TIDL_SignedShort:
    case TIDL_UnsignedShort: return 2;
    case TIDL_SinglePrecFloat:
    case TIDL_UnsignedWord:
    case TIDL_SignedWord: return 4;
    default : return 1;
  }
}

//-------------------------------------------------------------------------
// Manage memory used by TIDL
#include "ti_mem_manager.h"

#include "tidl_api_mem.h"

TIMemObject memObj_DMEM0;
TIMemObject memObj_DMEM1;
TIMemObject memObj_SARAM0;
TIMemObject memObj_EXTMEM;

// Iniitalize pools of memory for use by TIDL
static void init_mem_regions()
{
  uint8_t * L1Scratch = NULL;
  uint8_t * L2Scratch = NULL;
  uint8_t * L3Scratch = NULL;
  uint32_t  L1Size;
  uint32_t  L2Size;
  uint32_t  L3Size;
#if HOST_EMULATION
  // Copied from test app, but seems suspicious. Align to total size
  // of internal memory block?
  L1Scratch = (uint8_t*)tidl_memalign(L1_TOTAL_MEMORY_SIZE, L1_MEM_SIZE);
  L2Scratch = (uint8_t*)tidl_memalign(L2_TOTAL_MEMORY_SIZE, L2_MEM_SIZE);
  L3Scratch = (uint8_t*)tidl_memalign(L3_TOTAL_MEMORY_SIZE, L3_MEM_SIZE);
  L1Size = L1_MEM_SIZE;
  L2Size = L2_MEM_SIZE;
  L3Size = L3_MEM_SIZE;
#else
  L1Scratch = (uint8_t *) g_l1_mem_addr;
  L2Scratch = (uint8_t *) g_l2_mem_addr;
  L3Scratch = (uint8_t *) g_l3_mem_addr;
  L1Size = g_l1_mem_size;
  L2Size = g_l2_mem_size;
  L3Size = g_l3_mem_size;
#endif

  // We malloc L4 requests directly, to keep track of usage
  //uint8_t * L4Scratch = NULL;
  //L4Scratch = (uint8_t*)tidl_malloc(L4_MEM_SIZE);

  //TIDLTB_ASSERT_EXIT(((L1Scratch != NULL) && (L2Scratch != NULL) && (L3Scratch != NULL) && (L4Scratch != NULL)));

  TI_CreateMemoryHandle(&memObj_DMEM0,  L1Scratch, L1Size);
  TI_CreateMemoryHandle(&memObj_DMEM1,  L2Scratch, L2Size);
  TI_CreateMemoryHandle(&memObj_SARAM0, L3Scratch, L3Size);
  //TI_CreateMemoryHandle(&memObj_EXTMEM, L4Scratch, L4_MEM_SIZE);
}

// Fulfill TIDL memory requests by allocating from pre-allocated pools
static int32_t alloc_mem_records(IALG_MemRec * memRec,int32_t numMemRec)
{
  int32_t i;
  int32_t totalDdrSize = 0;
  int32_t totalHeapSize = 0;

  TIMemHandle memHdl_DMEM0 = &memObj_DMEM0;
  TIMemHandle memHdl_DMEM1 = &memObj_DMEM1;
  TIMemHandle memHdl_SARAM0 = &memObj_SARAM0;
  //TIMemHandle memHdl_EXTMEM = &memObj_EXTMEM;

  for (i = 0; i < numMemRec; i++)
  {
    if(memRec[i].space == IALG_DARAM0) /* L1 D*/
    {
      memRec[i].base = TI_GetMemoryChunk(memHdl_DMEM0, memRec[i].size,
        memRec[i].alignment);
    }
    else if(memRec[i].space == IALG_DARAM1) /* L2 SRAM*/
    {
      memRec[i].base = TI_GetMemoryChunk(memHdl_DMEM1, memRec[i].size,
        memRec[i].alignment);
    }
    else if(memRec[i].space == IALG_SARAM0) /* L3 MSMC SRAM*/
    {
      memRec[i].base = TI_GetMemoryChunk(memHdl_SARAM0, memRec[i].size,
        memRec[i].alignment);
    }
    #if 0
    else if((memRec[i].space == IALG_EXTERNAL) && (memRec[i].attrs == IALG_SCRATCH))

    {
      memRec[i].base = TI_GetMemoryChunk(memHdl_EXTMEM, memRec[i].size,
        memRec[i].alignment);
    }
    #endif
    #ifndef HOST_EMULATION
    else if((memRec[i].space == IALG_EXTERNAL) && (memRec[i].attrs == IALG_SCRATCH))
    {
      memRec[i].base = (uint8_t *) appMemAlloc(APP_MEM_HEAP_DDR_SCRATCH,
                                          memRec[i].size, memRec[i].alignment);
    }
    else if(memRec[i].space == IALG_EXTERNAL)
    {
      memRec[i].base = (uint8_t *) appMemAlloc(APP_MEM_HEAP_DDR,
                                          memRec[i].size, memRec[i].alignment);
    }
    #endif
    else
    {
      memRec[i].base = (void *) tidl_memalign(memRec[i].alignment, memRec[i].size);
      totalHeapSize += memRec[i].size;
    }
    if(memRec[i].base == NULL)
    {
     printf("Could not Allocate memory for memtab %d of size %d in %d\n", i, memRec[i].size,memRec[i].space);
     return IALG_EFAIL;
    }
    else
    {
      memset(memRec[i].base, 0, memRec[i].size);
    }
  }

  //printf("Num,    Space,     SizeinBytes,   SizeInMB\n");
  for (i = 0; i < numMemRec; i++)
  {
    //printf(" %3d, %5d, %12d,    %7.3f %p\n", i, memRec[i].space, memRec[i].size, memRec[i].size / (1024.0 * 1024), memRec[i].base);
    if(memRec[i].space == IALG_EXTERNAL)
      totalDdrSize += memRec[i].size;
  }
  //printf("Total External Memory (DDR) Size = %12d,    %7.3f \n", totalDdrSize, totalDdrSize / (1024.0 * 1024));
 // printf("Total Heap Size = %12d,    %7.3f \n", totalHeapSize, totalHeapSize / (1024.0 * 1024));

  return IALG_EOK;
}

// Free TIDL memory records
static int32_t free_mem_records(IALG_MemRec * memRec,int32_t numMemRec)
{
  if (memRec == NULL)  return IALG_EFAIL;

  int32_t i;
  TIMemHandle memHdl_DMEM0 = &memObj_DMEM0;
  TIMemHandle memHdl_DMEM1 = &memObj_DMEM1;
  TIMemHandle memHdl_SARAM0 = &memObj_SARAM0;
  //TIMemHandle memHdl_EXTMEM = &memObj_EXTMEM;

  for (i = 0; i < numMemRec; i++)
  {
    if(memRec[i].base == NULL)
    {
      return IALG_EFAIL;
    }
    if(memRec[i].space == IALG_DARAM0) {
      TI_ResetMemoryHandle(memHdl_DMEM0);
    }
    else if(memRec[i].space == IALG_DARAM1) {
      TI_ResetMemoryHandle(memHdl_DMEM1);
    }
    else if(memRec[i].space == IALG_SARAM0) {
      TI_ResetMemoryHandle(memHdl_SARAM0);
    }
    #ifndef HOST_EMULATION
    else if((memRec[i].space == IALG_EXTERNAL) && (memRec[i].attrs == IALG_SCRATCH))
    {
      appMemFree(APP_MEM_HEAP_DDR_SCRATCH, memRec[i].base, memRec[i].size);
    }
    else if(memRec[i].space == IALG_EXTERNAL)
    {
      appMemFree(APP_MEM_HEAP_DDR, memRec[i].base, memRec[i].size);
    }
    #endif
    #if 0
    else if((memRec[i].space == IALG_EXTERNAL) && (memRec[i].attrs == IALG_SCRATCH)) {
      TI_ResetMemoryHandle(memHdl_EXTMEM);
    }
    #endif
    else {
      tidl_free(memRec[i].base, memRec[i].size);
    }
  }
  return IALG_EOK;
}
