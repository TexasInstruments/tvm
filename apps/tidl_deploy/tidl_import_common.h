/*
 *
 * Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
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

#ifndef TIDL_IMPORT_COMMONH_
#define TIDL_IMPORT_COMMONH_ 1

#include "ti_dl.h"

//#define TIDL_IMPORT_ENABLE_DBG_PRINT

int32_t TIDL_QuantizeUnsignedMax(uint8_t * params, float * data, int32_t dataSize, float min, float max, int32_t numBits, int32_t weightsElementSizeInBits, int32_t * zeroWeightValue);

int32_t TIDL_normalize(float data, float min, float max);

bool TIDL_readProtoFromTextFile(const char* fileName, Message* proto);

bool TIDL_readProtoFromBinaryFile(const char* fileName, Message* proto);

int32_t TIDL_getDataID(sTIDL_DataParams_t *data, 
sTIDL_OrgNetwork_t * pOrgTIDLNetStructure,
int32_t            numLayer, 
int8_t             *bufName);

int32_t TIDL_isDataBufUsed(int32_t           dataId, 
sTIDL_Network_t   *pTIDLNetStructure,
int32_t           numLayer);
int32_t TIDL_isInputConv2D(sTIDL_OrgNetwork_t   *pOrgTIDLNetStruct,
int32_t              numLayer, 
const char           *bufName);
void TIDL_UpdateInDataBuff(sTIDL_OrgNetwork_t * pOrgTIDLNetStructure, 
uint32_t numLayers, sTIDL_DataParams_t dataBuf);

void TIDL_findRange(float * data, int32_t dataSize, float * minOut, float * maxOut, float scale);
int64_t TIDL_roundSat(int64_t val, uint8_t bits, int64_t min, int64_t max);
int32_t TIDL_findSymQ(float  min, float max);
void TIDL_setConv2dKernelType(sTIDL_Network_t   *pTIDLNetStructure, int32_t tiLayerIndex);

int32_t TIDL_alignParamsWrite(FILE *fp, sBuffer_t * buf, uint32_t *totalParamSize, uint32_t numBytes);
int32_t TIDL_writeModel(sTIDL_Network_t * tIDLNetStructure, sTIDL_OrgNetwork_t * orgTIDLNetStructure, const char * name, uint32_t numLayers);

void  TIDL_fillDataBufPadRequirements(sTIDL_Network_t * tIDLNetStructure);
void * my_malloc(int size);
void my_free(void *ptr);
int32_t TIDL_writeInfo(sTIDL_Network_t * tIDLNetStructure, const char * name, uint32_t numLayers, uint32_t currLayersGroupId);

#define TIDL_DATA_FORMAT_NHWC         0
#define TIDL_DATA_FORMAT_NCHW         1
#define TIDL_DATA_FORMAT_UNDEFINED  (-1)
#define TIDL_PADDING_TYPE_SAME         0
#define TIDL_PADDING_TYPE_VALID        1
#define TIDL_PADDING_TYPE_UNSUPPORTED  (-1)

#define NUM_WHGT_BITS   (gParams.numParamBits)
#define NUM_BIAS_BITS   (15U)
#define NUM_BIAS_BYTES   (((NUM_BIAS_BITS + 15U) >> 4) << 1)

#define SHORT_DATA_FOR_IP      (0)
#define PRINT_TENSOR_MINMAX (0)
#define LAYER_INFO_FILENAME "layer_info.txt"

#define TIDL_IMPORT_SUCCESS                                    1
#define TIDL_IMPORT_FAILURE                                    0

// TIDL import error code definitions
#define TIDL_IMPORT_NO_ERR                                     0
#define TIDL_IMPORT_ERR_NET_TOPOLOTY_NOT_SUPPORTED             1
#define TIDL_IMPORT_ERR_LAYER_PARAMS_NOT_FOUND                 2
#define TIDL_IMPORT_ERR_SIZE_NOT_MATCH                         3
#define TIDL_IMPORT_ERR_BIAS_SIZE_NOT_MATCH                    4
#define TIDL_IMPORT_ERR_INNER_PRODUCT_NO_BIAS                  5
#define TIDL_IMPORT_ERR_QUANT_STYLE_NOT_SUPPORTED              6
#define TIDL_IMPORT_ERR_INPUT_LAYER_NOT_FOUND                  7
#define TIDL_IMPORT_ERR_OUTPUT_LAYER_NOT_FOUND                 8
#define TIDL_IMPORT_ERR_PAD_NOT_MERGEABLE                      9
#define TIDL_IMPORT_ERR_IP_INPUT_NOT_FLATTENED                 10
#define TIDL_IMPORT_ERR_DWCONV_DEPTH_MULT_INVALID              11
#define TIDL_IMPORT_ERR_CONCAT_INVALID_DIM                     12
#define TIDL_IMPORT_ERR_TENSOR_TYPE_NOT_SUPPORTED              13
#define TIDL_IMPORT_ERR_WEIGHT_TENSOR_INIT_NOT_FOUND           14
#define TIDL_IMPORT_ERR_GEMM_PARAMS_INVALID                    15
#define TIDL_IMPORT_ERR_SOFTMAX_AXIS_INVALID                   16
#define TIDL_IMPORT_ERR_STRIDE_INVALID                         17
#define TIDL_IMPORT_ERR_ELTWISE_INPUT_SIZE_INVALID             18
#define TIDL_IMPORT_ERR_SCALE_BLOBS_SIZE_INVALID               19
#define TIDL_IMPORT_ERR_DETOUT_INPUT_SIZE_INVALID              20
#define TIDL_IMPORT_ERR_RAWDATA_TYPE_UNSUPPORTED               21
#define TIDL_IMPORT_ERR_BATCHNORM_DATA_SIZE_INVALID            22
#define TIDL_IMPORT_ERR_BATCHNORM_BLOBS_SIZE_INVALID           23
#define TIDL_IMPORT_ERR_SLICE_PARAMS_UNSUPPORTED               24
#define TIDL_IMPORT_ERR_SLICE_NUMBER_NOT_MATCH                 25
#define TIDL_IMPORT_ERR_CROP_PARAMS_UNSUPPORTED                26
#define TIDL_IMPORT_ERR_FLATTEN_PARAMS_UNSUPPORTED             27
#define TIDL_IMPORT_ERR_BIAS_NOT_MERGED                        28

template <class Tout>
float TIDL_QuantizeSignedMax(Tout * params, float * data, int32_t dataSize, float min, float max, int32_t weightsElementSizeInBits);
int32_t TIDL_getLayerIdx(sTIDL_OrgNetwork_t * pOrgTIDLNetStructure, int32_t numLayer, const char *bufName);
typedef struct {
  int32_t layerType;
  int32_t(*tidl_tfOutReshape)(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
    int32_t              layerIndex);
}sTIDL_tfOutRehapeMap_t;

typedef struct {
  char           *layerName;
  char           *layerOpsString;
  uint32_t       NumOps;
} TIDL_TFLayerMapping_t;

#ifdef PLATFORM_64BIT
#define MAX_NUM_PTRS_TO_STORE (TIDL_NUM_MAX_LAYERS*10)

#define STORE_PTR(dst, src) \
    sBufPtrInd++; \
    if(sBufPtrInd == (MAX_NUM_PTRS_TO_STORE)) { \
        printf("Out of memory for storing pointers! Increase MAX_NUM_PTRS_TO_STORE.\n"); \
            exit(-1); \
    } \
    sBuffPtrs[sBufPtrInd] = src; \
    dst = sBufPtrInd;  

#define LOAD_PTR(ptr) sBuffPtrs[ptr] 
#define RESET_PTR(ptr) ptr = 0
#else
#define STORE_PTR(dst, src)  dst = src
#define LOAD_PTR(ptr) ptr 
#define RESET_PTR(ptr) ptr = NULL
#endif

#ifdef TIDL_IMPORT_ENABLE_DBG_PRINT
#define TIDL_IMPORT_DBG_PRINT(dbg_msg) printf(dbg_msg)
#define TIDL_IMPORT_DBG_PRINT2(dbg_msg, var) printf(dbg_msg, var)
#else
#define TIDL_IMPORT_DBG_PRINT(dbg_msg)
#define TIDL_IMPORT_DBG_PRINT2(dbg_msg, var)
#endif

int32_t TIDL_tfOutReshapeDataLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeConvLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapePoolingLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeIdentity(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeBN(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeRelu(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeSoftmax(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeIPLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeDeConvLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeConcatLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeSliceLayre(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeCropLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeFlattenLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeArgmaxLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapePadLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_tfOutReshapeDetOutLayer(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_getTfOpParamMapId(const char  * name);
int32_t tidl_linkInputTensors(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_linkOutputTensors(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_isAllInsAvailable(sTIDL_LayerPC_t  *orgLayer, sTIDL_OrgNetwork_t  *ptempTIDLNetStructure, int32_t layerIndex);
int32_t tidl_sortLayersInProcOrder(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, sTIDL_OrgNetwork_t  *ptempTIDLNetStructure, int32_t layerIndex);
int32_t tidl_removeMergedLayersFromNet(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, sTIDL_OrgNetwork_t  *ptempTIDLNetStructure, int32_t layerIndex);
int32_t tidl_upateAInDataId(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex, int32_t oldId, int32_t currId);
int32_t tidl_sortDataIds(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_updateOutDataShape(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t startIdx, int32_t layerIndex, sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable);
int32_t tidl_getInLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, int32_t dataId);
int32_t tidl_getOutLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, int32_t dataId);
int32_t tidl_mergeFlattenLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergeBiasLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergePadLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergeBNLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergeReluLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergePoolLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_convertConv2DToIpLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable);
int32_t tidl_copyPCNetToDeviceNet(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, sTIDL_Network_t  &tIDLNetStructure, int32_t layerIndex, int weightsElementSizeInBits);
int32_t tidl_addOutDataLayer(sTIDL_Network_t  &tIDLNetStructure, int32_t tiLayerIndex);
int32_t tidl_fillInDataLayerShape(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, tidl_import_config * params, int32_t tiLayerIndex);
void TIDL_importQuantWriteLayerParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t numLayers, FILE *fp1);
int32_t tidl_addInDataLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, int32_t * dataIndex);
int32_t tidl_mergeDropoutLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_mergeReshapeLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, sTIDL_tfOutRehapeMap_t * sTIDL_tfOutRehapeTable);
int32_t TIDL_tfOutReshapeResize(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_addNormLayerToInData(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex, int32_t * dataIndex, tidl_import_config * params);
int32_t tidl_convertIpLayerInputShape(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_getLayerTypeMapIdx(const char* layerName, TIDL_TFLayerMapping_t* TIDL_TFLayerMap, int32_t tblSize);
int32_t tidl_isLayerType(const char* layerName, int32_t  startLayer, sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, TIDL_TFLayerMapping_t* TIDL_TFLayerMap, int32_t tblSize);
int32_t tidl_convertRelUToBNLayer(sTIDL_OrgNetwork_t  &pOrgTIDLNetStructure, int32_t layerIndex);
int32_t TIDL_isInputLayer(sTIDL_OrgNetwork_t * pOrgTIDLNetStructure,int32_t numLayer, const char *bufName, int32_t layerType);
void tidl_importEltWiseParams(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure, int32_t layerIndex);
int32_t tidl_countUnsupportedLayers(sTIDL_Network_t *pTIDLNetStructure, int32_t numLayers);
int32_t onnx_import(tidl_import_config * params);
int32_t caffe_import( tidl_import_config * params);
int32_t tf_import(tidl_import_config * params);
int32_t tfLite_import(tidl_import_config * params);
void tidl_printTfSupport();
void tidl_printOnnxSupport();
void tidl_printCaffeSupport();
void tidl_printTfLiteSupport();

int32_t tidl_getStringsFromList(char *list, char * names, int strLen);

#endif /*TIDL_IMPORT_COMMONH_ */
