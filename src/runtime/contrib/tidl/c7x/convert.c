/*
 *
 * Copyright (c) 2017 Texas Instruments Incorporated
 *
 * All rights reserved not granted herein.
 *
 * Limited License.
 *
 * Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
 * license under copyrights and patents it now or hereafter owns or controls to make,
 * have made, use, import, offer to sell and sell ("Utilize") this software subject to the
 * terms herein.  With respect to the foregoing patent license, such license is granted
 * solely to the extent that any such patent is necessary to Utilize the software alone.
 * The patent license shall not apply to any combinations which include this software,
 * other than combinations with devices manufactured by or for TI ("TI Devices").
 * No hardware patent is licensed hereunder.
 *
 * Redistributions must preserve existing copyright notices and reproduce this license
 * (including the above copyright notice and the disclaimer and (if applicable) source
 * code license limitations below) in the documentation and/or other materials provided
 * with the distribution
 *
 * Redistribution and use in binary form, without modification, are permitted provided
 * that the following conditions are met:
 *
 * *       No reverse engineering, decompilation, or disassembly of this software is
 * permitted with respect to any software provided in binary form.
 *
 * *       any redistribution and use are licensed by TI for use only with TI Devices.
 *
 * *       Nothing shall obligate TI to provide you with source code for the software
 * licensed and provided to you in object code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any object code compiled from the source code
 * and any resulting derivative works, are licensed by TI for use only with TI Devices.
 *
 * Neither the name of Texas Instruments Incorporated nor the names of its suppliers
 *
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * DISCLAIMER.
 *
 * THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "convert.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#define APP_ASSERT(x)               assert((x))

#define MAX_TENSOR_DIMS         (4u)
#define TIDL_MAX_PARAMS         (16u) //PC-- need it?

#define ABS_FLT(a) ((a) > 0)?(a):(-(a))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

static uint32_t   tidlrt_debuglevel = 0;

/**
  @struct  sTIDLRTTB_IntHandle_t
  @brief   This structure is internal handle for TIDL RT API wrapper
  */
typedef struct
{
    /** TIRL RT Craete time configuration parametes*/
    sTIDLRT_Params_t rtPrms;
} IntHandle_t;



static int debug_printf(const char *fmt, ...)
{
    va_list ap;
    int ret = 0;

    if(!tidlrt_debuglevel)
        goto out;

    va_start(ap, fmt);
    ret = vprintf(fmt, ap);
    va_end(ap);

out:
    return ret;
}

int32_t TIDLRT_setParamsDefault(sTIDLRT_Params_t *prms)
{
    int32_t status                  = 0;
    prms->netPtr                    = NULL;
    prms->ioBufDescPtr              = NULL;
    prms->net_capacity              = 0;
    prms->io_capacity               = 0;
#ifdef x86_64
    prms->flowCtrl                  = 1;
#else
    prms->flowCtrl                  = 0;
#endif
    prms->traceLogLevel             = 2;
    prms->traceWriteLevel           = 0;
    prms->traceBaseName             = 0;
    prms->TIDLWriteBinToFile        = NULL;
    prms->TIDLReadBinFromFile       = NULL;
    prms->TIDLVprintf               = vprintf;
    prms->quantRangeExpansionFactor = 1;
    prms->quantRangeUpdateFactor    = 0;
    prms->stats                     = NULL;

    debug_printf("TIDL_RT_OVX: Set default TIDLRT params done\n");
    return status;
}

int32_t TIDLRT_setTensorDefault(sTIDLRT_Tensor_t *tensor)
{
    int32_t status                  = 0;
    //tensor->name[]                = {0};
    tensor->elementType             = 0;
    tensor->numDim                  = 0;
    //tensor->dimValues             = {0, 0, 0};
    //tensor->pitch                 = {0, 0};
    //tensor->padValues[]           = {0};
    tensor->ptr                     = NULL;
    tensor->dataOffset              = 0;
    tensor->layout                  = 0;
    tensor->zeroPoint               = 0;
    tensor->scale                   = 0;
    tensor->memType                 = 0;
    debug_printf("TIDL_RT_OVX: Set default TIDLRT tensor done\n");
    return status;

}


static inline uint32_t uclamp(float f, uint32_t min_val, uint32_t max_val)
{
    uint32_t val = (uint32_t) f;
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    return val;
}

static inline int32_t clamp(float f, int32_t min_val, int32_t max_val)
{
    int32_t val = (int32_t) f;
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    return val;
}

static inline uint64_t uclamp_64(float f, uint64_t min_val, uint64_t max_val)
{
    uint64_t val = (uint64_t) f;
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    return val;
}

static inline int64_t clamp_64(float f, int64_t min_val, int64_t max_val)
{
    int64_t val = (int64_t) f;
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    return val;
}

static inline uint8_t sat_uint8(float f)
{
    return uclamp(f, 0, 255);
}

static inline int8_t sat_int8(float f)
{
    return clamp(f, -128, 127);
}

static inline uint16_t sat_uint16(float f)
{
    return uclamp(f, 0, 65535);
}

static inline int16_t sat_int16(float f)
{
    return clamp(f, -32768, 32767);
}

static inline uint32_t sat_uint32(float f)
{
    return uclamp_64(f, 0ull, 4294967295ull);
}

static inline int32_t sat_int32(float f)
{
    return clamp_64(f, -2147483648ll, 2147483647ll);
}

#ifdef HOST_EMULATION
#define restrict 
#endif

vx_status cp_nchw_float_ushort(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, uint16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0*h*w + i1*w + i2] * scale;
                if (data < 0.0f) data = 0.0f;
                if (data > 65535.0f) data = 65535.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (uint16_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nchw_float_short(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, int16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0*h*w + i1*w + i2] * scale;
                if (data < -32768.0f) data = -32768.0f;
                if (data > 32767.0f) data = 32767.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (int16_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nchw_float_uchar(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, uint8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0*h*w + i1*w + i2] * scale;
                if (data < 0.0f) data = 0.0f;
                if (data > 255.0f) data = 255.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (uint8_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nchw_float_char(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, int8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0*h*w + i1*w + i2] * scale;
                if (data < -128.0f) data = -128.0f;
                if (data > 127.0f) data = 127.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (int8_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nhwc_float_ushort(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, uint16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0 + i1*w*c + i2*c] * scale;
                if (data < 0.0f) data = 0.0f;
                if (data > 65535.0f) data = 65535.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (uint16_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nhwc_float_short(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, int16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0 + i1*w*c + i2*c] * scale;
                if (data < -32768.0f) data = -32768.0f;
                if (data > 32767.0f) data = 32767.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (int16_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nhwc_float_uchar(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, uint8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0 + i1*w*c + i2*c] * scale;
                if (data < 0.0f) data = 0.0f;
                if (data > 255.0f) data = 255.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (uint8_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nhwc_float_char(int32_t c, int32_t h, int32_t w,
                          float* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, int8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = rtPtr[i0 + i1*w*c + i2*c] * scale;
                if (data < -128.0f) data = -128.0f;
                if (data > 127.0f) data = 127.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (int8_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_nhwc_uchar_char(int32_t c, int32_t h, int32_t w,
                          uint8_t* restrict rtPtr,
                          int32_t offset, int32_t cp, int32_t lp,
                          float scale, int8_t* restrict ivPtr,
                          int in_zp, float in_scale)
{
    int32_t i0, i1, i2;
    scale = scale / in_scale;
    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = (rtPtr[i0 + i1*w*c + i2*c] - in_zp) * scale;
                if (data < -128.0f) data = -128.0f;
                if (data > 127.0f) data = 127.0f;
                ivPtr[offset + i0*cp + i1*lp + i2] = (int8_t)(data);
            }
        }
    }
    return VX_SUCCESS;
}


#ifndef HOST_EMULATION
vx_status cp_data_in_tidlrt_tensor(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *in, void *restrict input_buffer, uint32_t id)
#else
vx_status cp_data_in_tidlrt_tensor(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *in, void *input_buffer, uint32_t id)
#endif
{
    int32_t c, h, w, lp, cp;
    int32_t i0, i1, i2, offset, idx;
    float data;
    vx_status status = VX_SUCCESS;

    c = ioBufDesc->inNumChannels[id];
    w = ioBufDesc->inWidth[id];
    h = ioBufDesc->inHeight[id];
    lp = w + ioBufDesc->inPadL[id] + ioBufDesc->inPadR[id];
    cp = ioBufDesc->inChannelPitch[id];
    void *rtPtr  = in->ptr;
    void * ivPtr = input_buffer;
    offset = lp*ioBufDesc->inPadT[id] + ioBufDesc->inPadL[id];
    float scale = ioBufDesc->inTensorScale[id];
    float inScale = in->scale;
    int32_t zp = in->zeroPoint;
    int32_t layout = in->layout;

    if(in->elementType == TIDL_SinglePrecFloat)
    {
        inScale = 1.0;
        zp = 0;
    }

    // zp = 0, inScale = 1.0f
    if (layout == TIDLRT_LT_NCHW && in->elementType == TIDLRT_Float32)
    {
        if (ioBufDesc->inElementType[id] ==  TIDL_UnsignedShort)
            return cp_nchw_float_ushort(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint16_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_SignedShort)
            return cp_nchw_float_short(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int16_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_UnsignedChar)
            return cp_nchw_float_uchar(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint8_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_SignedChar)
            return cp_nchw_float_char(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr);
    }

    if (layout == TIDLRT_LT_NHWC && in->elementType == TIDLRT_Float32)
    {
        if (ioBufDesc->inElementType[id] ==  TIDL_UnsignedShort)
            return cp_nhwc_float_ushort(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint16_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_SignedShort)
            return cp_nhwc_float_short(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int16_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_UnsignedChar)
            return cp_nhwc_float_uchar(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint8_t*) ivPtr);
        if (ioBufDesc->inElementType[id] ==  TIDL_SignedChar)
            return cp_nhwc_float_char(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr);
    }

    // zp, inScale
    if (layout == TIDLRT_LT_NHWC && in->elementType == TIDLRT_Uint8)
    {
        if (ioBufDesc->inElementType[id] == TIDL_SignedChar)
            return cp_nhwc_uchar_char(c, h, w, (uint8_t*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr, zp, inScale);
    }

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                if(layout == TIDLRT_LT_NCHW)
                {
                    idx = i0*h*w + i1*w + i2;
                }
                else
                {
                    idx = i0 + i1*w*c + i2*c;
                }

                if(in->elementType ==  TIDLRT_Uint8)
                {
                    data =   ((uint8_t*)rtPtr)[idx];
                }
                else if(in->elementType ==  TIDLRT_Int8)
                {
                    data = ((int8_t*)rtPtr)[idx];
                }
                else if(in->elementType ==  TIDLRT_Uint16)
                {
                    data = ((uint16_t*)rtPtr)[idx];
                }
                else if(in->elementType ==  TIDLRT_Int16)
                {
                    data = ((int16_t*)rtPtr)[idx];
                }
                else if(in->elementType == TIDLRT_Uint32)
                {
                    data = ((uint32_t*)rtPtr)[idx];
                }
                else if(in->elementType == TIDLRT_Int32)
                {
                    data = ((int32_t*)rtPtr)[idx];
                }
                else if(in->elementType == TIDLRT_Float32)
                {
                    data = ((float*)rtPtr)[idx];
                }
                else
                {
                    return VX_FAILURE;
                }

                data = ((data - zp)/inScale) * scale;

                if(ioBufDesc->inElementType[id] ==  TIDL_UnsignedChar)
                {
                    ((uint8_t*)ivPtr)[offset + i0*cp + i1*lp + i2] = sat_uint8(data);
                }
                else if(ioBufDesc->inElementType[id] ==  TIDL_SignedChar)
                {
                    ((int8_t*)ivPtr)[offset + i0*cp + i1*lp + i2] = sat_int8(data);
                }
                else if(ioBufDesc->inElementType[id] ==  TIDL_UnsignedShort)
                {
                    ((uint16_t*)ivPtr)[offset + i0*cp + i1*lp + i2] = sat_uint16(data);
                }
                else if(ioBufDesc->inElementType[id] ==  TIDL_SignedShort)
                {
                    ((int16_t*)ivPtr)[offset + i0*cp + i1*lp + i2] = sat_int16(data);
                }
                else if(ioBufDesc->inElementType[id] == TIDL_SinglePrecFloat)
                {
                    ((float*)ivPtr)[offset + i0*cp + i1*lp + i2] = data;
                }
                else
                {
                    return VX_FAILURE;
                }
            }
        }
    }

    return status;
}

vx_status cp_out_nchw_ushort_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, uint16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0*h*w + i1*w + i2] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nchw_short_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, int16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0*h*w + i1*w + i2] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nchw_uchar_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, uint8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0*h*w + i1*w + i2] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nchw_char_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, int8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0*h*w + i1*w + i2] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nhwc_ushort_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, uint16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0 + i1*w*c + i2*c] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nhwc_short_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, int16_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0 + i1*w*c + i2*c] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nhwc_uchar_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, uint8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0 + i1*w*c + i2*c] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nhwc_char_float(int32_t c, int32_t h, int32_t w,
                              float* restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, int8_t* restrict ivPtr)
{
    int32_t i0, i1, i2;
    float inv_scale = 1.0f / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale;
                rtPtr[i0 + i1*w*c + i2*c] = data;
            }
        }
    }
    return VX_SUCCESS;
}

vx_status cp_out_nhwc_char_uchar(int32_t c, int32_t h, int32_t w,
                              uint8_t * restrict rtPtr,
                              int32_t offset, int32_t cp, int32_t lp,
                              float scale, int8_t* restrict ivPtr,
                              int32_t out_zp, float out_scale)
{
    int32_t i0, i1, i2;
    float inv_scale = out_scale / scale;

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                float data = ivPtr[offset + i0*cp + i1*lp + i2] * inv_scale
                           + out_zp;
                if (data < 0.0f) data = 0.0f;
                if (data > 255.0f) data = 255.0f;
                rtPtr[i0 + i1*w*c + i2*c] = (uint8_t) data;
            }
        }
    }
    return VX_SUCCESS;
}

#ifndef HOST_EMULATION
vx_status cp_data_out_tensor_tidlrt(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *out, void *restrict output_buffer, uint32_t id, float scale)
#else
vx_status cp_data_out_tensor_tidlrt(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *out, void *output_buffer, uint32_t id, float scale)
#endif
{
    int32_t c, h, w, lp, cp;
    int32_t i0, i1, i2, offset, zp, idx;
    float data, outScale;
    vx_status status = VX_SUCCESS;
    //volatile int debug = 1;
    //while(debug);


    c = ioBufDesc->outNumChannels[id];
    w = ioBufDesc->outWidth[id];
    h = ioBufDesc->outHeight[id];
    lp = w + ioBufDesc->outPadL[id] + ioBufDesc->outPadR[id];
    cp = ioBufDesc->outChannelPitch[id];
    void * rtPtr = out->ptr;
    void *ivPtr  = output_buffer;
    offset = lp*ioBufDesc->outPadT[id] + ioBufDesc->outPadL[id];
    int32_t layout = out->layout;
    if(out->elementType == TIDL_SinglePrecFloat)
    {
        out->scale = 1.0;
        out->zeroPoint = 0;
    }

    if(out->scale == -1)
    {
        out->scale = scale;
    }

    outScale = out->scale;
    zp = out->zeroPoint;

    // zp = 0, outScale = 1.0f
    if (layout == TIDLRT_LT_NCHW && out->elementType == TIDLRT_Float32)
    {
        if (ioBufDesc->outElementType[id] ==  TIDL_UnsignedShort)
            return cp_out_nchw_ushort_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint16_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_SignedShort)
            return cp_out_nchw_short_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int16_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_UnsignedChar)
            return cp_out_nchw_uchar_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint8_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_SignedChar)
            return cp_out_nchw_char_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr);
    }

    if (layout == TIDLRT_LT_NHWC && out->elementType == TIDLRT_Float32)
    {
        if (ioBufDesc->outElementType[id] ==  TIDL_UnsignedShort)
            return cp_out_nhwc_ushort_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint16_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_SignedShort)
            return cp_out_nhwc_short_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int16_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_UnsignedChar)
            return cp_out_nhwc_uchar_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (uint8_t*) ivPtr);
        if (ioBufDesc->outElementType[id] ==  TIDL_SignedChar)
            return cp_out_nhwc_char_float(c, h, w, (float*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr);
    }

    // zp, outScale
    if (layout == TIDLRT_LT_NHWC && out->elementType == TIDLRT_Uint8)
    {
        if (ioBufDesc->outElementType[id] == TIDL_SignedChar)
            return cp_out_nhwc_char_uchar(c, h, w, (uint8_t*) rtPtr, offset, cp, lp, scale, (int8_t*) ivPtr, zp, outScale);
    }

    for(i0 = 0; i0 < c; i0++)
    {
        for(i1 = 0; i1 < h; i1++)
        {
            for(i2 = 0; i2 < w; i2++)
            {
                if(ioBufDesc->outElementType[id] ==  TIDL_UnsignedChar)
                {
                    data = ((uint8_t*)ivPtr)[offset + i0*cp + i1*lp + i2];
                }
                else if(ioBufDesc->outElementType[id] ==  TIDL_SignedChar)
                {
                    data = ((int8_t*)ivPtr)[offset + i0*cp + i1*lp + i2];
                }
                else if(ioBufDesc->outElementType[id] ==  TIDL_UnsignedShort)
                {
                    data = ((uint16_t*)ivPtr)[offset + i0*cp + i1*lp + i2];
                }
                else if(ioBufDesc->outElementType[id] ==  TIDL_SignedShort)
                {
                    data = ((int16_t*)ivPtr)[offset + i0*cp + i1*lp + i2];
                }
                else if(ioBufDesc->outElementType[id] == TIDL_SinglePrecFloat)
                {
                    data = ((float*)ivPtr)[offset + i0*cp + i1*lp + i2];
                }
                else
                {
                    return VX_FAILURE;
                }

                if((outScale != scale) || (zp != 0))
                {
                    data = (((data/scale) * outScale) + zp);
                }

                if(layout == TIDLRT_LT_NCHW)
                {
                    idx = i0*h*w + i1*w + i2;
                }
                else
                {
                    idx = i0 + i1*w*c + i2*c;
                }
                if(out->elementType ==  TIDLRT_Uint8)
                {
                    ((uint8_t*)rtPtr)[idx] = sat_uint8(data);
                }
                else if(out->elementType ==  TIDLRT_Int8)
                {
                    ((int8_t*)rtPtr)[idx] = sat_int8(data);
                }
                else if(out->elementType ==  TIDLRT_Uint16)
                {
                    ((uint16_t*)rtPtr)[idx] = sat_uint16(data);
                }
                else if(out->elementType ==  TIDLRT_Int16)
                {
                    ((int16_t*)rtPtr)[idx] = sat_int16(data);
                }
                else if(out->elementType ==  TIDLRT_Uint32)
                {
                    ((uint32_t*)rtPtr)[idx] = sat_uint32(data);
                }
                else if(out->elementType ==  TIDLRT_Int32)
                {
                    ((int32_t*)rtPtr)[idx] = sat_int32(data);
                }
                else if(out->elementType == TIDLRT_Float32)
                {
                    ((float*)rtPtr)[idx] = data;
                }
                else
                {
                    return VX_FAILURE;
                }
            }
        }
    }
    return status;
}
