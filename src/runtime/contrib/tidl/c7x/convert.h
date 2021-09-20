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


#ifndef _TIDL_API_CONVERT_H_
#define _TIDL_API_CONVERT_H_

#include <stdint.h>
#include "itidl_ti.h"
#include "itidl_rt.h"

/*! \brief A 32-bit signed value.
 * \ingroup group_basic_features
 */
typedef int32_t  vx_int32;

/*! \brief Sets the standard enumeration type size to be a fixed quantity.
 * \details All enumerable fields must use this type as the container to
 * enforce enumeration ranges and sizeof() operations.
 * \ingroup group_basic_features
 */
typedef int32_t vx_enum;

/*! \brief The enumeration of all status codes.
 * \see vx_status.
 * \ingroup group_basic_features
 */
enum vx_status_e {
    VX_STATUS_MIN                       = -(vx_int32)25,/*!< \brief Indicates the lower bound of status codes in VX. Used for bounds checks only. */
    /* add new codes here */
    VX_ERROR_REFERENCE_NONZERO          = -(vx_int32)24,/*!< \brief Indicates that an operation did not complete due to a reference count being non-zero. */
    VX_ERROR_MULTIPLE_WRITERS           = -(vx_int32)23,/*!< \brief Indicates that the graph has more than one node outputting to the same data object. This is an invalid graph structure. */
    VX_ERROR_GRAPH_ABANDONED            = -(vx_int32)22,/*!< \brief Indicates that the graph is stopped due to an error or a callback that abandoned execution. */
    VX_ERROR_GRAPH_SCHEDULED            = -(vx_int32)21,/*!< \brief Indicates that the supplied graph already has been scheduled and may be currently executing. */
    VX_ERROR_INVALID_SCOPE              = -(vx_int32)20,/*!< \brief Indicates that the supplied parameter is from another scope and cannot be used in the current scope. */
    VX_ERROR_INVALID_NODE               = -(vx_int32)19,/*!< \brief Indicates that the supplied node could not be created.*/
    VX_ERROR_INVALID_GRAPH              = -(vx_int32)18,/*!< \brief Indicates that the supplied graph has invalid connections (cycles). */
    VX_ERROR_INVALID_TYPE               = -(vx_int32)17,/*!< \brief Indicates that the supplied type parameter is incorrect. */
    VX_ERROR_INVALID_VALUE              = -(vx_int32)16,/*!< \brief Indicates that the supplied parameter has an incorrect value. */
    VX_ERROR_INVALID_DIMENSION          = -(vx_int32)15,/*!< \brief Indicates that the supplied parameter is too big or too small in dimension. */
    VX_ERROR_INVALID_FORMAT             = -(vx_int32)14,/*!< \brief Indicates that the supplied parameter is in an invalid format. */
    VX_ERROR_INVALID_LINK               = -(vx_int32)13,/*!< \brief Indicates that the link is not possible as specified. The parameters are incompatible. */
    VX_ERROR_INVALID_REFERENCE          = -(vx_int32)12,/*!< \brief Indicates that the reference provided is not valid. */
    VX_ERROR_INVALID_MODULE             = -(vx_int32)11,/*!< \brief This is returned from <tt>\ref vxLoadKernels</tt> when the module does not contain the entry point. */
    VX_ERROR_INVALID_PARAMETERS         = -(vx_int32)10,/*!< \brief Indicates that the supplied parameter information does not match the kernel contract. */
    VX_ERROR_OPTIMIZED_AWAY             = -(vx_int32)9,/*!< \brief Indicates that the object refered to has been optimized out of existence. */
    VX_ERROR_NO_MEMORY                  = -(vx_int32)8,/*!< \brief Indicates that an internal or implicit allocation failed. Typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    VX_ERROR_NO_RESOURCES               = -(vx_int32)7,/*!< \brief Indicates that an internal or implicit resource can not be acquired (not memory). This is typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    VX_ERROR_NOT_COMPATIBLE             = -(vx_int32)6,/*!< \brief Indicates that the attempt to link two parameters together failed due to type incompatibilty. */
    VX_ERROR_NOT_ALLOCATED              = -(vx_int32)5,/*!< \brief Indicates to the system that the parameter must be allocated by the system.  */
    VX_ERROR_NOT_SUFFICIENT             = -(vx_int32)4,/*!< \brief Indicates that the given graph has failed verification due to an insufficient number of required parameters, which cannot be automatically created. Typically this indicates required atomic parameters. \see vxVerifyGraph. */
    VX_ERROR_NOT_SUPPORTED              = -(vx_int32)3,/*!< \brief Indicates that the requested set of parameters produce a configuration that cannot be supported. Refer to the supplied documentation on the configured kernels. \see vx_kernel_e. This is also returned if a function to set an attribute is called on a Read-only attribute.*/
    VX_ERROR_NOT_IMPLEMENTED            = -(vx_int32)2,/*!< \brief Indicates that the requested kernel is missing. \see vx_kernel_e vxGetKernelByName. */
    VX_FAILURE                          = -(vx_int32)1,/*!< \brief Indicates a generic error code, used when no other describes the error. */
    VX_SUCCESS                          =  0,/*!< \brief No error. */
};

/*! \brief A formal status type with known fixed size.
 * \see vx_status_e
 * \ingroup group_basic_features
 */
typedef vx_enum vx_status;



int32_t TIDLRT_setParamsDefault(sTIDLRT_Params_t *prms);

vx_status cp_data_in_tidlrt_tensor(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *in, void *restrict input_buffer, uint32_t id);
vx_status cp_data_out_tensor_tidlrt(sTIDL_IOBufDesc_t* ioBufDesc, sTIDLRT_Tensor_t *out, void *restrict output_buffer, uint32_t id, float scale);

#endif // _TIDL_API_CONVERT_H_
