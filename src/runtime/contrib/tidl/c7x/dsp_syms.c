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

/* This file is compiled into a pseudo firmware executable that exports symbols.
 * When TVM C7x deployable module gets built, it will link with this pseudo
 *     firmware, import these symbols, and build into a dynamically linked
 *     library/executable, where these symbols remain as undefined.
 * The real symbol resolution happens at the dynamic loading time with
 *     real symbol addresses obtained at dynamic loading time.
 *
 * Rationale for using a pseudo C7x firmware is that we do not have dependency
 *     on the real C7x firmware when compiling user DL graph/network into
 *     c7x depolyable module (c7x_depoly_mod.out)
 *
 * dsp_syms.c -> dsp_syms.out
 * dsp_syms.out -> dsp_syms.out.obj (encoded as a linkage section)
 * dsp_syms.out: providing symbols, needed during linking c7x_deploy_mod.out
 * dsp_syms.out.obj: compiled and linked into c7x_deploy_mod.out as a section.
 *                   The secion is extracted at loader time to populate the
 *                   dependent symbol table, which gets updated to real
 *                   addresses obtained from the firmware.
 */

#include <stdint.h>

/* psuedo vars for dynamic linking
 */
__declspec(dllexport) void* TIDL_VISION_FXNS;
__declspec(dllexport) void* g_l1_mem_addr;
__declspec(dllexport) void* g_l2_mem_addr;
__declspec(dllexport) void* g_l3_mem_addr;
__declspec(dllexport) uint32_t g_l1_mem_size;
__declspec(dllexport) uint32_t g_l2_mem_size;
__declspec(dllexport) uint32_t g_l3_mem_size;

/* psuedo functions for dynamic linking
 */
__declspec(dllexport) void printf() {}
__declspec(dllexport) void puts() {}
__declspec(dllexport) void vprintf() {}
__declspec(dllexport) void snprintf() {}
__declspec(dllexport) void vsnprintf() {}
__declspec(dllexport) void fputs() {}
__declspec(dllexport) void fflush() {}

__declspec(dllexport) void appMemAlloc() {}
__declspec(dllexport) void appMemFree() {}
__declspec(dllexport) void appUdmaGetObj() {}

__declspec(dllexport) void DmaUtilsAutoInc3d_configure() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_convertTrVirtToPhyAddr() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_deconfigure() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_deinit() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_getContextSize() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_getTrMemReq() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_init() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_prepareTr() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_trigger() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_wait() {}

__declspec(dllexport) void TVM_lockInterrupts() {}
__declspec(dllexport) void TVM_unlockInterrupts() {}
__declspec(dllexport) void TVM_cacheWbInv() {}

