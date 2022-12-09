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

--ram_model
--display_error_number
--diag_suppress=10290
--diag_suppress=10291
--priority
/* c7x firmware or compiled from c with exported symbols in dynamic symtab,
   symbol resolution performed at loader time. C7x firmware can be built
   after dsp_syms.out is built. */
-ldsp_syms.out
--dynamic=lib
/* compiled from asm with symbols encoded as absolute addresses in firmware,
   symbol resolution performed at link time. C7x firmware needs to be built
   before dsp_syms.obj is built. */
/*
-ldsp_syms.obj
--dynamic=exe
*/
-lrts7100_le.lib
--relocatable
--no_entry_point
--warn_sections
-x
-heap 0x0


MEMORY
{
    DDR:    o = 0x80000000 l = 0x19000000
}

SECTIONS
{
    .dsp_syms_out: type = COPY

    .gnu.offload_funcs > DDR
    .gnu.offload_vars  > DDR

    .mem_ddr  > DDR

    GROUP
    {
       .rodata:
       .neardata:
       .bss:
    } > DDR

    .text       > DDR
    .cinit      > DDR
    .const      > DDR
    .data       > DDR
    .switch     > DDR
    .far        > DDR
    .fardata    > DDR
    .plt        > DDR
    .sysmem     > DDR
}

/* import these symbols from C7x firmware */
--import=printf
--import=puts
--import=vprintf
--import=snprintf
--import=vsnprintf
--import=fputs
--import=fflush

--import=TIDL_VISION_FXNS
--import=g_l1_mem_addr
--import=g_l2_mem_addr
--import=g_l3_mem_addr
--import=g_l1_mem_size
--import=g_l2_mem_size
--import=g_l3_mem_size
--import=appMemAlloc
--import=appMemFree
--import=appUdmaGetObj

--import=DmaUtilsAutoInc3d_configure
--import=DmaUtilsAutoInc3d_convertTrVirtToPhyAddr
--import=DmaUtilsAutoInc3d_deconfigure
--import=DmaUtilsAutoInc3d_deinit
--import=DmaUtilsAutoInc3d_getContextSize
--import=DmaUtilsAutoInc3d_getTrMemReq
--import=DmaUtilsAutoInc3d_init
--import=DmaUtilsAutoInc3d_prepareTr
--import=DmaUtilsAutoInc3d_trigger
--import=DmaUtilsAutoInc3d_wait

--import=TVM_lockInterrupts
--import=TVM_unlockInterrupts
--import=TVM_cacheWbInv

/* workaround to prevent linking printf/puts from rts7100_le.lib */
--symbol_map=printf=__dummy_printf
--symbol_map=puts=__dummy_puts
--symbol_map=vprintf=__dummy_vprintf
--symbol_map=snprintf=__dummy_snprintf
--symbol_map=vsnprintf=__dummy_vsnprintf
--symbol_map=fputs=__dummy_fputs
--symbol_map=fflush=__dummy_fflush

/* redirect all fprintf to stdout */
--symbol_map=fprintf=TVM_C7x_fprintf

