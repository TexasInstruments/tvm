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

/* L2SRAM range must be in sync with Platform.xdc */
MEMORY
{
    DDR:    o = 0x80000000 l = 0x19000000
/*
    MSMC:   o = 0x0C000000 l = __MSMC_SIZE__
    L2SRAM: o = __L2SRAM_START__ l = __L2SRAM_SIZE__
*/
}

SECTIONS
{
    .llvmir:  type = COPY

    .gnu.offload_funcs > DDR
    .gnu.offload_vars  > DDR

    .mem_ddr  > DDR
/*
    .mem_msm  > MSMC

    GROUP
    {
        .mem_l2: align(128)
        .ocl_local_overlay:*   align(128) run_start(_ocl_local_overlay_start)
    } > L2SRAM
*/

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
    .dsp_syms_out > DDR
/*  if NOLOAD, then section data not included in the final .out file
    need to find NOLOAD, but keep in file option
    .dsp_syms_out (NOLOAD) : {} > DDR
*/
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

