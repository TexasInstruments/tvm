/* This file is compiled into a pseudo firmware executable that exports symbols.
 * When TVM C7x deployable module gets built, it will link with this pseudo
 *     firmware, import these symbols, and build into a dynamically linked
 *     library/executable, where these symbols remain as undefined.
 * The real symbol resolution happens at the dynamic loading time with
 *     real symbol addresses obtained at dynamic loading time.
 */

/* psuedo vars for dynamic linking
 */
__declspec(dllexport) void* TIDL_VISION_FXNS;
__declspec(dllexport) void* g_l1_mem_addr;
__declspec(dllexport) void* g_l2_mem_addr;
__declspec(dllexport) void* g_l3_mem_addr;

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

