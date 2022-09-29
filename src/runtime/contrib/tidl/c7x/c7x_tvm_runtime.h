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

//
// This API provides an abstraction layer that the C7x-specific TVM
// code generator can use to target C7x-specific features.
//
// DMA API
//   The concept is to use a C++ class to capture the layout of both the
//   external and local buffers, and automatically set up the DMA based
//   on the layouts. The model includes blocking (splitting up the transfer
//   into multiple blocks) and double buffering (using a pair of buffers so
//   one can be filling/emptying while the other is used for compute).
//   The implementation uses Yuan's DMA utility wrapper functions.
//
// Streaming API
//   Provides a simple class to enable configuration of the streaming
//   engine/streaming address generators.

#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <cstdint>
#include <string>
#include <sstream>
#include <type_traits>
#include "c7x.h"

#define max(a, b) __max((a), (b))
#define min(a, b) __min((a), (b))

// If MODEL_DMA is true, the DMA utilities are stubbed out
#if !MODEL_DMA
#include "tidl_api_mem.h"
#include "tvm_tidl_dmautils.h"
#else
// Define stubs for the DMA utility wrapper functions
#include <stdlib.h>
// from tvm_tidl_dmautils.h
typedef enum {
  TVMTIDL_DMAUTILS_CHANNEL_0 = 0,
  TVMTIDL_DMAUTILS_CHANNEL_1,
  TVMTIDL_DMAUTILS_CHANNEL_MAX
} tvmtidlDmaUtilsChannel;

// from pdk/packages/ti/drv/udma/dmautils/include/dmautils_autoincrement_3d.h
typedef enum{
    TVMTIDL_DMAUTILSAUTOINC3D_SYNC_1D = 0,
    TVMTIDL_DMAUTILSAUTOINC3D_SYNC_2D = 1,
    TVMTIDL_DMAUTILSAUTOINC3D_SYNC_3D = 2,
    TVMTIDL_DMAUTILSAUTOINC3D_SYNC_4D = 3
  }tvmtidlDmaUtilsAutoInc3d_SyncType;

uint8_t* tvm_tidl_l2_scratch_alloc(int32_t size)
  { return (uint8_t*)malloc(size); }
void tvm_tidl_l2_scratch_reset() {}
uint8_t* tvm_tidl_dmautils_init(int32_t num_channels, uint8_t *pTrMem_chs[])
  { return nullptr; }
int32_t tvm_tidl_configure_channel(uint8_t *dmaUtilsContext,
     int32_t ch, uint8_t *pTrMem_chs[],
     uint8_t *srcPtr, uint8_t *dstPtr, tvmtidlDmaUtilsAutoInc3d_SyncType syncType,
     uint16_t sicnt0, uint16_t sicnt1, uint16_t sicnt2, uint16_t sicnt3,
                       int32_t sdim1,   int32_t sdim2,   int32_t sdim3,
     uint16_t dicnt0, uint16_t dicnt1, uint16_t dicnt2, uint16_t dicnt3,
                       int32_t ddim1,   int32_t ddim2,   int32_t ddim3)
  { return 0; }
int32_t tvm_tidl_dmautils_deinit(uint8_t *dmaUtilsContext,
     int32_t num_channels, uint8_t *pTrMem_chs[])
  { return 0; }
void tvm_tidl_dmautils_trigger(uint8_t *dmaUtilscontext, int32_t channel)
  {}
void tvm_tidl_dmautils_wait(uint8_t *dmaUtilscontext, int32_t channel)
  {}
#endif

extern "C" {
  extern void tvmcrt_exit(int ecode);
  extern int32_t TVM_lockInterrupts();
  extern void    TVM_unlockInterrupts(int32_t);
  extern void tvm_tidl_argsort_nms(float *input, int *sort_num, int *output);
}

//---------------------------------------------------------------------------------
// DMAContext stores state information for the DMA Utils package, used
// as a handle bewteen the C++ model and the Utils package
class DMAContext
{
public:
  DMAContext(int n) : nchannels(n)
  {
    if (nchannels > 0)
      dmaUtilsContext = tvm_tidl_dmautils_init(nchannels, pTrMem_chs);
  }
  ~DMAContext()
  {
    if (nchannels > 0)
      tvm_tidl_dmautils_deinit(dmaUtilsContext, nchannels, pTrMem_chs);
  }
  int allocate_channel()
  {
    if (next_avail_channel >= nchannels)
      return -1;
    return next_avail_channel++;
  }
  uint8_t **get_pTrMem() { return pTrMem_chs; }
  uint8_t *get_dmaUtilsContext() { return dmaUtilsContext; }
private:
  uint8_t *pTrMem_chs[10];
  int next_avail_channel = 0;
  int nchannels = 0;
  uint8_t *dmaUtilsContext = nullptr;
};

//---------------------------------------------------------------------------------
// Allocator for L2 Memory. Declare one of these objects within the
// scope of L2 allocation.
class AllocL2Context
{
  public:
  AllocL2Context() { tvm_tidl_l2_scratch_reset(); }
  ~AllocL2Context() {}
  /* Only called in the generated C7x function for a layer */
  void *allocate(unsigned size)
  {
    void *ptr = tvm_tidl_l2_scratch_alloc(size);
    if (ptr == NULL)
    {
      printf("AllocL2Context.allocate failed for size: %d\n", size);
      tvmcrt_exit(-1);
    }
    return ptr;
  }
};

//---------------------------------------------------------------------------------
// Allocator for DDR Scratch Memory. Declare one of these objects within the
// scope of DDR allocation.  Opportunistically allocating in L2 if can,
// otherwise, use the linear DDR scratch memory allocator.
class AllocDDRContext
{
  public:
  AllocDDRContext() { tvm_tidl_l2_scratch_reset(); tvm_tidl_ddr_scratch_reset(); }
  ~AllocDDRContext() {}
  void *allocate(unsigned size)
  {
    void *ptr = tvm_tidl_l2_scratch_alloc(size);
    if (ptr == NULL)
    {
      ptr = tvm_tidl_ddr_scratch_alloc(size);
    }
    if (ptr == NULL)
    {
      printf("AllocDDRContext.allocate failed for size: %d\n", size);
      tvmcrt_exit(-1);
    }
    return ptr;
  }
};

//---------------------------------------------------------------------------------
// Critical section context.  Disable and restore interrupts at entry and exit.
class CriticalSectionContext
{
public:
  CriticalSectionContext()  { old_state = TVM_lockInterrupts(); }
  ~CriticalSectionContext() { TVM_unlockInterrupts(old_state); }
private:
  int32_t old_state;
};


//---------------------------------------------------------------------------------
// Layout represents the type of a multi-dimensional tensor in memory. It's a completely
// compile-time representation -- there is no runtime state.
template<typename Elem, int D3=1, int D2=1, int D1=1, int D0=1>
class Layout
{
public:
   using ElementType = Elem;
   static const int ElemSize = sizeof(Elem);
   static const int Dim3 = D3;
   static const int Dim2 = D2;
   static const int Dim1 = D1;
   static const int Dim0 = D0;
   static const int size = ElemSize * D0 * D1 * D2 * D3;
   static void dump()
   {
     #if DEBUG
     printf("Layout: [%d][%d][%d][%d] ElemSize=%d\n",
            D3, D2, D1, D0, ElemSize);
     #endif
   }
};

//---------------------------------------------------------------------------------
// A Buffer combines a Layout with an actual location. There are two buffer
// types: a single buffer, and a double buffer which is two adjacent single
// buffers.
// BufferBase is the base class common to both types.
template<typename BufferType>
class BufferBase
{
public:
  using Type = BufferType;
  // The constructor binds the buffer to an actual location.
  BufferBase(void *p) : ptr(p) {}
  BufferBase() : ptr(nullptr) {}
  void* get() { return ptr; }
  // This method is called after the DMA completes each block of a transfer.
  virtual void sync() {}
protected:
  void* ptr = nullptr;
};

// Single buffer
template<typename Layout_>
class Buffer : public BufferBase<Buffer<Layout_>>
{
  public:
  using Layout = Layout_;
  using Base = BufferBase<Buffer<Layout>>;
  using Base::Base;
  static const bool isDB = false;
  Buffer() : Base() {}
  Buffer(void *p) : Base(p) {}
  static const int size = Layout::size;
};

// Double buffer. The state member indicates the ping/pong status.
template<typename Layout_>
class DoubleBuffer : public BufferBase<DoubleBuffer<Layout_>>
{
  public:
  using Layout = Layout_;
  using Base = BufferBase<DoubleBuffer<Layout>>;
  using Base::Base;
  static const bool isDB = true;
  static const int size = 2 * Layout::size;
  void* get()
  {
    return (state == 0) ? Base::ptr
                        : (void *)((char *)Base::ptr + Layout::size);
  }
  void sync() { state ^= 1; }
  private:
  int state = 0;
};

//---------------------------------------------------------------------------------
#if DEBUG && MODEL_DMA
class APSim;  // forward declaration
#endif

// An AccessPattern is a canonical way of expressing the sequence of
// accesses in a multidimensional array, a la streaming engine or DMA.
class AccessPattern
{
  // APAxis represents the accesses along a single axis. The access proceeds
  // 'count' times, with addresses increasing by 'stride' bytes after each
  // access.  If the stride is 0 the axis rewinds each time. The 'extent' is
  // simply the length in bytes of the whole axis.
  struct APAxis
  {
    APAxis() : count(1), stride(0), extent(0) {}
    APAxis(uint32_t c, uint32_t s) :
      count(c), stride(s), extent((uint64_t)c*s) {}
    APAxis(const APAxis& src) :
      count(src.count), stride(src.stride), extent(src.extent) {}
    uint32_t count;    // number of items (element, row, block, etc)
    uint32_t stride;   // offset in bytes to next item
    uint64_t extent;   // total advance on this axis
  };

public:
  static const int maxaxes = 5;
  APAxis axes[maxaxes];
  int naxes = 0;
  // sync_axis: delimits blocks, for blocked (multi-stage) transfers
  int sync_axis = -1;
  // buffer offset where the transfer starts
  int offset = 0;
  #if DEBUG && MODEL_DMA
  // Access pattern simulator, for testing
  APSim *sim = nullptr;
  #endif

public:
  AccessPattern() {}

  // Construct default access pattern, given a layout. This results in a simple
  // linear access pattern. Generally this will flatten to a single dimension
  // unless some of the dimensions are too big to represent.
  template<typename Layout>
  AccessPattern(const Layout* layout)
  {
     add_dim(Layout::Dim0, Layout::ElemSize);
     add_dim(Layout::Dim1, Layout::Dim0 * Layout::ElemSize);
     add_dim(Layout::Dim2, Layout::Dim1 * Layout::Dim0 * Layout::ElemSize);
     add_dim(Layout::Dim3, Layout::Dim2 * Layout::Dim1 *
                           Layout::Dim0 * Layout::ElemSize);
  }

  // Add an additional dimension to the AP. Dimensions should be added
  // from inner to outer.
  void add_dim(uint32_t count, uint32_t stride)
  {
    // First dimension has implied stride of 1 byte, so adjust count
    if (naxes == 0)
    {
      count *= stride;
      stride = 1;
    }
    // If new dimension's stride matches previous dimension's extent, and
    // the count does not overflow, flatten the new dimension into the
    // previous one.
    if (naxes > 0 && (naxes-1) != sync_axis &&
        stride == axes[naxes-1].extent &&
	count * axes[naxes-1].count <= USHRT_MAX)
    {
      axes[naxes-1].count *= count;
      axes[naxes-1].extent *= count;
    }
    else if (naxes+1 == maxaxes)
       // error
       ;
    else
      axes[naxes++] = APAxis(count, stride);
  }
  void add_sync() { if (naxes >0) sync_axis = naxes-1; }

  // total extent (== extent of last dimension)
  uint64_t current_extent() { return naxes ? axes[naxes-1].extent : 0; }

  void dump()
  {
    #if DEBUG
    printf("count    stride\n");
    for(int i = 0; i < 4; ++i)
    {
      printf("%4d     %4d", axes[i].count, axes[i].stride);
      if (i == sync_axis) printf("    (sync)");
      printf("\n");
    }
    #endif
  }
};

//---------------------------------------------------------------------------------
// APSim: simulate an access pattern, for testing
#if DEBUG && MODEL_DMA
class APSim
{
  int i0 = 0;
  int i1 = 0;
  int i2 = 0;
  int i3 = 0;
  const AccessPattern& AP;
public:
  APSim(const AccessPattern& AP) : AP(AP) { reset(); }
  void reset() { i0=i1=i2=i3=0; }
  int currpos()
  {
    return i3*AP.axes[3].stride +
           i2*AP.axes[2].stride +
           i1*AP.axes[1].stride +
           i0*AP.axes[0].stride;
  }
  // Advance by one block
  uint32_t advance()
  {
    uint32_t nbytes = 0;
    bool sync = false;
    for(;;)
    {
      // advance i0
      if (i0++ < AP.axes[0].count)
        nbytes += AP.axes[0].stride;
      else
      {
        // advance i1
        if (AP.sync_axis == 0) sync = true;
          i0 = 0;
        if (++i1 >= AP.axes[1].count)
        {
          // advance i2
          if (AP.sync_axis == 1) sync = true;
          i1 = 0;
          if (++i2 >= AP.axes[2].count)
          {
            // advance i3
            if (AP.sync_axis == 2) sync = true;
            i2 = 0;
            if (++i3 >= AP.axes[3].count)
              sync = true;
          }
        }
      }
      if (sync) return nbytes;
    }
  }
};
#endif  // DEBUG && MODEL_DMA

//---------------------------------------------------------------------------------
// DMA represents an agent to transfer data from a SrcBuffer to a DstBuffer.
// The buffer types are used to set up access patterns for both buffers,
// which in turn are used to setup the DMA hardware itself.
// This can happen at init time since the setup is based strictly on the
// types.
template <typename SrcBuffer, typename DstBuffer>
class DMA
{
protected:
   // DMAUtils interface
   DMAContext& context;
   int channel;
   const char *name = "";

   // Src and Dst
   AccessPattern srcAP;
   AccessPattern dstAP;
   SrcBuffer* src = nullptr;
   DstBuffer* dst = nullptr;

   // State, during transfers
   int seq = 0;
   int dma_blocks = 1;
public:
   // Setup, with buffers and dma paramaters supplied
   DMA(DMAContext &ctx, SrcBuffer& s, DstBuffer& d,
           int num_blocks, int sync_axis, int soffset, int doffset,
           int sicnt0, int sicnt1, int sicnt2, int sicnt3, int sstride1, int sstride2, int sstride3,
           int dicnt0, int dicnt1, int dicnt2, int dicnt3, int dstride1, int dstride2, int dstride3,
           const char* n="") :
     context(ctx), channel(ctx.allocate_channel()), name(n), dma_blocks(num_blocks)
   {
     srcAP.sync_axis = sync_axis;
     srcAP.offset = soffset;
     srcAP.axes[0].count = sicnt0;
     srcAP.axes[1].count = sicnt1;
     srcAP.axes[2].count = sicnt2;
     srcAP.axes[3].count = sicnt3;
     srcAP.axes[1].stride = sstride1;
     srcAP.axes[2].stride = sstride2;
     srcAP.axes[3].stride = sstride3;
     dstAP.offset = doffset;
     dstAP.axes[0].count = dicnt0;
     dstAP.axes[1].count = dicnt1;
     dstAP.axes[2].count = dicnt2;
     dstAP.axes[3].count = dicnt3;
     dstAP.axes[1].stride = dstride1;
     dstAP.axes[2].stride = dstride2;
     dstAP.axes[3].stride = dstride3;

     bind(s,d);
   }

   // Attach actual buffer instances to this DMA object.
   void bind(SrcBuffer& s, DstBuffer& d)
   {
     src = &s;
     dst = &d;

     // TODO: this tvm_tidl API requires buffer addresses to be supplied,
     // therefore cannot be invoked until we have actual buffers. However,
     // if there were a separate API to supply the addresses then the rest
     // of the config could happen at init time.
     tvmtidlDmaUtilsAutoInc3d_SyncType sync = TVMTIDL_DMAUTILSAUTOINC3D_SYNC_4D;
     switch(srcAP.sync_axis)
     {
        case 0: sync = TVMTIDL_DMAUTILSAUTOINC3D_SYNC_1D; break;
        case 1: sync = TVMTIDL_DMAUTILSAUTOINC3D_SYNC_2D; break;
        case 2: sync = TVMTIDL_DMAUTILSAUTOINC3D_SYNC_3D; break;
        case 3: sync = TVMTIDL_DMAUTILSAUTOINC3D_SYNC_4D; break;
     }

     tvm_tidl_configure_channel(
       context.get_dmaUtilsContext(), channel, context.get_pTrMem(),
       (uint8_t*)src->get() + srcAP.offset, (uint8_t*)dst->get() + dstAP.offset, sync,
       srcAP.axes[0].count,
       srcAP.axes[1].count,
       srcAP.axes[2].count,
       srcAP.axes[3].count,
       srcAP.axes[1].stride,
       srcAP.axes[2].stride,
       srcAP.axes[3].stride,
       dstAP.axes[0].count,
       dstAP.axes[1].count,
       dstAP.axes[2].count,
       dstAP.axes[3].count,
       dstAP.axes[1].stride,
       dstAP.axes[2].stride,
       dstAP.axes[3].stride
     );
   }

   // Trigger transfer of next block
   void trigger()
   {
     #if DEBUG && MODEL_DMA
     printf("trigger %s[%d]: %d --> %d, channel=%d\n",
            name, seq, srcAP.sim->currpos(), dstAP.sim->currpos(), channel);
     #endif
     tvm_tidl_dmautils_trigger(context.get_dmaUtilsContext(), channel);
     ++seq;
   }

   // Wait for transfer of last-triggered block to finish
   void wait()
   {
     tvm_tidl_dmautils_wait(context.get_dmaUtilsContext(), channel);
     #if DEBUG && MODEL_DMA
     auto src_bytes = srcAP.sim->advance();
     auto dst_bytes = dstAP.sim->advance();
     assert(src_bytes == dst_bytes);
     printf("wait %s[%d]: ... copied %d bytes\n", name, seq-1, src_bytes);
     #endif
   }

   // Client API to invoke copy function.
   void copy()
   {
      copy_helper<SrcBuffer, DstBuffer>();
   }

   // Helper function to do synchronized copy. There are three versions:
   // no-double-buffering, double-buffer-dst, and double-buffer-src.
   // The selection is made at compile time based on the buffer types
   // via enable-if.
   template <typename SB, typename DB,
             std::enable_if_t<!SB::isDB && !DB::isDB, bool> = true>
   void copy_helper()
   {
     // Direct buffer-to-buffer copy. Trigger the transfer, and wait for
     // it to finish.
     trigger();
     wait();
   }

   // Specialization for double-buffered "copy-in": copy 1 block from
   // external src to double-buffered local dst
   template <typename SB, typename DB,
             std::enable_if_t<!SB::isDB && DB::isDB, bool> = true>
   void copy_helper()
   {
     #if DEBUG
     printf("cp in seq=%d\n", seq);
     #endif
     // if first block, trigger
     if (seq == 0)
     {
        dst->sync();
        trigger();
     }
     // wait for current block
     wait();
     dst->sync();
     // if not last block, trigger next block
     if (seq < dma_blocks)
       trigger();
   }

   // Specialization for double-buffered "copy-out": copy 1 block from
   // double-buffered local src to external dst
   template <typename SB, typename DB,
             std::enable_if_t<SB::isDB && !DB::isDB, bool> = true>
   void copy_helper()
   {
     #if DEBUG
     printf("cp out seq=%d\n", seq);
     #endif
     // if not first block, wait for previous block to finish
     if (seq != 0)
       wait();
     // trigger current block
     src->sync();
     trigger();
     // if last block, wait for current block to finish
     if (seq >= dma_blocks)
       wait();
   }

   void *src_ptr() { return src->get(); }
   void *dst_ptr() { return dst->get(); }

   void dump()
   {
     #if DEBUG
      printf("\nDMA config:\n");
      printf("nblocks=%d\n", dma_blocks);
      printf("src AccessPattern\n");
      srcAP.dump();
      printf("dst AccessPattern\n");
      dstAP.dump();
     #endif
   }
};

// DMA factory function -- enables type inference for construction
template <typename SrcBuffer, typename DstBuffer>
DMA<SrcBuffer, DstBuffer>
create_DMA(DMAContext &ctx, SrcBuffer& s, DstBuffer& d,
           int num_blocks, int sync_axis, int soffset, int doffset,
           int sicnt0, int sicnt1, int sicnt2, int sicnt3, int sstride0, int sstride1, int sstride2,
           int dicnt0, int dicnt1, int dicnt2, int dicnt3, int dstride0, int dstride1, int dstride2,
           const char* n="")
{
  return DMA<SrcBuffer, DstBuffer>(ctx, s, d, num_blocks, sync_axis, soffset, doffset,
           sicnt0, sicnt1, sicnt2, sicnt3, sstride0, sstride1, sstride2,
           dicnt0, dicnt1, dicnt2, dicnt3, dstride0, dstride1, dstride2,
           n);
}

//---------------------------------------------------------------------------------
// Streaming Engine / Streaming Address Generator API

//  SE_element_type_flag - convert byte size to return SE element type flag
template <int nbytes> struct SE_element_type_flag;
template <> struct SE_element_type_flag<1>
   { static const __SE_ELETYPE val = __SE_ELETYPE_8BIT; };
template <> struct SE_element_type_flag<2>
   { static const __SE_ELETYPE val = __SE_ELETYPE_16BIT; };
template <> struct SE_element_type_flag<4>
   { static const __SE_ELETYPE val = __SE_ELETYPE_32BIT; };
template <> struct SE_element_type_flag<8>
   { static const __SE_ELETYPE val = __SE_ELETYPE_64BIT; };

/*-----------------------------------------------------------------------------
* SE_veclen_flag - convert number of bytes to SE vector length flag
*----------------------------------------------------------------------------*/
template <int bytes> struct SE_veclen_flag;
template <> struct SE_veclen_flag<1>
   { static const __SE_VECLEN val = __SE_VECLEN_1ELEM; };
template <> struct SE_veclen_flag<2>
   { static const __SE_VECLEN val = __SE_VECLEN_2ELEMS; };
template <> struct SE_veclen_flag<4>
   { static const __SE_VECLEN val = __SE_VECLEN_4ELEMS; };
template <> struct SE_veclen_flag<8>
   { static const __SE_VECLEN val = __SE_VECLEN_8ELEMS; };
template <> struct SE_veclen_flag<16>
   { static const __SE_VECLEN val = __SE_VECLEN_16ELEMS; };
template <> struct SE_veclen_flag<32>
   { static const __SE_VECLEN val = __SE_VECLEN_32ELEMS; };
template <> struct SE_veclen_flag<64>
   { static const __SE_VECLEN val = __SE_VECLEN_64ELEMS; };

/*-----------------------------------------------------------------------------
* SA_veclen_flag - convert number of elements to SA vector length flag
*----------------------------------------------------------------------------*/
template <int bytes> struct SA_veclen_flag;
template <> struct SA_veclen_flag<1>
   { static const __SA_VECLEN val = __SA_VECLEN_1ELEM; };
template <> struct SA_veclen_flag<2>
   { static const __SA_VECLEN val = __SA_VECLEN_2ELEMS; };
template <> struct SA_veclen_flag<4>
   { static const __SA_VECLEN val = __SA_VECLEN_4ELEMS; };
template <> struct SA_veclen_flag<8>
   { static const __SA_VECLEN val = __SA_VECLEN_8ELEMS; };
template <> struct SA_veclen_flag<16>
   { static const __SA_VECLEN val = __SA_VECLEN_16ELEMS; };
template <> struct SA_veclen_flag<32>
   { static const __SA_VECLEN val = __SA_VECLEN_32ELEMS; };
template <> struct SA_veclen_flag<64>
   { static const __SA_VECLEN val = __SA_VECLEN_64ELEMS; };

// SE Configuration: Full dimmensionality support for C7x {ICNT0:ICNT5} and {DIM1:DIM5}
template <typename Type, int Veclen,
  unsigned icnt0, unsigned icnt1, unsigned icnt2, unsigned icnt3, unsigned icnt4,
  unsigned icnt5, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned dim5>
class SEConfig
{
public:
  SEConfig() : se_params(__gen_SE_TEMPLATE_v1())
  {
    se_params.ELETYPE  = SE_element_type_flag<sizeof(Type)>::val;
    se_params.VECLEN   = SE_veclen_flag<Veclen>::val;
    se_params.DIMFMT   = __SE_DIMFMT_6D; // assigning to 6D always has no h/w overhead
    se_params.ICNT0    = icnt0;
    se_params.ICNT1    = icnt1;
    se_params.ICNT2    = icnt2;
    se_params.ICNT3    = icnt3;
    se_params.ICNT4    = icnt4;
    se_params.ICNT5    = icnt5;
    se_params.DIM1     = dim1;
    se_params.DIM2     = dim2;
    se_params.DIM3     = dim3;
    se_params.DIM4     = dim4;
    se_params.DIM5     = dim5;
  }
  __SE_TEMPLATE_v1 params() const { return se_params; }
private:
  __SE_TEMPLATE_v1 se_params;
};

// SA Configuration: Full dimmensionality support for C7x {ICNT0:ICNT5} and {DIM1:DIM5}
template <typename Type, int Veclen,
          unsigned icnt0, unsigned icnt1, unsigned icnt2, unsigned icnt3,
          unsigned icnt4, unsigned icnt5,  unsigned dim1, unsigned dim2, unsigned dim3,
          unsigned dim4, unsigned dim5>
class SAConfig
{
public:
  SAConfig() : sa_params(__gen_SA_TEMPLATE_v1())
  {
    sa_params.VECLEN   = SA_veclen_flag<Veclen>::val;
    sa_params.DIMFMT   = __SA_DIMFMT_6D; // assigning to 6D always has no h/w overhead
    sa_params.ICNT0    = icnt0;
    sa_params.ICNT1    = icnt1;
    sa_params.ICNT2    = icnt2;
    sa_params.ICNT3    = icnt3;
    sa_params.ICNT4    = icnt4;
    sa_params.ICNT5    = icnt5;
    sa_params.DIM1     = dim1;
    sa_params.DIM2     = dim2;
    sa_params.DIM3     = dim3;
    sa_params.DIM4     = dim4;
    sa_params.DIM5     = dim5;
  }
  __SA_TEMPLATE_v1 params() const { return sa_params; }
private:
  __SA_TEMPLATE_v1 sa_params;
};
