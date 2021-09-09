// sample kernels using runtime API
#include "c7x_tvm_runtime.h"

// Test cases for DMA API
int main()
{
  DMAContext DMAContext(2);

  #if 0
  {
  using A_Buffer = Buffer<Layout<char, 1, 1, 14, 14>>;
  using A_Local  = Buffer<Layout<char, 1, 1, 14, 14>>;
  DMA<A_Buffer, A_Local> A_DMA(DMAContext);
  A_DMA.dump();
  }
  #endif

  #if 0
  {
  using A_Buffer = Buffer<Layout<float, 1, 672, 14, 14>>;
  using A_Local  = Buffer<Layout<float, 1, 672, 14, 14>>;
  DMA<A_Buffer, A_Local> A_DMA(DMAContext);
  A_DMA.dump();
  }
  #endif

  #if 0
  {
  using A_Buffer = Buffer<Layout<float, 1, 672, 14, 14>>;
  using A_Local  = DoubleBuffer<Layout<float, 1,   8, 14, 14>>;
  DMA<A_Buffer, A_Local> A_DMA(DMAContext);
  A_DMA.dump();
  for (int i = 0; i < 84; ++i)
  {
    A_DMA.trigger();
    A_DMA.wait();
  }
  }
  #endif

  #if 0
  {
  using A_Buffer = Buffer<Layout<float, 1, 4, 128, 128>>;
  using A_Local  = DoubleBuffer<Layout<float, 1, 1, 8, 128>>;
  DMA<A_Buffer, A_Local> A_DMA(DMAContext);
  A_DMA.dump();
  for (int i = 0; i < 16; ++i)
  {
    A_DMA.trigger();
    A_DMA.wait();
  }
  }
  #endif

  #if 1
  {
  using ExtBuf = Buffer<Layout<float, 1, 672, 14, 14>>;
  using LocalBuf = DoubleBuffer<Layout<float, 1, 8, 14, 14>>;
  ExtBuf A_buffer, C_buffer;
  LocalBuf A_local, C_local;
  DMA<ExtBuf, LocalBuf> A_DMA(DMAContext, A_buffer, A_local, "A");
  DMA<LocalBuf, ExtBuf> C_DMA(DMAContext, C_local, C_buffer, "C");
  A_DMA.dump();
  C_DMA.dump();
  void *A_ptr = A_DMA.dst_ptr();
  void *C_ptr = A_DMA.src_ptr();

  for (int i = 0; i < 84; ++i)
  {
    A_DMA.copy();
    A_ptr = A_DMA.dst_ptr();
    printf("compute, A_ptr=%p C_ptr=%p\n", A_ptr, C_ptr);
    C_DMA.copy();
    C_ptr = C_DMA.src_ptr();
  }
  }
  #endif
}
