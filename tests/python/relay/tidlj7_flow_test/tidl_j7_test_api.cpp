#include "itidl_rt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

static int g_handle = 0;


static std::vector<char> read_file(const char* filename) {
  std::vector<char> ret;
  std::ifstream s(filename, std::ios::binary | std::ios::in);
  if (!s.is_open()) {
    std::cerr << "Failed to open file " << filename << "\n";
    return ret;
  }

  ret.assign(std::istreambuf_iterator<char>(s), std::istreambuf_iterator<char>());

  return ret;
}

extern "C" {

// Helper functions used by the python harness to test  the API
static int TIDLRT_create_called_ = 0;
int TIDLRT_create_called() { return TIDLRT_create_called_; }
static int TIDLRT_delete_called_ = 0;
int TIDLRT_delete_called() { return TIDLRT_delete_called_; }
static int TIDLRT_invoke_called_ = 0;
int TIDLRT_invoke_called() { return TIDLRT_invoke_called_; }
static int TIDLRT_deactivate_called_ = 0;
int TIDLRT_deactivate_called() { return TIDLRT_deactivate_called_; }
static int TIDLRT_setParamsDefault_called_ = 0;
int TIDLRT_setParamsDefault_called() { return TIDLRT_setParamsDefault_called_; }
static int TIDLRT_setTensorDefault_called_ = 0;
int TIDLRT_setTensorDefault_called() { return TIDLRT_setTensorDefault_called_; }

int32_t TIDLRT_create(sTIDLRT_Params_t* prms, void** handle) {
  TIDLRT_create_called_++;
  *handle = &g_handle;

  // Verify the network data
  auto net_data = read_file("tempDir/subgraph0_net.bin");
  if (std::memcmp(net_data.data(), prms->netPtr, net_data.size()) != 0)
    return 1;

  // Verify the params data
  auto params_data = read_file("tempDir/subgraph0_params_1.bin");
  if (std::memcmp(params_data.data(), prms->ioBufDescPtr, params_data.size())
      != 0)
    return 1;

  return 0;
}

int32_t TIDLRT_delete(void* handle) {
  TIDLRT_delete_called_++;

  if (handle != &g_handle) {
    std::cerr << "Invalid handle passed to TIDLRT_delete\n";
    return 1;
  }

  return 0;
}

int32_t TIDLRT_invoke(void* handle, sTIDLRT_Tensor_t* in[], sTIDLRT_Tensor_t* out[]) {
  TIDLRT_invoke_called_++;
  if (handle != &g_handle) {
    std::cerr << "Invalid handle passed to TIDLRT_invoke\n";
    return 1;
  }

  // Assume just one input.
  sTIDLRT_Tensor_t& in_tensor = *in[0];
  if (std::strcmp((char*) in_tensor.name, "tidl_0_i0")) {
    std::cerr << "Invalid input tensor name, " << in_tensor.name << "\n";
    return 1;
  }

  if (in_tensor.elementType != TIDLRT_Float32) {
    std::cerr << "Invalid element type found, got " << in_tensor.elementType << ", expected "
              << TIDLRT_Float32 << "\n";
    return 1;
  }

  if (in_tensor.numDim != 4) {
    std::cerr << "Invalid input ndim found, got " << in_tensor.numDim << ", expected 4\n";
    return 1;
  }

  decltype(in_tensor.dimValues) golden_dimValues = {1, 2, 2, 3};
  if (std::memcmp(in_tensor.dimValues, golden_dimValues, 4) != 0) {
    std::cerr << "Invalid dimension found\n";
    return 1;
  }

  if (in_tensor.layout != TIDLRT_LT_NCHW) {
    std::cerr << "Invalid layout found\n";
    return 1;
  }

  // No way to check the input data

  sTIDLRT_Tensor_t& out_tensor = *out[0];
  std::cout << "output tensor name: " << (char*) out_tensor.name << std::endl;
  if (out_tensor.name[0] != '\0') {
    std::cerr << "Invalid output tensor name, expected null string\n";
    return 1;
  }

  if (out_tensor.elementType != TIDLRT_Float32) {
    std::cerr << "Invalid element type found, got " << out_tensor.elementType << ", expected "
              << TIDLRT_Float32 << "\n";
    return 1;
  }

  if (out_tensor.numDim != 2) {
    std::cerr << "Invalid output ndim found, got " << out_tensor.numDim << ", expected 2\n";
    return 1;
  }

  decltype(out_tensor.dimValues) out_golden_dimValues = {1, 1001};
  if (std::memcmp(out_tensor.dimValues, out_golden_dimValues, 2) != 0) {
    std::cerr << "Invalid dimension found\n";
    return 1;
  }

  if (out_tensor.layout != TIDLRT_LT_NCHW) {
    std::cerr << "Invalid layout found\n";
    return 1;
  }

  float* d = (float*)out_tensor.ptr;
  #if 0
  d[0] = 20.0f;
  d[1] = 30.0f;
  d[2] = 40.0f;
  #endif
  std::ifstream fin(
           "tempDir/tidl_import_subgraph0.txt0030_00001_01001x00001_float.bin",
           std::ios::binary);
  fin.read((char*) d, 1001*sizeof(float));

  return 0;
}

int32_t TIDLRT_deactivate(void* handle) {
  TIDLRT_deactivate_called_++;
  return 0;
}

int32_t TIDLRT_setParamsDefault(sTIDLRT_Params_t* prms) {
  TIDLRT_setParamsDefault_called_++;
  std::memset(prms, 0, sizeof(*prms));
  return 0;
}

int32_t TIDLRT_setTensorDefault(sTIDLRT_Tensor_t* tensor) {
  TIDLRT_setTensorDefault_called_++;
  std::memset(tensor, 0, sizeof(*tensor));
  return 0;
}

}
