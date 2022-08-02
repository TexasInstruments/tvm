TVM+TIDL: TVM with TIDL Offload
===============================

We leverage TVM's BYOC (Bring Your Own Codegen) mechanism to offload subgraphs to
TIDL (Texas Instruments' Deep Learning library) for accelerated execution on TI's
J7 family of SoCs.  Layers unsupported by TIDL are left with TVM code generation
and runtime.


Branches
--------
  * tidl-j7 - This is the release branch
  * tidl-j7-dev - This is the internal development branch
    (in sync with TIDL development branch)


Tags
----
  TVM+TIDL releases are synchronized with TI's PSDK (Processor SDK) releases.
The following are the TVM+TIDL tags compatible with target file system in PSDK releases.

| PSDK release | TVM+TIDL release tag | Key features                                              |
|--------------|----------------------|-----------------------------------------------------------|
| 8.4          | TIDL\_PSDK\_8.4      | Bug fixes, J721S2 support, debug support                  |
| 8.2          | TIDL\_PSDK\_8.2, TI.8.2.{0, 1, 2} | C7x code generation support                  |
| 8.1          | TIDL\_PSDK\_8.1      |                                                           |
| 8.0          | TIDL\_PSDK\_8.0      |                                                           |
| 7.3          | TIDL\_PSDK\_7.3      | TIDL offload, unsupported layers run on Arm               |

  Suffix "RC" stands for release candidates, suffix "UPDATE" stands for updates
that are still compatible with certain releases.


How to Build x86\_64 Package for Compilation
--------------------------------------------
```console
# download and install corresponding PSDK_RTOS to <PSDKR_PATH>
# download and install clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04
# https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

git clone <this_repo>; cd tvm
git checkout <corresponding_tag>
git submodule update --init --recursive

mkdir build_x86; cd build_x86
cmake -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="/path/to/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
make clean; make -j$(nproc)

# build python package in $TVM_HOME/python/dist
cd ..; rm -fr build; ln -s build_x86 build
cd python; python3 ./setup.py bdist_wheel; ls dist
```


Building the TVM Graph Executor (Runtime) for aarch64
-----------------------------------------------------
```console
export ARM64_GCC_PATH=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

mkdir build_aarch64; cd build_aarch64
cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
make clean; make -j$(nproc) runtime
```


Release Details
---------------

#### TIDL\_PSDK\_8.4
- Including bug fixes in TI.8.2.2
- J721S2 support
- Merge with upstream neo-ai tvm 1.11.2
- Tensor debug support for TVM Arm runtime and TVM C7x runtime

#### TI.8.2.2
- Bug fixes in C7x code generation
    - Fix streaming engine pass for loops containing multiple accesses to same tensor (CODEGEN-9810)
    - Add nop() function to support Reshape op in TVM C runtime (CODEGEN-9795)
    - Avoid overriding generic op strategy in "hls.py" (back-ported from upstream TVM)
    - Add C7x strategy for concatenate instead of using generic strategy (CODEGEN-9794)
    - Do not vectorize inner loop with iterations less than vector length (CODEGEN-9841)
    - Do not vectorize if loop body contains call (CODEGEN-9424)
    - Fix vectorization factor to use largest data type in computation (CODEGEN-9848)
    - Add 64-bit integer support in streaming engine config (CODEGEN-9909)
    - Fix TVM C runtime to support more than 255 functions/layers (CODEGEN-9981)
    - Return tvm runtime create failure to OpenVX node (CODEGEN-8941)
    - Apply tiling and dma schedule only to broadcast ops in injective.py (CODEGEN-10004)

#### TI.8.2.0, TI.8.2.1, TIDL\_PSDK\_8.2
- initial release of C7x code generation support
- merge with neo-ai-tvm 1.10.0

