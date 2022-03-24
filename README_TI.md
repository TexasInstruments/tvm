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
| 8.2          | TI.8.2.0, TI.8.2.1   | C7x code generation support                               |
| 8.1          | TIDL\_PSDK\_8.1      |                                                           |
| 8.0          | TIDL\_PSDK\_8.0      |                                                           |
| 7.3          | TIDL\_PSDK\_7.3      | TIDL offload, unsupported layers run on Arm               |

  Suffix "RC" stands for release candidates, suffix "UPDATE" stands for updates
that are still compatible with certain releases.

How to Build for X86\_64 Compilation
------------------------------------
```console
$ # download and install correponding PSDK_RTOS to <PSDKR_PATH>
$ # download and install clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04
$ git clone <this_repo>; cd tvm
$ git checkout <corresponding_tag>
$ git submodule update --init --recursive
$ mkdir build; cd build
$ cmake -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM=/path/to/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
$ make clean; make -j$(nproc)
```

Release Details
---------------

TI.8.2.0, TI.8.2.1
- initial release of C7x code generation support
- merge with neo-ai-tvm 1.10.0

