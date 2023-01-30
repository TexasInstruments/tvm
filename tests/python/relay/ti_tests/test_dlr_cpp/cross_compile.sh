# Set up targetfs if not already  (the psdk version might differ)
# wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-Snl3iJzGTW/08.05.00.08/tisdk-default-image-j721s2-evm.tar.xz
# mkdir targetfs; cd targetfs
# tar xf ../tisdk-default-image-j721s2-evm.tar.xz
# wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/psdkr/pywhl/dlr-1.10.0-py3-none-any.whl
# unzip dlr-1.10.0-py3-none-any.whl -d usr/lib/python3.8/site-packages/

# On x86 host:
${ARM64_GCC_PATH}/bin/aarch64-none-linux-gnu-g++ -O3 \
  -I ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages/dlr/include \
  test_infer.cpp \
  -L ${TARGET_FS_PATH}/usr/lib \
  -L ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages/dlr -ldlr \
  -o cross.out

# On EVM:
# LD_LIBRARY_PATH=/usr/lib/python3.8/site-packages/dlr ./cross.out
