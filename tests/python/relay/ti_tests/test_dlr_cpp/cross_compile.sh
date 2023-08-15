# Set up targetfs if not already  (the psdk version might differ)
# wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-Snl3iJzGTW/09.00.00.08/tisdk-adas-image-j721s2-evm.tar.xz
# mkdir targetfs; cd targetfs
# tar xf ../tisdk-adas-image-j721s2-evm.tar.xz
# If usr/lib/python3.10/site-packages/dlr does not exist, copy from EVM:
# scp -r root@evm:/usr/lib/python3.10/site-packages/dlr usr/lib/python3.10/site-packages/

# On x86 host:
${ARM64_GCC_PATH}/bin/aarch64-none-linux-gnu-g++ -O3 \
  -I ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages/dlr/include \
  test_infer.cpp \
  -L ${TARGET_FS_PATH}/usr/lib \
  -L ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages/dlr -ldlr \
  -o cross.out

# On EVM:
# LD_LIBRARY_PATH=/usr/lib/python3.10/site-packages/dlr ./cross.out
