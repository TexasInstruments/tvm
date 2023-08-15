# On EVM:
platform=$1
g++ -O3 \
  -I /usr/lib/python3.10/site-packages/dlr/include \
  test_infer.cpp \
  -L /usr/lib/python3.10/site-packages/dlr -ldlr \
  -o native_${platform}.out

# LD_LIBRARY_PATH=/usr/lib/python3.10/site-packages/dlr ./native_${platform}.out
