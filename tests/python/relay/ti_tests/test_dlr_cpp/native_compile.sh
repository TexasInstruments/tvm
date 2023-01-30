# On EVM:
g++ -O3 \
  -I /usr/lib/python3.8/site-packages/dlr/include \
  test_infer.cpp \
  -L /usr/lib/python3.8/site-packages/dlr -ldlr \
  -o native.out

# LD_LIBRARY_PATH=/usr/lib/python3.8/site-packages/dlr ./native.out
