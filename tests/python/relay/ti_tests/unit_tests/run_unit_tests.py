#!/usr/bin/env python3

import os
import sys
import subprocess
from unit_utils import platform

unit_tests = [
"broadcast_mul_odd_blocks_odd_iters.py",
"sigmoid_approx.py",
"resize_nchw_1x2.py",
"resize_index_simplify.py",
"conv2d_1x2_stride.py",
"sigmoid_tidl.py",
"scatter_nd.py",
"topk.py",
"argsort.py",
# tanh support has been deprecated in TIDL
#"tanh_tidl.py",
"conv2d_transpose_IOHW.py",
"multiply_large_last_dim.py",
"multiply_2_dim_broadcast.py",
# Disable, known failure with PSDK 9.0
#"priorities.py",
"SOC_envvar.py",
"scatter_nd_tidl.py",
"scatter_nd_extern.py",
"ya_scatter_nd.py",
]

failed_tests = []

if __name__ == "__main__":
  for test in unit_tests:
    # Disable, known failure with PSDK 9.1, 9.2, 10.0
    # [C7x_1 ] This core (16) is different than CLEC RTMAP CPU (1) programming for channel 0
    if platform == "J784S4" and test in ["multiply_large_last_dim.py", "broadcast_mul_odd_blocks_odd_iters.py"]:
      continue

    print(f"\n\n##### Running test: {test} #####\n")
    try:
      subprocess.run(["python3", test, "--platform", platform], check=True)
    except:
      failed_tests.append(test)

  if not failed_tests:
    print("Pass")
    sys.exit(0)
  else:
    print("Failed tests:")
    for test in failed_tests:
      print(f"  {test}")
    sys.exit(1)
