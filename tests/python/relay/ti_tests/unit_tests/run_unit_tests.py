#!/usr/bin/env python3

import os
import sys
import subprocess

unit_tests = [
"broadcast_mul_odd_blocks_odd_iters.py",
]

failed_tests = []

if __name__ == "__main__":
  for test in unit_tests:
    try:
      subprocess.run(["python3", test], check=True)
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
