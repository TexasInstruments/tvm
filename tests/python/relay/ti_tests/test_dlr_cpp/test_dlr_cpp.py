# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" Test DLR C/C++ inference API " """

import os
import sys
import subprocess

platform = "J7" if len(sys.argv) <= 1 else sys.argv[1]

try:
  subprocess.run(["sh", "-v", "./native_compile.sh", f"{platform}"], check=True)
except:
  print("Failed: test_dlr_cpp compilation")
  sys.exit(1)

try:
  os.environ['LD_LIBRARY_PATH'] = "/usr/lib/python3.8/site-packages/dlr"
  artifacts_dir = f"../artifacts/mv2_onnx_{platform}_target_tidl_c7x"
  subprocess.run([f"./native_{platform}.out", artifacts_dir], check=True)
except:
  print("Failed: test_dlr_cpp inference")
  sys.exit(1)

sys.exit(0)
