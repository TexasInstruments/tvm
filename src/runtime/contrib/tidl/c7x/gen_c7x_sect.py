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
"""Embed an input file as a section in C7x assembly"""

import sys

with open(sys.argv[1], "rb") as fi:
  with open(sys.argv[2], "wt") as fo:
    fo.write("\t.sect \".dsp_syms_out\"\n\t.retain")
    nbytes = 0
    byte = fi.read(1)
    while byte:
      val = int.from_bytes(byte, byteorder='little', signed=True)
      if ((nbytes % 10) == 0):
        fo.write(f"\n\t.byte {val}")
      else:
        fo.write(f", {int(val)}")
      nbytes += 1
      byte = fi.read(1)
    fo.write("\n")
