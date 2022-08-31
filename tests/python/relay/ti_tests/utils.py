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
"""Utils used in TVM compilation/inference."""


import os


def disable_outputs():
  """ Redirect stdout/stderr to /dev/null """
  null_fd = os.open(os.devnull, os.O_WRONLY)
  saved_stdout_fd = os.dup(1)
  saved_stderr_fd = os.dup(2)
  os.dup2(null_fd, 1)
  os.dup2(null_fd, 2)
  return [ null_fd, saved_stdout_fd, saved_stderr_fd ]


def restore_outputs(fds):
  """ Restore stdout/stderr to saved fds """
  os.dup2(fds[2], 2)
  os.dup2(fds[1], 1)
  os.close(fds[0])


def get_artifacts_folder(model_name, platform, is_target, w_tidl, w_c7x, batch_size=0):
  """ Return the artifacts folder name based on config """
  artifacts_folder = "artifacts/" + model_name + "_" + platform
  artifacts_folder = artifacts_folder + "_" + ("target" if is_target else "host")
  artifacts_folder = artifacts_folder + "_" + ("tidl" if w_tidl else "notidl")
  artifacts_folder = artifacts_folder + "_" + ("c7x" if w_c7x else "noc7x")
  artifacts_folder = artifacts_folder + (("_bs" + str(batch_size)) if batch_size != 0 else "")
  return artifacts_folder

