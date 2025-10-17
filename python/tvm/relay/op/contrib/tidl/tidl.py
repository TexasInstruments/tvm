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
# pylint: disable=invalid-name, unused-argument
"""TI C7X support for TIDL."""
import logging
import tvm
import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

logger = logging.getLogger("TVMC")

# Global variable to store TIDL configuration from tvmc
_global_tidl_config = None

def partition_for_c7x(mod, params=None, mod_name="default", **opts):
    """Partition a Relay graph for TI C7x DSP + MMA.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to be partitioned
    params : Dict[str, tvm.nd.NDArray]
        Parameters for the Relay module
    opts: Dict[str, Any]
        Additional parameters passed from composite target system

    Returns
    -------
    tvm.IRModule
        The partitioned relay module.
    """
    from tvm.relay.backend.contrib.tidl import TIOffloadCompiler
    from tvm.driver.tvmc import TVMCException

    # Get configuration from global variable set by tvmc
    global _global_tidl_config
    if _global_tidl_config is None:
        raise TVMCException(
            "TIDL configuration not found. Make sure to use tvmc with --target-tidl-* options."
        )

    config = _global_tidl_config.copy()

    # Extract graph_input_list from config
    graph_input_list = config.pop("graph_input_list", [])

    if not graph_input_list:
        logger.warning("No calibration data provided. TIDL quantization may be suboptimal.")

    # create TI offload compiler with configuration
    # Separate constructor parameters from delegate_options
    constructor_params = {"platform", "tidl_tools_path", "enable_tidl_offload", "reuse_tidl_artifacts"}
    delegate_params = {"artifacts_folder", "tensor_bits", "enable_c7x_codegen", "compile_for_device", "deny_list", "od_meta_arch_type", "od_meta_layers_names_list"}

    # Build constructor arguments
    constructor_args = {k: v for k, v in config.items() if k in constructor_params}

    # Build delegate_options from remaining parameters
    delegate_options = {k: v for k, v in config.items() if k in delegate_params}

    # Convert object detection target attributes to the format expected by TIOffloadCompiler
    if "od_meta_arch_type" in delegate_options and delegate_options["od_meta_arch_type"] != -1:
        delegate_options["object_detection:meta_arch_type"] = delegate_options.pop("od_meta_arch_type")
    if "od_meta_layers_names_list" in delegate_options and delegate_options["od_meta_layers_names_list"]:
        delegate_options["object_detection:meta_layers_names_list"] = delegate_options.pop("od_meta_layers_names_list")

    if delegate_options:
        constructor_args["delegate_options"] = delegate_options

    compiler = TIOffloadCompiler(**constructor_args)

    partitioned_mod, status = compiler.enable(mod, params, graph_input_list)

    if status <= 0:
        if status <= 0:
            raise TVMCException(
                "TIDL compilation was not performed. This may be due to missing tidl tools "
                "or max_num_tidl_subgraphs set to 0."
            )
        elif status < 0:
            raise TVMCException(
                "TIDL compilation failed."
            )

    return partitioned_mod
