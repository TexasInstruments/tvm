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
# pylint: disable=unused-argument
"""
Provides support to compile networks both AOT and JIT.
"""
import logging
import os.path
import re
import itertools
from copy import deepcopy
from typing import Any, Optional, Dict, List, Union, Callable, Sequence
from pathlib import Path
from collections import defaultdict

import tvm
from tvm import autotvm, auto_scheduler
from tvm import relay
from tvm.driver.tvmc.registry import generate_registry_args, reconstruct_registry_entity
from tvm.ir.instrument import PassInstrument, PassTimingInstrument, PassPrintingInstrument
from tvm.ir.memory_pools import WorkspaceMemoryPools
from tvm.target import Target
from tvm.relay.backend import Executor, Runtime
from tvm.relay.analysis.operations_distribution import analyze_operations_distribution
from tvm.relay.transform.suffixes import tag_suffixes

from . import composite_target, frontends, TVMCException
from .model import TVMCModel, TVMCPackage
from .main import register_parser
from .target import target_from_cli, generate_target_args, reconstruct_target_args
from .pass_config import parse_configs
from .pass_list import parse_pass_list_str
from .transform import generate_transform_args, parse_graph_transform_args, apply_graph_transforms
from .shape_parser import parse_shape_string
from .workspace_pools import generate_workspace_pools_args, workspace_pools_recombobulate

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_compile_parser(subparsers, _, json_params):
    """Include parser for 'compile' subcommand"""

    parser = subparsers.add_parser("compile", help="compile a model.")
    parser.set_defaults(func=drive_compile)
    parser.add_argument(
        "--cross-compiler",
        default="",
        help="the cross compiler to generate target libraries, e.g. 'aarch64-linux-gnu-gcc'.",
    )
    parser.add_argument(
        "--cross-compiler-options",
        default="",
        help="the cross compiler options to generate target libraries, e.g. '-mfpu=neon-vfpv4'.",
    )
    # Begin TI
    parser.add_argument(
        "--enable-tidl-offload",
        type=int,
        choices=[0, 1],
        default=1,
        help="enable TIDL offload (default: 1, used when --target=tidl).",
    )
    parser.add_argument(
        "--compile-for-device",
        type=int,
        choices=[0, 1],
        default=1,
        help="compile for device (aarch64) instead of host (x86) (default: 1, used when --target=tidl).",
    )
    parser.add_argument(
        "--c7x-codegen",
        type=int,
        choices=[0, 1],
        default=0,
        help="enable C7x code generation (default: 0, used when --target=tidl).",
    )
    parser.add_argument(
        "--tidl-calibration-input",
        type=str,
        default=None,
        help="path to calibration input .npz file (required when --enable-tidl-offload=1, used when --target=tidl).",
    )
    parser.add_argument(
        "--tidl-config",
        type=str,
        default=None,
        help="path to YAML config file containing compile_options (used when --target=tidl).",
    )
    # End TI
    generate_transform_args(parser)
    parser.add_argument(
        "--dump-code",
        metavar="FORMAT",
        default="",
        help="comma separated list of formats to export the input model, e.g. 'asm,ll,tir,relay'.",
    )
    parser.add_argument(
        "--dump-offloads",
        default="",
        help="output a mapping of which operations of the initial Relay "
        "will be transferred to which backend, indicating the composite "
        "that includes those operations, "
        "e.g. '--dump-offloads -' to dump to the console, "
        "e.g. '--dump-offloads <path_to_file>' to dump to the file. "
        "If not presented, no output is done. ",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="module.tar",
        help="output the compiled module to a specified archive. Defaults to 'module.tar'.",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        choices=["so", "mlf"],
        default="so",
        help="output format. Use 'so' for shared object or 'mlf' for Model Library Format "
        "(only for microTVM targets). Defaults to 'so'.",
    )
    parser.add_argument(
        "--pass-config",
        action="append",
        metavar=("name=value"),
        help="configurations to be used at compile time. This option can be provided multiple "
        "times, each one to set one configuration value, "
        "e.g. '--pass-config relay.backend.use_auto_scheduler=0', "
        "e.g. '--pass-config tir.add_lower_pass=opt_level1,pass1,opt_level2,pass2'.",
    )

    generate_target_args(parser)
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        default="",
        help="path to an auto-tuning log file by AutoTVM. If not presented, "
        "the fallback/tophub configs will be used.",
    )
    generate_registry_args(parser, Executor, "graph")
    generate_registry_args(parser, Runtime, "cpp")

    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity.")
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file.")
    parser.add_argument(
        "-O",
        "--opt-level",
        default=3,
        type=int,
        choices=range(0, 4),
        metavar="[0-3]",
        help="specify which optimization level to use. Defaults to '3'.",
    )
    parser.add_argument(
        "--input-shapes",
        help="specify non-generic shapes for model to run, format is "
        '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]".',
        type=parse_shape_string,
        default=None,
    )
    parser.add_argument(
        "--disabled-pass",
        help="disable specific passes, comma-separated list of pass names.",
        type=parse_pass_list_str,
        default="",
    )
    parser.add_argument(
        "--module-name",
        default="default",
        help="The output module name. Defaults to 'default'.",
    )
    parser.add_argument(
        "--print-pass-times",
        action="store_true",
        help="print compilation time per pass",
    )
    parser.add_argument(
        "--print-ir-before",
        help="print IR before each named pass of a comma-separated list of pass names."
        "e.g. '--print-ir-before [tir.SplitHostDevice,tir.ConvertSSA]' ",
        default="",
    )
    parser.add_argument(
        "--print-ir-after",
        help="print IR after each named pass of a comma-separated list of pass names."
        "e.g. '--print-ir-after [tir.SplitHostDevice,tir.ConvertSSA]' ",
        default="",
    )
    for one_entry in json_params:
        parser.set_defaults(**one_entry)

    generate_workspace_pools_args(parser)


def drive_compile(args):
    """Invoke tvmc.compiler module with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.

    Returns
    -------
    int
        Zero if successfully completed

    """

    if not os.path.isfile(args.FILE):
        raise TVMCException(
            f"Input file '{args.FILE}' doesn't exist, is a broken symbolic link, or a directory."
        )

    # Begin TI
    # Early exit for tidl target - use lightweight direct compilation path
    if args.target == "tidl":
        return drive_compile_tidl(args)
    # End TI

    tvmc_model = frontends.load_model(args.FILE, args.model_format, args.input_shapes)

    dump_code = [x.strip() for x in args.dump_code.split(",")] if args.dump_code else None

    dump_offloads = args.dump_offloads if args.dump_offloads else ""

    additional_targets = reconstruct_target_args(args)
    workspace_pools_target, extra_targets = target_from_cli(args.target, additional_targets)
    transform_args = parse_graph_transform_args(args)

    compile_model(
        tvmc_model,
        args.target,
        opt_level=args.opt_level,
        executor=reconstruct_registry_entity(args, Executor),
        runtime=reconstruct_registry_entity(args, Runtime),
        tuning_records=args.tuning_records,
        package_path=args.output,
        cross=args.cross_compiler,
        cross_options=args.cross_compiler_options,
        output_format=args.output_format,
        dump_code=dump_code,
        dump_offloads=dump_offloads,
        target_host=None,
        disabled_pass=args.disabled_pass,
        pass_context_configs=args.pass_config,
        mod_name=args.module_name,
        additional_target_options=additional_targets,
        workspace_pools=(
            workspace_pools_recombobulate(args, [workspace_pools_target], extra_targets)
        ),
        print_pass_times=args.print_pass_times,
        print_ir_before=args.print_ir_before,
        print_ir_after=args.print_ir_after,
        **transform_args,
    )

    return 0


# Begin TI
def _get_model_input_details(model_path: str) -> List[Dict[str, Any]]:
    """Get input tensor details from ONNX model.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing 'name', 'shape', and 'dtype' for each input.
    """
    import numpy as np
    import onnx

    if not model_path.endswith('.onnx'):
        raise TVMCException(f"Only ONNX models are supported, got: {model_path}")

    model = onnx.load(model_path)
    input_details = []
    for inp in model.graph.input:
        name = inp.name
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in inp.type.tensor_type.shape.dim]
        dtype_map = {
            1: np.float32,   # FLOAT
            2: np.uint8,     # UINT8
            3: np.int8,      # INT8
            6: np.int32,     # INT32
            7: np.int64,     # INT64
            10: np.float16,  # FLOAT16
        }
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)
        input_details.append({"name": name, "shape": shape, "dtype": dtype})
    return input_details


def _load_calibration_data(
    npz_path: str,
    input_details: List[Dict[str, Any]],
    num_frames: int
) -> List[Dict[str, Any]]:
    """Load calibration data from .npz file.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing calibration data.
    input_details : List[Dict[str, Any]]
        List of input tensor details from the model.
    num_frames : int
        Number of calibration frames to load.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, each mapping input names to numpy arrays.
    """
    import numpy as np

    calib_data = np.load(npz_path)
    calibration_list = []

    # Build list of calibration inputs
    for frame_idx in range(num_frames):
        input_dict = {}
        for inp in input_details:
            name = inp['name']
            dtype = inp['dtype']
            expected_shape = inp['shape']

            # Try to find matching data in npz file
            # The npz might have data as single array or per-frame arrays
            if name in calib_data:
                data = calib_data[name]
                # If single frame repeated, use it directly
                if len(data.shape) == len(expected_shape):
                    if data.shape != tuple(expected_shape):
                        logger.warning(
                            f"Shape mismatch for input '{name}': expected {expected_shape}, "
                            f"got {data.shape}. Will attempt to use anyway."
                        )
                    input_dict[name] = data.astype(dtype)
                # If multiple frames stacked, extract the specific frame
                elif data.shape[0] > frame_idx:
                    frame_data = data[frame_idx]
                    if frame_data.shape != tuple(expected_shape):
                        logger.warning(
                            f"Shape mismatch for input '{name}' frame {frame_idx}: "
                            f"expected {expected_shape}, got {frame_data.shape}. Will attempt to use anyway."
                        )
                    input_dict[name] = frame_data.astype(dtype)
                else:
                    # Reuse first frame if not enough frames
                    input_dict[name] = data[0].astype(dtype)
            else:
                # If name not found, try to use first array in npz
                logger.warning(
                    f"Input name '{name}' not found in npz file. "
                    f"Available names: {list(calib_data.keys())}. Attempting to use first array."
                )
                arrays = list(calib_data.values())
                if arrays:
                    data = arrays[0]
                    if len(data.shape) == len(expected_shape):
                        input_dict[name] = data.astype(dtype)
                    elif data.shape[0] > frame_idx:
                        input_dict[name] = data[frame_idx].astype(dtype)
                    else:
                        input_dict[name] = data[0].astype(dtype)
                else:
                    raise TVMCException(
                        f"No calibration data found in {npz_path}. "
                        f"The npz file appears to be empty or has no valid arrays."
                    )

        calibration_list.append(input_dict)

    return calibration_list


def drive_compile_tidl(args):
    """Invoke tidl.compile_model directly for TIDL compilation

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.

    Returns
    -------
    int
        Zero if successfully completed

    """
    import yaml
    import numpy as np
    from tvm.contrib import tidl

    # args.FILE now points to the model file
    model_path = args.FILE

    if not os.path.isfile(model_path):
        raise TVMCException(
            f"Model file '{model_path}' doesn't exist or is not a file."
        )

    logger.info(f"Model path: {model_path}")

    # Load YAML config if provided
    delegate_options = {}
    if args.tidl_config:
        config_file = args.tidl_config
        if not os.path.isfile(config_file):
            raise TVMCException(
                f"Config file '{config_file}' doesn't exist or is not a file."
            )

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_file}")

        if "compile_options" not in config:
            raise TVMCException(
                "Config file must contain 'compile_options' section."
            )

        delegate_options = config["compile_options"]

    # Get command-line flags
    enable_tidl_offload = bool(args.enable_tidl_offload)
    compile_for_device = bool(args.compile_for_device)
    c7x_codegen = args.c7x_codegen

    # Merge c7x_codegen into delegate_options (command-line takes precedence)
    delegate_options["advanced_options:c7x_codegen"] = c7x_codegen

    # Validate calibration input is provided when tidl offload is enabled
    if enable_tidl_offload and not args.tidl_calibration_input:
        raise TVMCException(
            "TIDL offload requires calibration input. "
            "Please provide --tidl-calibration-input <path_to_npz_file>."
        )

    # Read TIDL_TOOLS_PATH from environment variable
    tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH")
    if tidl_tools_path is not None:
        delegate_options["tidl_tools_path"] = tidl_tools_path
    elif enable_tidl_offload:
        # Only raise error if TIDL offload is enabled
        raise TVMCException(
            "TIDL offload requires TIDL_TOOLS_PATH environment variable to be set."
        )

    # Set artifacts folder
    if args.output:
        artifacts_folder = args.output
    else:
        artifacts_folder = "./model-artifacts"

    os.makedirs(artifacts_folder, exist_ok=True)
    delegate_options["artifacts_folder"] = artifacts_folder
    logger.info(f"Artifacts will be stored in: {artifacts_folder}")

    # SOC environment variable must be present in the env, otherwise throw error
    if "SOC" not in os.environ:
        raise TVMCException(
            "Environment variable SOC must be set (e.g., am68pa, am68a, am69a, am67a, am62a)."
        )

    platform = os.environ["SOC"]

    # Get input shapes
    if args.input_shapes:
        input_shape_dict = args.input_shapes
    else:
        # Infer shapes from the model
        logger.info("No input shapes provided, inferring from model...")
        input_details = _get_model_input_details(model_path)
        # Convert to format expected by compile_model: dict mapping names to tuples
        input_shape_dict = {inp["name"]: tuple(inp["shape"]) for inp in input_details}

    # Handle calibration inputs
    calibration_input_list = []
    if args.tidl_calibration_input:
        inputs_path = args.tidl_calibration_input

        if not os.path.isfile(inputs_path):
            raise TVMCException(f"Calibration input file '{inputs_path}' doesn't exist.")

        if not inputs_path.endswith('.npz'):
            raise TVMCException(
                f"Only .npz format is supported for calibration inputs, got: {inputs_path}."
            )

        logger.info(f"Loading calibration inputs from: {inputs_path}")

        # Get input details from model
        input_details = _get_model_input_details(model_path)

        # Determine number of calibration frames
        if "advanced_options:calibration_frames" in delegate_options:
            calib_frames = delegate_options["advanced_options:calibration_frames"]
        else:
            calib_frames = 2  # Default

        # Load the npz file to check how many frames are available
        try:
            calib_data = np.load(inputs_path)
        except Exception as e:
            raise TVMCException(f"Failed to load calibration data from '{inputs_path}': {str(e)}")

        # Assume first array represents the frames
        first_array = list(calib_data.values())[0]
        available_frames = first_array.shape[0] if len(first_array.shape) > len(input_details[0]['shape']) else 1

        # Handle calibration frames logic
        if available_frames == 1:
            # Single input - replicate for all frames
            num_frames = calib_frames
        elif available_frames > calib_frames:
            # More inputs than needed - use only first N
            num_frames = calib_frames
            logger.warning(
                f"Number of available frames ({available_frames}) exceeds calibration_frames ({calib_frames}). "
                f"Using only first {num_frames} frames."
            )
        elif available_frames < calib_frames:
            # Fewer inputs than needed - error
            raise TVMCException(
                f"Number of available frames ({available_frames}) is less than "
                f"calibration_frames ({calib_frames})."
            )
        else:
            # Exact match
            num_frames = calib_frames

        # Update calibration_frames in delegate_options to match actual frames used
        delegate_options["advanced_options:calibration_frames"] = num_frames

        # Load calibration data
        calibration_input_list = _load_calibration_data(inputs_path, input_details, num_frames)
        logger.info(f"Loaded {len(calibration_input_list)} calibration frames from {inputs_path}")

    logger.info(
        f"Compiling model for TIDL (platform={platform}, "
        f"device={compile_for_device}, offload={enable_tidl_offload}, "
        f"c7x_codegen={c7x_codegen})"
    )

    # Call tidl.compile_model directly
    try:
        status = tidl.compile_model(
            platform=platform,
            compile_for_device=compile_for_device,
            enable_tidl_offload=enable_tidl_offload,
            delegate_options=delegate_options,
            calibration_input_list=calibration_input_list,
            model_path=model_path,
            input_shape_dict=input_shape_dict,
        )

        if not status:
            raise TVMCException("TIDL compilation failed.")

        logger.info("TIDL compilation completed successfully.")
        return 0
    except Exception as e:
        raise TVMCException(f"Exception during TIDL compilation: {str(e)}")
# End TI


def compile_model(
    tvmc_model: TVMCModel,
    target: str,
    opt_level: int = 3,
    executor: Optional[Executor] = Executor("graph"),
    runtime: Optional[Runtime] = Runtime("cpp"),
    tuning_records: Optional[str] = None,
    package_path: Optional[str] = None,
    cross: Optional[Union[str, Callable]] = None,
    cross_options: Optional[str] = None,
    output_format: str = "so",
    dump_code: Optional[List[str]] = None,
    dump_offloads: str = "",
    target_host: Optional[str] = None,
    disabled_pass: Optional[str] = None,
    pass_context_configs: Optional[List[str]] = None,
    additional_target_options: Optional[Dict[str, Dict[str, Any]]] = None,
    use_vm: bool = False,
    mod_name: Optional[str] = "default",
    workspace_pools: Optional[WorkspaceMemoryPools] = None,
    print_pass_times: bool = False,
    print_ir_before: Optional[List[str]] = None,
    print_ir_after: Optional[List[str]] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
    desired_layout: Optional[str] = None,
    desired_layout_ops: Optional[List[str]] = None,
    mixed_precision: bool = False,
    mixed_precision_ops: Optional[List[str]] = None,
    mixed_precision_calculation_type: Optional[str] = None,
    mixed_precision_acc_type: Optional[str] = None,
):
    """Compile a model from a supported framework into a TVM module.

    This function takes a union of the arguments of both frontends.load_model
    and compiler.compile_relay. The resulting TVM module can be executed using
    the graph executor.

    Parameters
    ----------
    tvmc_model : TVMCModel
        The model object that should be compiled.
    target : str
        The target for which to compile. Can be a plain string or
        a path.
    opt_level : int
        The option that controls various sorts of optimizations.
    tuning_records : str
        A path to tuning records produced using tvmc.tune. When provided,
        compilation will use more optimized kernels leading to better results.
    package_path : str, optional
        The path to export the compiled model to. If not provided it will
        be saved in a temporary directory.
    cross : str or callable object, optional
        Function that performs the actual compilation
    cross_options : str, optional
        Command line options to be passed to the cross compiler.
    output_format : str
        What format to use when saving the function library. Must be one of "so" or "tar".
        When compiling for a remote device without a cross compiler, "tar" will likely work better.
    dump_code : list[str], optional
        Dump the generated code for the specified source types, on
        the requested target. Choose from: ["asm", "ll", "tir", "relay"].
    dump_offloads : str
        Dump the information about the partition of input model's layers by external codegen.
        Can be '' to not dump at all, '-' to dump to the console
        or '<path_to_file>' to dump to the specified file.
    target_host : str, optional
        The target of the host machine if host-side code
        needs to be generated.
    disabled_pass: str, optional
        Comma-separated list of passes which needs to be disabled
        during compilation.
    pass_context_configs: list[str], optional
        List of strings containing a set of configurations to be passed to the
        PassContext.
    additional_target_options: Optional[Dict[str, Dict[str, Any]]]
        Additional target options in a dictionary to combine with initial Target arguments
    use_vm: bool
        Whether to use the VM to compile the model as opposed to the graph executor
    mod_name: str, optional
        The module name
    workspace_pools: WorkspaceMemoryPools, optional
        Specification of WorkspacePoolInfo objects to be used as workspace memory in the
        compilation.
    print_pass_times: bool
        To enable printing a breakdown of compilation times by pass. Disabled by default.
    print_ir_before: list[str], optional
        To print IR before each named pass of a comma-separated list of passes.
    print_ir_after: list[str], optional
        To print IR after each named pass of a comma-separated list of passes.
    instruments: Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.
    desired_layout: str, optional
        Can be one of "NCHW" or "NHWC". When specified, compatible operations in the graph
        will have their layout set to this format. Tasks will then be tuned using this
        specified layout.
    desired_layout_ops: list[str], optional
        The list of operators to be transformed with desired layout.
    mixed_precision: bool
        To enable mixed precision transformation. Disabled by default.
    mixed_precision_ops: list[str], optional
        The list of operators to be converted to mixed precision.
        Set to ["nn.conv2d", "nn.dense"] by default
    mixed_precision_calculation_type: str
        The calculation dtype to be used while mixed precision. Set to "float16" by default.
    mixed_precision_acc_type: str
        The accumulation data type to be used while mixed precision. Set to "float16" by default.

    Returns
    -------
    compiled_model : TVMCPackage
        The compiled TVMCModel ready to be run.

    """
    mod, params = tvmc_model.mod, tvmc_model.params

    if dump_code is None:
        dump_code = []
    if not isinstance(dump_code, list):
        dump_code = [dump_code]
    dumps = {}

    config = parse_configs(pass_context_configs)
    if "tir" in dump_code:
        config, dumps = add_tir_to_dumps(config, dumps)

    initial_relay = None
    if dump_offloads != "":
        # add suffixes to the span field for calls in Relay
        mod = tag_suffixes(mod)
        # remember initial Relay
        initial_relay = deepcopy(mod)

    tvm_target, extra_targets = target_from_cli(target, additional_target_options)
    tvm_target, target_host = Target.canon_target_and_host(tvm_target, target_host)

    partition_functions = []
    partition_opts = []
    for codegen_from_cli in extra_targets:
        codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
        partition_functions.append(codegen["pass_pipeline"])
        partition_opts.append(codegen_from_cli["opts"])
        if codegen["config_key"] is not None:
            config[codegen["config_key"]] = codegen_from_cli["opts"]

    if print_pass_times:
        timing_inst = PassTimingInstrument()
        instruments = [timing_inst] if instruments is None else [timing_inst] + instruments

    if print_ir_before or print_ir_after:
        print_ir_instr = PassPrintingInstrument(
            print_before_pass_names=print_ir_before, print_after_pass_names=print_ir_after
        )
        instruments = [print_ir_instr] if instruments is None else [print_ir_instr] + instruments

    with tvm.transform.PassContext(
        opt_level=opt_level,
        config=config,
        disabled_pass=disabled_pass,
        instruments=instruments,
    ):
        transform_args = parse_graph_transform_args(locals())
        mod = apply_graph_transforms(mod, transform_args)

        for partition_function, opts in zip(partition_functions, partition_opts):
            mod = partition_function(mod, params, mod_name=mod_name, **opts)

        if initial_relay:
            # dump which operations are offloaded to which backend
            dump_operation_offloads(mod, initial_relay, dump_offloads)

        if tuning_records and os.path.exists(tuning_records):
            logger.debug("tuning records file provided: %s", tuning_records)

            use_autoscheduler = True
            try:
                auto_scheduler.load_records(tuning_records)
            except tvm._ffi.base.TVMError:
                use_autoscheduler = False

            if use_autoscheduler:
                with auto_scheduler.ApplyHistoryBest(tuning_records):
                    config["relay.backend.use_auto_scheduler"] = True
                    logger.debug("building relay graph with autoscheduler")
                    graph_module = build(
                        mod,
                        tvm_target=tvm_target,
                        executor=executor,
                        runtime=runtime,
                        params=params,
                        use_vm=use_vm,
                        mod_name=mod_name,
                        workspace_pools=workspace_pools,
                    )
            else:
                with autotvm.apply_history_best(tuning_records):
                    logger.debug("building relay graph with tuning records")
                    graph_module = build(
                        mod,
                        tvm_target=tvm_target,
                        executor=executor,
                        runtime=runtime,
                        params=params,
                        use_vm=use_vm,
                        mod_name=mod_name,
                        workspace_pools=workspace_pools,
                    )
        else:
            logger.debug("building relay graph (no tuning records provided)")
            graph_module = build(
                mod,
                tvm_target=tvm_target,
                executor=executor,
                runtime=runtime,
                params=params,
                use_vm=use_vm,
                mod_name=mod_name,
                workspace_pools=workspace_pools,
            )

        # Generate output dump files with sources
        for source_type in dump_code:
            if source_type == "relay":
                dumps[source_type] = str(mod)
            elif source_type == "tir":
                dumps[source_type] = "\n".join(dumps[source_type])
            else:
                lib = graph_module.lib if use_vm else graph_module.get_lib()
                # TODO lib.get_source call have inconsistent behavior for unsupported
                #      formats (@leandron).
                dumps[source_type] = lib.get_source(source_type)
                for smod in lib.imported_modules:
                    dumps[smod.type_key] = smod.get_source()

        # Create a new tvmc model package object from the graph definition.
        package_path = tvmc_model.export_package(
            graph_module, package_path, cross, cross_options, output_format
        )

        # Write dumps to file.
        if dumps:
            save_dumps(package_path, dumps)

        # Print compilation times per pass
        if print_pass_times:
            print("Compilation time breakdown by pass:")
            print(timing_inst.render())

        return TVMCPackage(package_path)


def build(
    mod: tvm.IRModule,
    tvm_target: str,
    executor: Executor,
    runtime: Runtime,
    params: Dict[str, tvm.nd.NDArray],
    use_vm: bool,
    mod_name: str,
    workspace_pools: Optional[WorkspaceMemoryPools],
):
    """
    Builds the model with the provided executor.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module corresponding to this model.
    tvm_target : str
        The target for which to compile. Can be a plain string or
        a path.
    executor : Executor
        The graph executor to build the model if use_vm is not True
    runtime : Runtime
        The runtime configuration.
    params : dict
        A parameter dictionary for the model.
    use_vm: bool
        Whether to use the VM to compile the model as opposed to the graph executor
    mod_name: str
        The module name

    """
    if use_vm:
        logger.debug("building with vm compile")
        return relay.vm.compile(mod, target=tvm_target, params=params)
    logger.debug("building with relay build")
    return relay.build(
        mod,
        target=tvm_target,
        executor=executor,
        runtime=runtime,
        params=params,
        mod_name=mod_name,
        workspace_memory_pools=workspace_pools,
    )


def add_tir_to_dumps(config, dumps):
    """
    Creates a debug pass that dumps TIR functions as a list of strings.
    """
    key = "tir"
    phase = 3  # final TIR phase before codegen
    dumps[key] = []

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def _dump_tir_pass(tir_func, _, __):
        dumps[key].append(str(tir_func))
        return tir_func

    tir_lower_passes = config.get("tir.add_lower_pass", [])
    tir_lower_passes.append((phase, _dump_tir_pass))
    config["tir.add_lower_pass"] = tir_lower_passes

    return config, dumps


def save_dumps(module_name: str, dumps: Dict[str, str], dump_root: str = "."):
    """
    Serialize dump files to the disk.

    Parameters
    ----------
    module_name : str
        File name, referring to the module that generated
        the dump contents
    dumps : dict
        The output contents to be saved into the files
    dump_root : str, optional
        Path in which dump files will be created
    """

    for dump_format in dumps:
        dump_name = module_name + "." + dump_format
        with open(Path(dump_root, dump_name), "w") as f:
            f.write(dumps[dump_format])


def dump_operation_offloads(mod: tvm.ir.IRModule, initial_mod: tvm.ir.IRModule, dump_path: str):
    """This helper function forms a line-by-line output of the initial Relay lines,
    indicating which operations are ported to which target,
    and indicating the composite that includes those operations;
    the 'generic' target refers to operations uploaded to the host, e.g
    'target1        <-     target1.qnn_conv2d'
    'target1        <-          %0 = qnn.conv2d(%tfl.quantize, %v_param_1, ...'
    'target1        <-          %1 = nn.bias_add(%0, %v_param_2, axis=3);'
    'target1        <-          %2 = qnn.requantize(%1, meta[relay.Constant]...'
    'target2        <-     target2.reshape'
    'target2        <-          %3 = reshape(%2, newshape=[1, 1001]);'
    'generic        <-     %4 = nn.pad(%3, -128f, pad_width=[[0, 0], [1, 1]...'

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The partitioned IRModule with external global functions.
    initial_mod : tvm.ir.IRModule
        The initial IRModule that gets generated from a relay frontend.
    dump_path: str
        Value of the "dump_offloads" compiler atribute.
        Could be dash ("-") or file path or empty string for
        printing to console, file or doing nothing respectively.
    """
    print_to_console = dump_path == "-"
    save_to_file = all([dump_path != "-", dump_path != ""])

    if print_to_console or save_to_file:
        operations_distribution = analyze_operations_distribution(mod)

        def annotate_f(x):
            ret = ""
            if isinstance(x, relay.Call):
                # if there is no x.span.source_name.name in operations_distribution,
                # this could mean that the span was not copied during the application of passes
                # to the Relay, in which case we can not associate the initial Relay string
                # with the resulting Relay call
                source_name = x.span.source_name.name
                suffix = tvm.relay.transform.suffixes.SUFFIX_STRING
                result = re.search(r"(.*)(" + suffix + r")(.*)", source_name)
                func_id = result.group(1)
                if func_id in operations_distribution:
                    compiler_name, op_name = operations_distribution[func_id]
                    ret = (
                        f", compiler_name: {compiler_name}, op_name: {op_name}, "
                        f"func_id: {func_id}"
                    )
                else:
                    ret = ", compiler_name: unknown, op_name: unknown, func_id: unknown"
            elif isinstance(x, (relay.Tuple, relay.TupleGetItem)):
                ret = ", compiler_name: none, op_name: none, func_id: none"

            return ret

        initial_relay_astext = initial_mod.astext(show_meta_data=False, annotate=annotate_f).split(
            "\n"
        )

        # funcs_list is a list of internal composite/function IDs.
        # funcs_list helps keep the order of lines from the initial Relay.
        funcs_list = []

        # target_statistic is a mapping of the target name to the
        # number of initial Relay calls offloaded on the target
        target_statistic = defaultdict(int)

        # funcs_dict is a mapping of the generated analyze_operations_distribution
        # internal composite/function IDs to a list, where:
        # 1st element is
        #   (1a): "generic"|"unknown"|"none"* or
        #   (1b): specific target name, like "ethos-u" or "cmsis-nn"
        # 2nd element is
        #   (2a): corresponding initial Relay line for the case (1a) or
        #   (2b): the name of the target composite functon in the other case (1b)
        # 3rd element or subsequent ones are presented only for the case (2b)
        # and are the initial Relay's lines included in the corresponding
        # target composite functon
        #
        # *Description of what is meant by "generic"|"unknown"|"none":
        # "generic" means that operation will be run on a host
        # "unknown" means that unique identifier of this Relay line not found in the partitioned
        #           Relay and therefore not present in the operations_distribution dictionary
        # "none" means that this Relay line is not relay.Call
        funcs_dict = {}

        # Here we group together initial Relay lines from the one composite
        counter = itertools.count()
        for s in initial_relay_astext:
            result = re.search(
                r"(compiler_name: )(.*)(, op_name: )(.*)(, func_id: )((.*)(?=;)|(.*))", s
            )
            if result:
                target_name = result.group(2)
                op_name = result.group(4)
                func_id = result.group(6)
                if target_name != "none":
                    target_statistic[target_name] += 1

                # create an identifier for each "unknown" or "none" case to keep the lines order
                if func_id == "unknown" or func_id == "none" or target_name == "generic":
                    func_id = str(next(counter) * -1)

                if func_id not in funcs_dict:
                    funcs_list.append(func_id)
                    funcs_dict[func_id] = [target_name]
                    if target_name not in ["unknown", "generic", "none"]:
                        funcs_dict[func_id].append(op_name)

                s = re.sub(r", compiler_name: (.*)", "", s).lstrip()
                funcs_dict[func_id].append(s)

        # Here we prepare the output for printing.
        # The output in most cases keeps the original order of the Relay lines
        # but some lines are moved to be in the corresponding composite group
        output = []
        total = 0
        output.append("Total number of operators and distribution by targets")
        output.append("Total:")
        for target, statistic in target_statistic.items():
            total += statistic
            output.append(f"{target}: {statistic}")
        output[1] += f" {total}"
        output[len(target_statistic) + 1] += "\n"

        for func_id in funcs_list:
            _list = funcs_dict[func_id]

            if _list[0] != "none":
                output.append(f"{_list[0]:<15}<-{' ':5}{_list[1]}")
            else:
                output.append(f"{' ':>22}{_list[1]}")

            if _list[0] == "unknown":
                output.append(
                    "Warning: The above line means that some pass(es) \
                              in Relay partitioning"
                )
                output.append("do not copy the span when the call is recreated")
                output.append(
                    "and a line from initial Relay could not be associated \
                              with the resulting Relay"
                )
            for el in _list[2:]:
                output.append(f"{_list[0]:<15}<-{' ':10}{el}")

        if print_to_console:
            print("\n" + "\n".join(output))
        if save_to_file:
            file_path = os.path.abspath(dump_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("\n".join(output))
                f.write("\n")
