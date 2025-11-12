# TVMC for TIDL Target

## Overview

TVMC supports compiling models for the TIDL target on TI processors.

## Basic Usage

```bash
python -m tvm.driver.tvmc compile model.onnx \
  --target=tidl \
  --tidl-config config.yaml \
  --tidl-calibration-input calibration.npz \
  --enable-tidl-offload 1 \
  --compile-for-device 1 \
  --c7x-codegen 0 \
  --output ./artifacts/
```

## Command Line Options

### Required Arguments

- `FILE` - Path to the input model file (e.g., model.onnx)
- `--target=tidl` - Specify TIDL as the target backend
- `--tidl-calibration-input <path>` - Path to calibration input .npz file (required when `--enable-tidl-offload 1`)

### Optional Arguments

- `--tidl-config <path>` - Path to YAML config file containing compile_options (optional)
- `--enable-tidl-offload {0,1}` - Enable TIDL offload (default: 1)
- `--compile-for-device {0,1}` - Compile for device (aarch64) instead of host (x86_64) (default: 1)
- `--c7x-codegen {0,1}` - Enable C7x code generation (default: 0)
- `--output <path>` - Output directory for compiled artifacts (default: ./model-artifacts)

## Configuration File Format

The configuration file is an optional YAML file that contains compile options for TIDL.

### Compile Options

The YAML file contains a single `compile_options` section with TIDL-specific options.

```yaml
compile_options:
  "debug_level": 0
  "tensor_bits": 8
  "accuracy_level": 1
  "advanced_options:calibration_frames": 2
  "advanced_options:calibration_iterations": 5
  "advanced_options:mixed_precision_factor": -1
  "advanced_options:quantization_scale_type": 0
  "advanced_options:high_resolution_optimization": 0
  "advanced_options:pre_batchnorm_fold": 1
  "ti_internal_nc_flag": 1601
  "advanced_options:activation_clipping": 1
  "advanced_options:weight_clipping": 1
  "advanced_options:bias_calibration": 1
  "advanced_options:channel_wise_quantization": 0
  "advanced_options:add_data_convert_ops": 3
  "advanced_options:inference_mode": 0
  "advanced_options:num_cores": 1
  "advanced_options:c7x_codegen": 0
```

**Note:**
- Command-line options take precedence over YAML options
- For example, `--c7x-codegen 1` will override `"advanced_options:c7x_codegen": 0` in the YAML

## Configuration File Example

```yaml
compile_options:
  "debug_level": 0
  "tensor_bits": 8
  "accuracy_level": 1
  "advanced_options:calibration_frames": 2
  "advanced_options:calibration_iterations": 5
  "advanced_options:num_cores": 1
  "advanced_options:c7x_codegen": 0
  "advanced_options:activation_clipping": 1
  "advanced_options:weight_clipping": 1
  "advanced_options:bias_calibration": 1
```

## Key Compile Options

### Common Options

- `debug_level` - Debug verbosity level (0-3)
- `tensor_bits` - Quantization bit width (8, 16)
- `accuracy_level` - Accuracy vs performance tradeoff (0-9)

### Advanced Options

- `calibration_frames` - Number of frames for calibration
- `calibration_iterations` - Number of calibration iterations
- `c7x_codegen` - Enable C7x code generation (0=disable, 1=enable)
- `num_cores` - Number of C7x cores to use
- `activation_clipping` - Enable activation clipping
- `weight_clipping` - Enable weight clipping
- `channel_wise_quantization` - Enable channel-wise quantization

### Other Options

- `deny_list` - Comma-separated list of operators to exclude from offload

### Object Detection Options

- `object_detection:meta_layers_names_list` - Path to prototxt file for meta layers
- `object_detection:meta_arch_type` - Object detection architecture type

## Example Usage

### Compile with TIDL offload (default)

```bash
python -m tvm.driver.tvmc compile model.onnx \
  --target=tidl \
  --tidl-config config.yaml \
  --tidl-calibration-input calibration.npz \
  --output ./artifacts/
```

### Compile for aarch64 device with C7x codegen

```bash
python -m tvm.driver.tvmc compile model.onnx \
  --target=tidl \
  --tidl-config config.yaml \
  --tidl-calibration-input calibration.npz \
  --compile-for-device 1 \
  --c7x-codegen 1 \
  --output ./artifacts/
```

### Compile without TIDL offload (host-only)

```bash
python -m tvm.driver.tvmc compile model.onnx \
  --target=tidl \
  --tidl-config config.yaml \
  --enable-tidl-offload 0 \
  --output ./artifacts/
```

### Compile without config file (using default options)

```bash
python -m tvm.driver.tvmc compile model.onnx \
  --target=tidl \
  --tidl-calibration-input calibration.npz \
  --output ./artifacts/
```

## Environment Variables

The following environment variables must be set:

- `SOC` - Target SoC platform (e.g., `am68pa`, `am68a`, `am69a`, `am67a`, `am62a`)
- `TIDL_TOOLS_PATH` - Path to TIDL tools (required when `--enable-tidl-offload 1`)

Example:
```bash
export SOC=am68pa
export TIDL_TOOLS_PATH=/path/to/tidl_tools
```
