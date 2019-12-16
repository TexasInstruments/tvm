# TVM code generation for TI's AM5729/49 SoC

In order to leverage improved performance of hardware accelerators (2xC66x and 4xEVE) on AM5729, TIDL stack need to be interfaced with TVM runtime. Code generation side TVM modifications will create dynamic calls to libtidl_api.so, available in TI's Processor Linux SDK file system. All other necessary software components are also pre-installed in Processor Linux SDK for AM5729/49.

``NeoTvmCodeGen.py`` script compiles Tensorflow models to TVM runtime (NEO-AI-DLR) compatible modules.
Please note that script requires TF 1.14/1.15 to be installed, as well as Python OpenCV and ANTLR4 packages:

```
    pip3 install --user tensorflow==1.14
    pip3 install opencv-python --user
    pip3 install antlr4-python3-runtime --user
```

If xNN model is not compatible with TIDL framework for edge devices, it falls back to ARM only execution, and compiles a set of artifacts for ARMv7 platform (in ./output4/<modelname> folder)

```
    deploy_graph.json
    deploy_lib.so
    deploy_param.params
```

If we use a xNN model that can be fully compiled for execution using TIDL library (AM5729/49 SoC with EVE and DSP accelerators), it creates Relay graph with ``TidlInference()`` node, and compiles into set of artifacts that can be used by NEO-AI-DLR runtime. In this case there are two additional tidl specific files (used by TIDL runtime stack and EVE/DSP HWA).

```
    tidl_subgraph_net.bin
    tidl_subgraph_params.bin
```

Codegen examples:

- ``python3 NeoTvmCodeGen.py mobileNet1 (or mobilenet2)``
    First mobilenet1 model need to be downloaded and decompressed (please check README in mobilenet1 subfolder). TF mobilenet1 model is compatible with TIDL, and it will be compiled into set of files in ``./output4/mobilenet1``

- ``python3 NeoTvmCodeGen.py mobileNet1 -a``
    This is an example of forced compilation for ARM-only execution. This is useful for comparing accuracy, execution time.
- ``python3 NeoTvmCodeGen.py mobileNet2 -b 16``
    This is an example forcing default batch size (of 4) to be replaced with the batch size of 16.
    Since AM5729 SoC has 4 independent accelerators, increase in throughput is achieved via increased batch size (at the cost of latency).


```
Command line usage

    python3 NeoTvmCodeGen.py --help


    usage: NeoTvmCodeGen.py [-h] [--forced_arm_offload] [--forced_tidl_offload]
                            [--batch_size BATCH_SIZE] [--input_node INPUT_NODE]
                            [--output_node OUTPUT_NODE]
                            [--calibration_image CALIBRATION_IMAGE]
                            [--input_shape INPUT_SHAPE [INPUT_SHAPE ...]]
                            modelName

    positional arguments:
      modelName             Model name

    optional arguments:
      -h, --help            show this help message and exit
      --forced_arm_offload, -a
                            Force ARM-only execution
      --forced_tidl_offload, -t
                            Force TIDL offload
      --batch_size BATCH_SIZE, -b BATCH_SIZE
                           Batch size
      --input_node INPUT_NODE, -i INPUT_NODE
                            Input node name
      --output_node OUTPUT_NODE, -o OUTPUT_NODE
                            Output node name
      --calibration_image CALIBRATION_IMAGE, -c CALIBRATION_IMAGE
                            Calibration image
      --input_shape INPUT_SHAPE [INPUT_SHAPE ...], -s INPUT_SHAPE [INPUT_SHAPE ...]
                            Input shape: H W C, e.g. -s 224 224 3


```

