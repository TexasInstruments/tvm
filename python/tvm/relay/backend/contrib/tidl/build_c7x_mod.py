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
"""Build C7x deployable module for C7x TVM C Runtime"""

import os
import sys
import subprocess
import tvm
import tvm.contrib.tidl.c7x as c7x
from tvm import relay
from tvm import transform
from tvm.relay.expr_functor import ExprMutator
from . import tidl

def enable_c7x_mod(tidl_compiler, mod, mod_pre, params, num_tidl_subgraphs):
    """
    This function builds a c7x deployable module that c7x TVM C runtime can run,
    returns an Arm wrapper deployable module that has c7x deployable module embedded in

    Parameters
    ----------
    tidl_compiler: TIDLCompiler
        TIDLCompiler instance
    mod : tvm.relay.Module
        Partitioned Relay IR graph between TIDL subgraphs and TIDL-unsupported layers
        To be compiled with "c7x" codegen into a c7x deployable module
    mod_pre : tvm.relay.Module
        Prepared Relay IR graph before partitioning
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    num_tidl_subgraphs: int
        Number of TIDL subgraphs

    Returns
    -------
    mod_arm : tvm.relay.Module
        Wrapper graph that runs on Arm: ["main"](ins) -> outs { tidl_c7x_0(ins) }
    """
    status = build_c7x_mod(tidl_compiler, mod, params, num_tidl_subgraphs)
    if status == -1:
        raise Exception("Building C7x tvm deployable module failed.")

    print("Creating Arm wrapper tvm module...")
    mod_arm = relay.transform.InferType()(mod_pre)

    # Outline "main" function body to subgraph "tidl_tvm_0" (representing c7x deployable module)
    # See relay_graph.wrapper.txt in artifacts_folder/tempDir.  E.g.
    """
    def @main(%data: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {
      @tidl_tvm_0(%data) /* ty=Tensor[(1, 1000), float32] */
    }

    def @tidl_tvm_0(%data_c7x: Tensor[(1, 3, 224, 224), float32], global_symbol="tidl_tvm_0", Primitive=1, Compiler="tidl", Inline=1) -> Tensor[(1, 1000), float32] {
      %0 = nn.conv2d(%data_c7x, meta[relay.Constant][0] /* ty=Tensor[(16, 3, 3, 3), float32] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 112, 112), float32] */;
        ... ... ...
      nn.batch_flatten(%326) /* ty=Tensor[(1, 1000), float32] */
    }
    """

    func_arm = mod_arm["main"]
    gv_c7x = relay.GlobalVar("tidl_tvm_0")
    # Using same param name in both main() and tidl_tvm_0() will cause internal TVM error
    # Add "_c7x" suffix as a workaround
    params_c7x = [relay.Var(old.name_hint + "_c7x", old.checked_type) for old in func_arm.params]
    old_new_param_map = dict(zip(func_arm.params, params_c7x))
    body_c7x = ParamRenamer(old_new_param_map).visit(func_arm.body)
    func_c7x = relay.Function(params_c7x, body_c7x, func_arm.ret_type)
    func_c7x = func_c7x.with_attr("global_symbol", "tidl_tvm_0")
    func_c7x = func_c7x.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func_c7x = func_c7x.with_attr("Compiler", tidl_compiler.tidl_target)
    func_c7x = func_c7x.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    mod_arm[gv_c7x] = func_c7x

    # Modify "main" function body to be simply calling "tidl_tvm_0"
    mod_arm["main"] = relay.Function(params=func_arm.params,
                                     body=gv_c7x(*func_arm.params),
                                     ret_type=func_arm.ret_type, type_params=None,
                                     attrs=func_arm.attrs)
    mod_arm = relay.transform.InferType()(mod_arm)
    with open(os.path.join(tidl_compiler.temp_folder, "relay_graph.wrapper.txt"), "w") as fo:
        print(mod_arm.astext(show_meta_data=False), file=fo)

    return mod_arm


def bin_to_c(infile, outfile, array_name):
    """
    Encode infile as char array in outfile, similar to "xxd -i" mode
    Update: cl7x/acpia7x has problem with big char array initialization in C file (~600MB)
    Workaround: Instead of use C array with (big) initialization, directly
                encode the binary file data in assembly, as asm7x has no problem
                with big assembly files

    Parameters
    ----------
    infile: string
        input file name
    outfile: string
        output file name
    array_name: string
        c array name

    Returns
        None
    -------
    """
    num_bytes = 0
    with open(infile, "rb") as fi, open(outfile+"_embed.asm", "wt") as fo:
        fo.write(f"\t.sect \".const\"\n\t.clink")
        fo.write(f"\n\t.global ||{array_name}||\n||{array_name}||:")
        byte = fi.read(1)
        while byte:
            if (num_bytes % 16 == 0):
                fo.write(f"\n\t.byte {int.from_bytes(byte, byteorder='little', signed=True)}")
            else:
                fo.write(f",{int.from_bytes(byte, byteorder='little', signed=True)}")
            num_bytes += 1
            byte = fi.read(1)
    with open(outfile, "wt") as fo:
        fo.write(f"extern const unsigned char {array_name}[];\n")
        fo.write(f"const unsigned int {array_name + '_len'} = {num_bytes};\n")


def gen_model_tvm_funcs(outfile, num_tidl_subgraphs):
    """
    Generate tvm model entry functions (create, process, delete)

    Parameters
    ----------
    outfile: string
        output file name
    num_tidl_subgraphs: int
        number of TIDL subgraphs

    Returns
        None
    -------
    """

    with open(outfile, "wt") as fo:
        fo.write(f'''
#include <tvm/runtime/c_runtime_api.h>
#include "graph.json.c"
#include "params.bin.c"
#include "tidl_api.h"
#include "bundle.h"
#define EXPORT __attribute__((visibility("protected")))

static void *tvm_handle = NULL;
static int32_t tvm_rt_debug_level = 0;
static void *  tvm_rt_trace_ptr = NULL;
static int32_t tvm_rt_trace_size = 0;
static int32_t tvm_rt_trace_node = 0;
int32_t tvm_rt_get_debug_level() {{ return tvm_rt_debug_level; }}
void*   tvm_rt_get_trace_ptr()   {{ return tvm_rt_trace_ptr; }}
int32_t tvm_rt_get_trace_size()  {{ return tvm_rt_trace_size; }}
int32_t tvm_rt_get_trace_node()  {{ return tvm_rt_trace_node; }}
static void tvm_rt_set_rt_info(tvm_tidl_rt_info *rt_info)
{{
  if (rt_info == NULL)  return;
  tvm_rt_debug_level = rt_info->tvm_rt_debug_level;
  tvm_rt_trace_ptr   = (void *) rt_info->tvm_rt_trace_ptr;
  tvm_rt_trace_size  = rt_info->tvm_rt_trace_size;
  tvm_rt_trace_node  = rt_info->tvm_rt_trace_node;
}}


EXPORT int tvm_main_create(void *rt_info)
{{
  tvm_rt_set_rt_info((tvm_tidl_rt_info *)rt_info);
''')

        for i in range(num_tidl_subgraphs):
            fo.write(f'''
  extern int tidl_{i}_init(void*);
  if (tidl_{i}_init(rt_info) != 0)  return -1;''')

        fo.write(f'''
  char* json_data = (char*)(graph_json);
  char* params_data = (char*)(params_bin);
  tvm_handle = tvm_runtime_create(json_data, params_data, params_bin_len);

  if (tvm_handle == NULL)  return -1;
  return 0;
}}

EXPORT int tvm_main_process(int32_t num_inputs, int32_t num_outputs,
                     uint32_t* input_names_offset, uint8_t* input_names,
                     void *tensors[])
{{
  for (int i = 0; i < num_inputs; i++)
  {{
    const char *name = (const char *) (input_names + input_names_offset[i]);
    if (tvm_runtime_set_input_raw(tvm_handle, name, tensors[i]) != 0)  return -1;
  }}

  if (tvm_runtime_run(tvm_handle) != 0)  return -1;

  for (int i = 0; i < num_outputs; i++)
  {{
    if (tvm_runtime_get_output_raw(tvm_handle, i, tensors[num_inputs + i]) != 0)  return -1;
  }}
  return 0;
}}

EXPORT int tvm_main_delete()
{{''')

        for i in range(num_tidl_subgraphs):
            fo.write(f'''
  extern void tidl_{i}_destroy(void);
  tidl_{i}_destroy();''')

        fo.write(f'''
  tvm_runtime_destroy(tvm_handle);
  tvm_handle = NULL;
  return 0;
}}
''')

def gen_c7x_source(tidl_compiler, mod, params, num_tidl_subgraphs):
    """
    This function generates c7x source files for TIDL-unsupported layers and TIDL subgraphs

    Parameters
    ----------
    tidl_compiler: TIDLCompiler
        TIDLCompiler instance
    mod : tvm.relay.Module
        Partitioned Relay IR graph between TIDL subgraphs and TIDL-unsupported layers
        To be compiled with "c7x" codegen into a c7x deployable module
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    num_tidl_subgraphs: int
        Number of TIDL subgraphs
    """
    temp_folder = tidl_compiler.temp_folder
    print("Building C7x tvm deployable module: generating c files...")

    if (tidl_compiler.c7x_codegen == 9):  # debug mode: generating generic c code
        with tidl.build_config(tidl_compiler=tidl_compiler, gen_c7x_mod_enabled=1):
            graph, lib, params_c7x = relay.build(mod, tvm.target.Target("c", host="c"),
                                                 params=params)
    else:
        with tidl.build_config(tidl_compiler=tidl_compiler, gen_c7x_mod_enabled=1):
            with c7x.c7x_target_config():
                graph, lib, params_c7x = relay.build(mod, tvm.target.Target("c7x", host="c7x"),
                                                     params=params)
    tidl.remove_tidl_params(params_c7x)
    modules = lib._collect_dso_modules()
    for i in range(len(modules)):
        if modules[i].type_key == 'c':
            modules[i].save(os.path.join(temp_folder, f"model_{i}.c"), 'c')
    graph_fname = os.path.join(temp_folder, "graph.json")
    with open(graph_fname, "w") as fo:
        fo.write(graph)
    params_fname = os.path.join(temp_folder, "params.bin")
    with open(params_fname, "wb") as fo:
        fo.write(relay.save_param_dict(params_c7x))

    from tvm.micro import func_registry
    func_registry.graph_json_to_c_func_registry(graph_fname,
                                                os.path.join(temp_folder, "func_registry.c"))

    bin_to_c(graph_fname, graph_fname + ".c", "graph_json")
    bin_to_c(params_fname, params_fname + ".c", "params_bin")
    for i in range(num_tidl_subgraphs):
        subgraph_net_fname = os.path.join(temp_folder, f"subgraph{i}_net.bin")
        c_net_fname = os.path.join(temp_folder, f"subgraph{i}_net.c")
        subgraph_params_fname = os.path.join(temp_folder, f"subgraph{i}_params_1.bin")
        c_params_fname = os.path.join(temp_folder, f"subgraph{i}_params.c")
        bin_to_c(subgraph_net_fname, c_net_fname, f"subgraph{i}_net_bin")
        bin_to_c(subgraph_params_fname, c_params_fname, f"subgraph{i}_params_1_bin")

    gen_model_tvm_funcs(os.path.join(temp_folder, "tvm_main.c"), num_tidl_subgraphs)

def build_c7x_mod(tidl_compiler, mod, params, num_tidl_subgraphs):
    """
    This function builds a c7x deployable module that c7x TVM C runtime can run,
    returns an Arm wrapper deployable module that has c7x deployable module embedded in

    Parameters
    ----------
    tidl_compiler: TIDLCompiler
        TIDLCompiler instance
    mod : tvm.relay.Module
        Partitioned Relay IR graph between TIDL subgraphs and TIDL-unsupported layers
        To be compiled with "c7x" codegen into a c7x deployable module
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    num_tidl_subgraphs: int
        Number of TIDL subgraphs

    Returns
    -------
    status: int
        1: success, -1: failure
    """
    if os.environ.get("TIDL_REBUILD_ONLY") == None:
        gen_c7x_source(tidl_compiler, mod, params, num_tidl_subgraphs)

    print("Building C7x tvm deployable module: building... (log in c7x_deploy_mod.log)")
    # if script from python package:      tvm/relay/backend/contrib/tidl_build_c7x_mod.py
    # if script from dev repo: tvm/python/tvm/relay/backend/contrib/tidl_build_c7x_mod.py
    tvm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if not os.path.exists(os.path.join(tvm_root, "src/runtime/contrib/tidl/c7x")):
        tvm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
    tvm_c7x_root = os.path.join(tvm_root, "src/runtime/contrib/tidl/c7x")
    abs_temp_folder = os.path.abspath(tidl_compiler.temp_folder)
    log_file = os.path.join(abs_temp_folder, "c7x_deploy_tvm.log")
    command  = f'make TVM_ROOT={tvm_root} TVM_C7X_ROOT={tvm_c7x_root} QUIET= ' + \
               f' -C {abs_temp_folder} -f {tvm_c7x_root}/Makefile.c7x_mod -j$(nproc)'
    print(command)
    with open(log_file, "w") as fo:
        p_status = subprocess.run([command], stdout=fo, stderr=fo, shell=True).returncode
    if p_status != 0:
        with open(log_file, "r") as fi:
            print(fi.read())

    return 1 if p_status == 0 else -1

class ParamRenamer(ExprMutator):
    """
    Renames old param var to new param var.  E.g. "data" to "data_c7x"
    """
    def __init__(self, old_new_param_map):
        ExprMutator.__init__(self)
        self.old_new_param_map = old_new_param_map

    def visit_var(self, var):
        if var in self.old_new_param_map:
            return self.old_new_param_map[var]
        return super().visit_var(var)


