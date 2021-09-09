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
from tvm import relay
from tvm import transform
from tvm.relay.expr_functor import ExprMutator
from . import tidl

def bin_to_c(infile, outfile, array_name):
    """
    Encode infile as char array in outfile, similar to "xxd -i" mode

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
    with open(infile, "rb") as fi, open(outfile, "wt") as fo:
        fo.write(f"const unsigned char {array_name}[] = {{")
        byte = fi.read(1)
        num_bytes = 0
        while byte:
            if (num_bytes % 12 == 0):
                fo.write(f"\n  0x{byte.hex()},")
            else:
                fo.write(f" 0x{byte.hex()},")
            num_bytes += 1;
            byte = fi.read(1)
        fo.write("\n};\n")
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

EXPORT int tvm_main_create()
{{''')

        for i in range(num_tidl_subgraphs):
            fo.write(f'''
  extern void tidl_{i}_init(void);
  tidl_{i}_init();''')

        fo.write(f'''
  char* json_data = (char*)(graph_json);
  char* params_data = (char*)(params_bin);
  tvm_handle = tvm_runtime_create(json_data, params_data, params_bin_len);

  return 0;
}}

EXPORT int tvm_main_process(int32_t num_inputs, int32_t num_outputs,
                     uint32_t* input_names_offset, uint8_t* input_names,
                     void *tensors[])
{{
  for (int i = 0; i < num_inputs; i++)
  {{
    const char *name = (const char *) (input_names + input_names_offset[i]);
    tvm_runtime_set_input_raw(tvm_handle, name, tensors[i]);
  }}

  tvm_runtime_run(tvm_handle);

  for (int i = 0; i < num_outputs; i++)
  {{
    tvm_runtime_get_output_raw(tvm_handle, i, tensors[num_inputs + i]);
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
    temp_folder = tidl_compiler.temp_folder
    print("Building C7x tvm deployable module: generating c files...")
    #c_target = "c7x"
    c_target = "c"
    with tidl.build_config(tidl_compiler=tidl_compiler, gen_c7x_mod=1):
      with transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
        graph, lib, params_c7x = relay.build_module.build(mod, target=c_target,
                                                          target_host=c_target, params=params)
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

    print("Building C7x tvm deployable module: building... (log in c7x_deploy_mod.log)")
    # if script from python package:      tvm/relay/backend/contrib/tidl_build_c7x_mod.py 
    # if script from dev repo: tvm/python/tvm/relay/backend/contrib/tidl_build_c7x_mod.py 
    tvm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if not os.path.exists(os.path.join(tvm_root, "src/runtime/contrib/tidl/c7x")):
        tvm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
    tvm_c7x_root = os.path.join(tvm_root, "src/runtime/contrib/tidl/c7x")
    abs_temp_folder = os.path.abspath(temp_folder)
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


def enable_c7x_mod(tidl_compiler, mod, mod_orig, params, num_tidl_subgraphs):
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
    mod_orig : tvm.relay.Module
        Original Relay IR graph
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
        print("Building C7x tvm deployable module failed.  Reverting to Arm execution.")
        return mod

    print("Creating Arm wrapper tvm module...")
    mod_arm = relay.transform.RemoveUnusedFunctions()(mod_orig)
    mod_arm["main"] = relay.build_module.bind_params_by_name(mod_arm["main"], params)
    mod_arm["main"] = tidl.RemoveTrainingOperators().visit(mod_arm["main"])
    mod_arm = relay.transform.FoldConstant()(mod_arm)
    mod_arm = relay.transform.EliminateCommonSubexpr()(mod_arm)
    mod_arm = relay.transform.InferType()(mod_arm)

    # Outline "main" function body to subgraph "tidl_tvm_0" (representing c7x deployable module)
    func_arm = mod_arm["main"]
    gv_c7x = relay.GlobalVar("tidl_tvm_0")
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

