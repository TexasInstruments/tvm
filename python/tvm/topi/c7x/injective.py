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
# pylint: disable=invalid-name, unused-variable
"""Schedule for injective operators"""
import tvm
from tvm import te
from .. import tag


def schedule_injective(outs):
    """C7X CPU schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.Target.current(allow_none=False)
    print(f"schedule_injective for c7x, target={target}")

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s : te.schedule.Schedule = te.create_schedule([E.op for E in outs])
    E : te.tensor.Tensor = outs[0]

    stage : te.schedule.Stage = s[E]
    op : te.tensor.ComputeOp = stage.op
    a : te.tensor.Tensor = op.input_tensors[0]
    b : te.tensor.Tensor = op.input_tensors[1]
    aa : te.tensor.Tensor = s.cache_read(a, "global", op);
    bb : te.tensor.Tensor = s.cache_read(b, "global", op);
    #cc : te.tensor.Tensor = s.cache_write(E, "global");

    axes = list(op.axis)  # type ir.container.Array[tir.expr.IterVar]
    # double buffer
    if 1:
        s[aa].compute_at(stage, axes[0])
        s[aa].double_buffer()
    #split, vectorize by 8
    if 0:
        (xo, xi) = stage.split(axes[-1], 8)
        stage.vectorize(xi)
    #tile
    if 0:
       (xo, yo, xi, yi) = stage.tile(axes[0], axes[1], 16, 32)
       stage.double_buffer()
    #fuse
    if 0:
        xy : tir.expr.IterVar = stage.fuse(axes[0], axes[1])
    #te.schedule.AutoInlineInjective(s)
    show(s)

    return s

from tvm.contrib import tedd
import graphviz as gv

def show(s):
#    code = tvm.lower(s, args=[], simple_mode=True)
#    print(code)
    print(f"schedule:\n{s}")

    dotstr = tedd.viz_dataflow_graph(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='df').render()

    dotstr = tedd.viz_schedule_tree(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='tree').render()

    dotstr = tedd.viz_itervar_relationship_graph(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='iter').render()

#    print(dot)
#    tedd.viz_dataflow_graph(s, show_svg = True)
#    tedd.viz_dataflow_graph(s, dot_file_path="./dfg.dot")
