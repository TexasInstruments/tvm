
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
"""TIDL graphviz visualizer for Relay graph """

# This is a simple visualization routine that renders a Relay IR function
# as a graph using the graphviz package. It shows functions tagged as 
# GlobalVars (i.e. partitioned subgraphs) as inlined subgraphs delimited with
# boxes.

# see https://discuss.tvm.ai/t/rfc-visualizing-relay-program-as-graph/4825/14
import tvm
from tvm import relay
import tvm.ir
from tvm.relay.expr import Tuple, GlobalVar
from tvm.relay.function import Function

from graphviz import Digraph

def visualize(root, mod):
    ''' Produce a graphviz graph for a Relay function '''
    # map from nodes to Digraph identifier
    node_dict = {}

    def _add_edge(dot, n1, n2):
        ''' Add an edge from n1 to n2 '''
        # Don't add edges for ignored nodes 
        if n1 not in node_dict or n2 not in node_dict:
            return
        # When adding a edge from a subfunction call, override the source to be the 
        # called function instead of the call node, which has the effect of inlining
        # the callee's graph
        if isinstance(n1, relay.expr.Call) and isinstance(n1.op, relay.expr.GlobalVar):
            func = mod[n1.op.name_hint]
            if func in node_dict:
                n1 = func
        dot.edge(node_dict[n1], node_dict[n2])

    def visit(node, dot):
        ''' Recursively walk an IR expression to build a graphviz graph '''
        if node in node_dict:
            return
        # Ignore uninteresting nodes
        if isinstance(node, tvm.ir.Op) or isinstance(node, relay.expr.Constant):
            return
        node_idx = str(len(node_dict))
        node_dict[node] = node_idx
        if isinstance(node, relay.expr.Var):
            dot.node(node_idx, f'Var:{node.name_hint}')
        elif isinstance(node, relay.Function):
            visit(node.body, dot)
            dot.node(node_idx, f'Function {node_dict[node.body]}')
            _add_edge(dot, node.body, node)
        elif isinstance(node, relay.expr.TupleGetItem):
            visit(node.tuple_value, dot)
            dot.node(node_idx, f'TupleGetItem(idx={node.index})')
            _add_edge(dot, node.tuple_value, node)
        elif isinstance(node, relay.expr.Tuple):
            dot.node(node_idx, f'Tuple')
            for expr in node.fields:
                visit(expr, dot)
                _add_edge(dot, expr, node)
        elif isinstance(node, relay.expr.GlobalVar):
            # Create a subgraph for the called function
            fname = node.name_hint
            body = mod[fname]
            with dot.subgraph(name=f'cluster_{fname}') as sg:
                sg.attr(label=fname, color='blue')
                visit(body, sg)
        elif isinstance(node, relay.expr.Call):
            arg_dests = [node] * len(node.args)
            if isinstance(node.op, relay.Function):
                dot.node(node_idx, f'Call(Function({node_dict[node.op.body]}))')
            elif isinstance(node.op, tvm.ir.op.Op):
                dot.node(node_idx, f'Call(op={str(node.op)})')
            elif isinstance(node.op, relay.expr.GlobalVar):
                # Visit the GlobalVar to create a subgraph for the called function
                visit(node.op, dot)
                # Tie edges from incoming arguments to formal parameters of callee. 
                # This 'inlines' the callee's subgraph
                body = mod[node.op.name_hint]
                arg_dests = body.params
            else:
                raise RuntimeError(f'Unknown call op. node_idx: {node_idx}, op: {type(node.op)}')
            # Visit incoming arguments and add edges from them to either this node or 
            # the formal parameters of callee
            for idx,arg in enumerate(node.args):
                visit(arg, dot)
                _add_edge(dot, arg, arg_dests[idx])
        else:
            raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

    # Initialize a directed graph
    dot = Digraph(format='svg')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box')
    dot.attr('node', fontsize='11')

    # Walk the Relay function to build the graph
    visit(root, dot)
    return dot

def visualize_relay_graph(module, filename):
    ''' Top level entry function to build and save a graphviz file from a
        Relay graph'''
    dot = visualize(module['main'], module)
    dot.render(filename)
