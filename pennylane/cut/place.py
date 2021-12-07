# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for applying cuts to the circuit graph"""
from pennylane.cut.mark import MeasureNode, PrepareNode, OperationNode, WireCut, GateCut
import networkx as nx
from typing import Tuple, Dict
from pennylane.operation import Operator


def apply_cuts(g):
    original_nodes = tuple(g.nodes)

    for n in original_nodes:
        if isinstance(n, WireCut):
            _remove_wire_node(n, g)
        if isinstance(n, GateCut):
            _remove_gate_node(n, g)


def to_nodes_and_edges(g: nx.Graph) -> Tuple[Dict[Operator, str], Tuple[Tuple[str, str]]]:
    mapping = {op: f"node_{i}" for i, op in enumerate(g.nodes)}
    edges = tuple((mapping[e1], mapping[e2]) for e1, e2 in g.edges)
    return mapping, edges


def place_cuts(g: nx.Graph):



def place_cuts(g: nx.Graph, wire_capacity: int, gate_capacity: int, **kwargs):
    ...

def _remove_wire_node(n, g):
    predecessors = g.predecessors(n)
    successors = g.successors(n)

    g.remove_node(n)

    for p in predecessors:
        for wire in p.wires:
            if wire in n.wires:
                op = MeasureNode(wires=wire)
                g.add_node(op)
                g.add_edge(p, op)

    for s in successors:
        for wire in s.wires:
            if wire in n.wires:
                op = PrepareNode(wires=wire)
                g.add_node(op)
                g.add_edge(op, s)


def _remove_gate_node(n, g):
    predecessors = list(g.predecessors(n))
    successors = list(g.successors(n))

    g.remove_node(n)

    n_wires = n.wires

    for wire in n_wires:
        p_wire = [p for p in predecessors if wire in p.wires][0]
        s_wire = [s for s in successors if wire in s.wires][0]

        op = OperationNode(wires=wire)
        g.add_node(op)
        g.add_edge(p_wire, op)
        g.add_edge(op, s_wire)
