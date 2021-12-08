# Copyright 2021 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""""""
import pennylane as qml
from pennylane import qcut
import networkx as nx


def compare_ops(ops1, ops2):
    """Compares two lists of operations"""
    assert len(ops1) == len(ops2)
    for o1, o2 in zip(ops1, ops2):
        assert o1.name == o2.name
        assert o1.wires == o2.wires
        assert o1.parameters == o2.parameters


class TestWireCutNode:

    def test_simple(self):
        op = qcut.WireCut(wires=0)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(op)
            qml.RY(0.5, wires=0)

        g = tape.graph.graph

        qcut.remove_wire_cut_node(op, g)

        ops = list(nx.topological_sort(g))
        expected_ops = [
            qml.RX(0.4, wires=0),
            qcut.MeasureNode(wires=0),
            qcut.PrepareNode(wires=0),
            qml.RY(0.5, wires=0),
        ]

        compare_ops(ops, expected_ops)
        expected_edges = [
            (qml.RX(0.4, wires=0), qcut.MeasureNode(wires=0), {}),
            (qcut.MeasureNode(wires=0), qcut.PrepareNode(wires=0), {"pair": (qcut.MeasureNode(wires=0), qcut.PrepareNode(wires=0))}),
            (qcut.PrepareNode(wires=0), qml.RY(0.5, wires=0), {}),
        ]
        edges = list(g.edges(data=True))

        compare_ops(edges[0][:2], expected_edges[0][:2])
        compare_ops(edges[1][:2], expected_edges[1][:2])
        compare_ops(edges[2][:2], expected_edges[2][:2])

        assert edges[0][-1] == {}
        assert edges[2][-1] == {}

        data = edges[1][-1]
        assert list(data.keys()) == ["pair"]
        compare_ops(list(data.values())[0], (qcut.MeasureNode(wires=0), qcut.PrepareNode(wires=0)))



