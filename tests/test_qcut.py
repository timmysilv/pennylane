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
from pennylane.wires import Wires


def compare_operations(op1, op2):
    """Compares two operations"""
    assert op1.name == op2.name
    assert op1.wires == op2.wires
    assert op1.parameters == op2.parameters


def compare_measurements(op1, op2):
    """Compares two measurements"""
    assert op1.name == op2.name
    assert op1.wires == op2.wires
    assert op1.return_type == op2.return_type

    obs1 = getattr(op1, "obs", None)
    obs2 = getattr(op2, "obs", None)

    if obs1 is not None:
        assert obs2 is not None
        assert obs1.name == obs2.name


def compare_ops_list(ops1, ops2):
    """Compares two lists of operators"""
    assert len(ops1) == len(ops2)
    for op1, op2 in zip(ops1, ops2):
        if isinstance(op1, qml.measure.MeasurementProcess):
            compare_measurements(op1, op2)
        else:
            compare_operations(op1, op2)


def test_tape_to_graph():
    """Test for the tape_to_graph function"""
    with qml.tape.QuantumTape() as tape:
        qml.BasisState([0, 1], wires=[0, 1])
        qml.RX(0.4, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=1)
        qcut.WireCut(wires=0)
        qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
        qml.probs(wires=[0, 1])

    g = qcut.tape_to_graph(tape)

    assert isinstance(g, nx.MultiDiGraph)

    expected_nodes = [
        (qml.BasisState([0, 1], wires=[0, 1]), {'order': 0}), (qml.RX(0.4, wires=[0]), {'order': 1}),
        (qml.CNOT(wires=[0, 1]), {'order': 2}), (qml.Hadamard(wires=[1]), {'order': 3}),
        (qcut.WireCut(wires=[0]), {'order': 4}), (qml.expval(qml.PauliZ(wires=[0])), {'order': 4}),
        (qml.expval(qml.PauliX(wires=[1])), {'order': 5}), (qml.probs(wires=[0, 1]), {'order': 6})
    ]

    for n1, n2 in zip(expected_nodes, g.nodes(data=True)):
        compare_ops_list([n1[0]], [n2[0]])
        assert n1[1] == n2[1]

    expected_edges = [
        (qml.BasisState([0, 1], wires=[0, 1]), qml.RX(0.4, wires=0), {"wire": 0}),
        (qml.BasisState([0, 1], wires=[0, 1]), qml.CNOT(wires=[0, 1]), {"wire": 1}),
        (qml.RX(0.4, wires=0), qml.CNOT(wires=[0, 1]), {"wire": 0}),
        (qml.CNOT(wires=[0, 1]), qml.Hadamard(wires=1), {"wire": 1}),
        (qml.CNOT(wires=[0, 1]), qcut.WireCut(wires=0), {"wire": 0}),
        (qml.Hadamard(wires=1), qml.expval(qml.PauliX(1)), {"wire": 1}),
        (qml.Hadamard(wires=1), qml.probs(wires=[0, 1]), {"wire": 1}),
        (qcut.WireCut(wires=0), qml.expval(qml.PauliZ(0)), {"wire": 0}),
        (qcut.WireCut(wires=0), qml.probs(wires=[0, 1]), {"wire": 0}),
    ]

    for e1, e2 in zip(expected_edges, g.edges(data=True)):
        compare_ops_list(e1[:2], e2[:2])
        assert e1[-1] == e2[-1]


class TestRemoveWireCutNode:
    """Tests for the remove_wire_cut_node function"""

    def test_standard(self):
        """Test on a typical circuit configuration"""
        op = qcut.WireCut(wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.RX(0.4, wires=1)
            qml.apply(op)
            qml.RY(0.5, wires=0)
            qml.CRX(0.1, wires=[1, 0])

        g = qcut.tape_to_graph(tape)
        qcut.remove_wire_cut_node(op, g)

        ops = list(nx.topological_sort(g))
        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.RX(0.4, wires=1),
            qcut.MeasureNode(wires=0),
            qcut.MeasureNode(wires=1),
            qcut.PrepareNode(wires=0),
            qcut.PrepareNode(wires=1),
            qml.RY(0.5, wires=0),
            qml.CRX(0.1, wires=[1, 0]),
        ]

        compare_ops_list(ops, expected_ops)
        order = [g.nodes(data="order")[o] for o in ops]
        assert order == [0, 1, 2, 2, 2.5, 2.5, 3, 4]

        expected_edges = [
            (qml.CNOT(wires=[0, 1]), qml.RX(0.4, wires=1), {"wire": 1}),
            (qml.CNOT(wires=[0, 1]), qcut.MeasureNode(wires=0), {"wire": 0}),
            (qml.RX(0.4, wires=1), qcut.MeasureNode(wires=1), {"wire": 1}),
            (qml.RY(0.5, wires=0), qml.CRX(0.1, wires=[1, 0]), {"wire": 0}),
            (qcut.MeasureNode(wires=0), qcut.PrepareNode(wires=0), {"wire": 0}),
            (qcut.PrepareNode(wires=[0]), qml.RY(0.5, wires=[0]), {'wire': 0}),
            (qcut.MeasureNode(wires=[1]), qcut.PrepareNode(wires=[1]), {'wire': 1}),
            (qcut.PrepareNode(wires=[1]), qml.CRX(0.1, wires=[1, 0]), {'wire': 1}),
        ]

        edges = g.edges(data=True)
        assert len(edges) == len(expected_edges)
        for e1, e2 in zip(expected_edges, edges):
            compare_ops_list(e1[:2], e2[:2])
            assert e1[-1] == e2[-1]

    def test_no_successor(self):
        ...

    def test_no_predecessor(self):
        ...


class TestFragmentGraph:
    """Tests for the fragment_graph function"""

    def test_standard(self):
        """Test on a typical circuit cutting configuration"""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qcut.WireCut(wires=0)
            qml.S(wires=0)

        g = qcut.tape_to_graph(tape)
        qcut.remove_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        compare_ops_list(subgraphs[0].nodes, [qml.Hadamard(wires=0), qcut.MeasureNode(wires=0)])
        compare_ops_list(subgraphs[1].nodes, [qml.S(wires=0), qcut.PrepareNode(wires=0)])

        e0 = subgraphs[0].edges(data=True)
        e1 = subgraphs[1].edges(data=True)

        assert len(e0) == 1
        assert len(e1) == 1

        e0 = list(e0)[0]
        compare_ops_list(e0[:2], [qml.Hadamard(wires=0), qcut.MeasureNode(wires=0)])
        assert e0[-1] == {"wire": 0}

        e1 = list(e1)[0]
        compare_ops_list(e1[:2], [qcut.PrepareNode(wires=0), qml.S(wires=0)])
        assert e1[-1] == {"wire": 0}

        assert list(communication_graph.nodes) == [0, 1]
        c_edges = communication_graph.edges(data=True)

        assert len(c_edges) == 1
        c_edge = list(c_edges)[0]

        assert c_edge[:2] == (0, 1)
        compare_ops_list(c_edge[-1]["pair"],  (qcut.MeasureNode(wires=[0]), qcut.PrepareNode(wires=[0])))


class TestGraphToTape:
    """Tests for the graph_to_tape"""

    def test_standard(self):
        """Test on a typical circuit cutting configuration"""
        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            qml.S(wires=2)

            qcut.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qcut.WireCut(wires=1)

            qml.CNOT(wires=[0, 1])
            qml.PauliY(2)

            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

        g = qcut.tape_to_graph(tape)
        qcut.remove_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tape_0 = qcut.graph_to_tape(subgraphs[0])
        tape_1 = qcut.graph_to_tape(subgraphs[1])

        with qml.tape.QuantumTape() as expected_tape_0:
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)

            qcut.MeasureNode(wires=1)
            qcut.PrepareNode(wires=2)

            qml.CNOT(wires=[0, 2])

            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as expected_tape_1:
            qml.S(wires=2)

            qcut.PrepareNode(wires=1)
            qml.CNOT(wires=[1, 2])
            qcut.MeasureNode(wires=1)

            qml.PauliY(2)

            qml.expval(qml.PauliZ(2))

        compare_ops_list(tape_0.operations + tape_0.measurements, expected_tape_0.operations + tape_0.measurements)
        compare_ops_list(tape_1.operations + tape_1.measurements, expected_tape_1.operations + tape_1.measurements)


def test_find_new_wire():
    """Test for the _find_new_wire function"""
    w = Wires([0, -1, "d", 33, 2.3, "Alice", 1])

    assert qcut._find_new_wire(w) == 2


class TestExpandFragmentTapes:
    """Tests for the expand_fragment_tapes function"""
    def test_standard(self):
        """Test on a typical circuit cutting configuration"""
        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            qml.S(wires=2)

            qcut.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qcut.WireCut(wires=1)

            qml.CNOT(wires=[0, 1])
            qml.PauliY(2)

            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

        g = qcut.tape_to_graph(tape)
        qcut.remove_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tape_0 = qcut.graph_to_tape(subgraphs[0])
        tape_1 = qcut.graph_to_tape(subgraphs[1])

        tapes_0, prep_nodes_0, meas_nodes_0 = qcut.expand_fragment_tapes(tape_0)
        tapes_1, prep_nodes_1, meas_nodes_1 = qcut.expand_fragment_tapes(tape_1)
        # TODO