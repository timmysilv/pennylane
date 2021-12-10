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
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph, weakly_connected_components

from pennylane.operation import AnyWires, Operation, Operator, Tensor
from pennylane.tape import QuantumTape
from pennylane.transforms import batch_transform
from pennylane.measure import MeasurementProcess
from pennylane.wires import Wires
from pennylane import apply, PauliX, PauliY, PauliZ, Identity, Hadamard, S, expval


class WireCut(Operation):
    num_wires = AnyWires
    grad_method = None

    def expand(self) -> QuantumTape:
        with QuantumTape() as tape:
            ...
        return tape


class MeasureNode(Operation):
    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    num_wires = 1
    grad_method = None


@batch_transform
def cut_circuit(
    tape: QuantumTape, method: Optional[Union[str, Callable]] = None, **kwargs
) -> Tuple[Tuple[QuantumTape], Callable]:
    """Main transform"""
    g = tape_to_graph(tape)
    remove_wire_cut_nodes(g)

    if method is not None:
        find_and_place_cuts(g, method=method, **kwargs)

    fragments, communication_graph = fragment_graph(g)
    fragment_tapes = [graph_to_tape(f) for f in fragments]

    configurations = [expand_fragment_tapes(t) for t in fragment_tapes]
    shapes = [len(c) for c in configurations]

    tapes = tuple(tape for c in configurations for tape in c)

    return tapes, partial(contract, shapes=shapes, communication_graph=communication_graph)


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """Converts a quantum tape to a directed multigraph."""
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}

    for order, op in enumerate(tape.operations):
        graph.add_node(op, order=order)
        for wire in op.wires:
            if wire_latest_node[wire] is not None:
                parent_op = wire_latest_node[wire]
                graph.add_edge(parent_op, op, wire=wire)
            wire_latest_node[wire] = op

    for m in tape.measurements:
        obs = getattr(m, "obs", None)
        if obs is not None and isinstance(obs, Tensor):
            for o in obs.obs:
                m_ = MeasurementProcess(m.return_type, obs=o)

                graph.add_node(m_, order=order)
                order += 1
                for wire in o.wires:
                    parent_op = wire_latest_node[wire]
                    graph.add_edge(parent_op, m_, wire=wire)
        else:
            graph.add_node(m, order=order)
            order += 1

            for wire in m.wires:
                parent_op = wire_latest_node[wire]
                graph.add_edge(parent_op, m, wire=wire)

    return graph


def remove_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """Removes a WireCut node from the graph"""
    predecessors = graph.pred[node]
    successors = graph.succ[node]

    predecessor_on_wire = {}
    for op, data in predecessors.items():
        for d in data.values():
            wire = d["wire"]
            predecessor_on_wire[wire] = op

    successor_on_wire = {}
    for op, data in successors.items():
        for d in data.values():
            wire = d["wire"]
            successor_on_wire[wire] = op

    order = graph.nodes[node]["order"]
    graph.remove_node(node)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)
        graph.add_node(meas, order=order)
        graph.add_node(prep, order=order + 0.5)

        graph.add_edge(meas, prep, wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas, wire=wire)
        if successor is not None:
            graph.add_edge(prep, successor, wire=wire)


def remove_wire_cut_nodes(graph: MultiDiGraph):
    """Remove all WireCuts from the graph"""
    for op in list(graph.nodes):
        if isinstance(op, WireCut):
            remove_wire_cut_node(op, graph)


def find_and_place_cuts(graph: MultiDiGraph, method: Union[str, Callable], **kwargs):
    """Automatically find additional cuts and place them in the graph. A ``method`` can be
    explicitly passed as a callable, or built-in ones can be used by specifying the corresponding
    string."""
    ...  # calls ``method`` (see ``example_method``) and ``place_cuts``


def example_method(
    graph: MultiDiGraph,
    max_wires: Optional[int],
    max_gates: Optional[int],
    num_partitions: Optional[int],
    **kwargs
) -> Tuple[Tuple[Tuple[Operator, Operator, Any]], Dict[str, Any]]:
    """Example method passed to ``find_cuts``. Returns a tuple of wire cuts of the form
    ``Tuple[Tuple[Operator, Operator, Any]]`` specifying the wire to cut between two operators. An
    additional results dictionary is also returned that can contain optional optimization results.
    """
    ...


def place_cuts(graph: MultiDiGraph, wires: Tuple[Tuple[Operator, Operator, Any]]):
    """Places wire cuts in ``graph`` according to ``wires`` which contains pairs of operators along
    with the wire passing between them to be cut."""
    ...


def fragment_graph(graph: MultiDiGraph) -> Tuple[Tuple[MultiDiGraph], MultiDiGraph]:
    """Fragments a cut graph into a collection of subgraphs as well as returning the
    communication/quotient graph."""
    edges = list(graph.edges)
    cut_edges = []

    for node1, node2, _ in edges:
        if isinstance(node1, MeasureNode):
            assert isinstance(node2, PrepareNode)
            cut_edges.append((node1, node2))
            graph.remove_edge(node1, node2)

    subgraph_nodes = weakly_connected_components(graph)
    subgraphs = tuple(graph.subgraph(n) for n in subgraph_nodes)

    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))

    for node1, node2 in cut_edges:
        for i, subgraph in enumerate(subgraphs):
            if subgraph.has_node(node1):
                start_fragment = i
            if subgraph.has_node(node2):
                end_fragment = i

        communication_graph.add_edge(start_fragment, end_fragment, pair=(node1, node2))

    return subgraphs, communication_graph


def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """Converts a circuit graph to the corresponding quantum tape."""
    wires = Wires.all_wires([n.wires for n in graph.nodes])

    ordered_ops = sorted((order, op) for op, order in graph.nodes(data="order"))
    wire_map = {w: w for w in wires}

    with QuantumTape() as tape:
        for _, op in ordered_ops:
            new_wires = [wire_map[w] for w in op.wires]
            op._wires = Wires(new_wires)  # TODO: find a better way to update operation wires
            apply(op)

            if isinstance(op, MeasureNode):
                measured_wire = op.wires[0]
                new_wire = _find_new_wire(wires)
                wires += new_wire
                wire_map[measured_wire] = new_wire

    return tape


def _find_new_wire(wires: Wires) -> int:
    """Finds a new wire label that is not in ``wires``."""
    ctr = 0
    while ctr in wires:
        ctr += 1
    return ctr


def _prep_zero_state(wire):
    Identity(wire)


def _prep_one_state(wire):
    PauliX(wire)


def _prep_plus_state(wire):
    Hadamard(wire)


def _prep_iplus_state(wire):
    Hadamard(wire)
    S(wires=wire)


PREPARE_SETTINGS = [_prep_zero_state, _prep_one_state, _prep_plus_state, _prep_iplus_state]
OBSERVABLE_BASIS = [PauliX, PauliY, PauliZ, Identity]
MEASURE_SETTINGS = [lambda wire: expval(o(wires=wire)) for o in OBSERVABLE_BASIS]


def expand_fragment_tapes(tape: QuantumTape, prepare_settings: Optional[Sequence[Callable]] = PREPARE_SETTINGS, measure_settings: Optional[Sequence[Callable]] = MEASURE_SETTINGS) -> Tuple[Tuple[QuantumTape], Tuple[PrepareNode], Tuple[MeasureNode]]:
    """Expands a fragment tape into a tape for each configuration."""
    prepare_nodes = [o for o in tape.operations if isinstance(o, PrepareNode)]
    measure_nodes = [o for o in tape.operations if isinstance(o, MeasureNode)]

    prepare_settings_prod = [prepare_settings] * len(prepare_nodes)
    measure_settings_prod = [measure_settings] * len(measure_nodes)
    p = product(*(prepare_settings_prod + measure_settings_prod))
    print(list(p))

    tapes = []

    # for

    return prepare_nodes, measure_nodes


def contract(results: Sequence, shapes: Sequence[int], communication_graph: MultiDiGraph):
    """Returns the result of contracting the tensor network."""
    ...
