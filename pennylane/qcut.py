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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph

from pennylane.operation import AnyWires, Operation, Operator
from pennylane.tape import QuantumTape
from pennylane.transforms import batch_transform
from pennylane.measure import MeasurementProcess


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

    # Iterate over ``WireCut`` operations in tape and remove them using remove_wire_cut_node()
    for op in tape.operations:
        if isinstance(op, WireCut):
            remove_wire_cut_node(op, g)

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
    graph.add_nodes_from(tape.operations)
    graph.add_nodes_from(tape.measurements)

    wire_latest_node = {w: None for w in tape.wires}

    for op in tape.operations + tape.measurements:
        for wire in op.wires:
            if wire_latest_node[wire] is not None:
                parent_op = wire_latest_node[wire]
                graph.add_edge(parent_op, op, wire=wire)

            if not isinstance(op, MeasurementProcess):
                wire_latest_node[wire] = op

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

    graph.remove_node(node)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)
        graph.add_node(meas)
        graph.add_node(prep)

        graph.add_edge(meas, prep, pair=(meas, prep), wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas, wire=wire)
        if successor is not None:
            graph.add_edge(prep, successor, wire=wire)


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
    ...


def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """Converts a circuit graph to the corresponding quantum tape."""
    ...


def expand_fragment_tapes(tape: QuantumTape) -> Tuple[QuantumTape]:
    """Expands a fragment tape into a tape for each configuration."""
    ...


def contract(results: Sequence, shapes: Sequence[int], communication_graph: MultiDiGraph):
    """Returns the result of contracting the tensor network."""
    ...
