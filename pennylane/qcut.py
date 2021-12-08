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

from networkx import DiGraph

from pennylane.operation import AnyWires, Operation, Operator
from pennylane.tape import QuantumTape
from pennylane.transforms import batch_transform


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
    g = tape.graph.graph.copy()

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


def remove_wire_cut_node(node: WireCut, graph: DiGraph):
    """Removes a WireCut node from the graph"""
    ...


def find_and_place_cuts(graph: DiGraph, method: Union[str, Callable], **kwargs):
    """Automatically find additional cuts and place them in the graph. A ``method`` can be
    explicitly passed as a callable, or built-in ones can be used by specifying the corresponding
    string."""
    ...  # calls ``method`` (see ``example_method``) and ``place_cuts``


def example_method(
    graph: DiGraph,
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


def place_cuts(graph: DiGraph, wires: Tuple[Tuple[Operator, Operator, Any]]):
    """Places wire cuts in ``graph`` according to ``wires`` which contains pairs of operators along
    with the wire passing between them to be cut."""
    ...


def fragment_graph(graph: DiGraph) -> Tuple[Tuple[DiGraph], DiGraph]:
    """Fragments a cut graph into a collection of subgraphs as well as returning the
    communication/quotient graph."""
    ...


def graph_to_tape(graph: DiGraph) -> QuantumTape:
    """Converts a circuit graph to the corresponding quantum tape."""
    ...


def expand_fragment_tapes(tape: QuantumTape) -> Tuple[QuantumTape]:
    """Expands a fragment tape into a tape for each configuration."""
    ...


def contract(results: Sequence, shapes: Sequence[int], communication_graph: DiGraph):
    """Returns the result of contracting the tensor network."""
    ...
