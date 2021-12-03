# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for marking manually-placed cut locations"""

from pennylane.operation import Operation, AnyWires
from pennylane.ops.identity import Identity
from pennylane.tape import QuantumTape


class wire(Operation):
    r"""pennylane.cut.wire(wires)
    Manually place a wire cut in a circuit.

    Marks the placement of a wire cut of the type introduced in
    `Peng et al. <https://arxiv.org/abs/1904.00102>`__. Behaves like an :class:`~.Identity`
    operation if cutting subsequent functionality is not applied to the whole circuit.

    Args:
        wires (Sequence[int] or int): the wires to be cut
    """
    num_wires = AnyWires
    grad_method = None

    def label(self, decimals=None, base_label=None):
        return "|️"

    def expand(self):
        with QuantumTape() as tape:
            ...
        return tape
