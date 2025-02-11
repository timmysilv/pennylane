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
"""
This module contains functions to load circuits from other frameworks as
PennyLane templates.
"""
from collections import defaultdict
from importlib import metadata
from sys import version_info


# Error message to show when the PennyLane-Qiskit plugin is required but missing.
_MISSING_QISKIT_PLUGIN_MESSAGE = (
    "Conversion from Qiskit requires the PennyLane-Qiskit plugin. "
    "You can install the plugin by running: pip install pennylane-qiskit. "
    "You may need to restart your kernel or environment after installation. "
    "If you have any difficulties, you can reach out on the PennyLane forum at "
    "https://discuss.pennylane.ai/c/pennylane-plugins/pennylane-qiskit/"
)

# get list of installed plugin converters
__plugin_devices = (
    defaultdict(tuple, metadata.entry_points())["pennylane.io"]
    if version_info[:2] == (3, 9)
    else metadata.entry_points(group="pennylane.io")  # pylint:disable=unexpected-keyword-arg
)
plugin_converters = {entry.name: entry for entry in __plugin_devices}


def load(quantum_circuit_object, format: str, **load_kwargs):
    r"""Load external quantum assembly and quantum circuits from supported frameworks
    into PennyLane templates.

    .. note::

        For more details on which formats are supported
        please consult the corresponding plugin documentation:
        https://pennylane.ai/plugins.html

    **Example:**

    >>> qc = qiskit.QuantumCircuit(2)
    >>> qc.rz(0.543, [0])
    >>> qc.cx(0, 1)
    >>> my_circuit = qml.load(qc, format='qiskit')

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quantum_circuit_object: the quantum circuit that will be converted
            to a PennyLane template
        format (str): the format of the quantum circuit object to convert from
        **load_kwargs: keyword argument to pass when converting the quantum circuit
            using the plugin

    Keyword Args:
        measurements (list[MeasurementProcess]): the list of PennyLane measurements that
            overrides the terminal measurements that may be present in the imput circuit.
            Currently, only supported for Qiskit's `QuantumCircuit <https://docs.pennylane.ai/projects/qiskit>`_.

    Returns:
        function: the PennyLane template created from the quantum circuit
        object
    """

    if format in plugin_converters:
        # loads the plugin load function
        plugin_converter = plugin_converters[format].load()

        # calls the load function of the converter on the quantum circuit object
        return plugin_converter(quantum_circuit_object, **(load_kwargs or {}))

    raise ValueError(
        "Converter does not exist. Make sure the required plugin is installed "
        "and supports conversion."
    )


def from_qiskit(quantum_circuit, measurements=None):
    """Loads Qiskit `QuantumCircuit <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`_
    objects by using the converter in the PennyLane-Qiskit plugin.

    **Example:**

    >>> qc = qiskit.QuantumCircuit(2)
    >>> qc.rz(0.543, [0])
    >>> qc.cx(0, 1)
    >>> my_circuit = qml.from_qiskit(qc)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quantum_circuit (qiskit.QuantumCircuit): a quantum circuit created in qiskit
        measurements (list[MeasurementProcess]): the list of PennyLane measurements that
            overrides the terminal measurements that may be present in the input circuit.

    Returns:
        function: the PennyLane template created based on the ``QuantumCircuit`` object

    .. details::
        :title: Usage Details

        The ``measurement`` keyword allows one to add a list of PennyLane measurements
        that will override the terminal measurements present in their ``QuantumCircuit``.

        .. code-block:: python

            import pennylane as qml
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.measure(0, 0)
            qc.rz(0.24, [0])
            qc.cx(0, 1)
            qc.measure_all()

            measurements = [qml.expval(qml.Z(0)), qml.vn_entropy([1])]
            quantum_circuit = qml.from_qiskit(qc, measurements=measurements)

            @qml.qnode(qml.device("default.qubit"))
            def circuit_loaded_qiskit_circuit():
                return quantum_circuit()

        >>> print(qml.draw(circuit_loaded_qiskit_circuit)())
        0: ──H──┤↗├──RZ(0.24)─╭●─┤  <Z>
        1: ───────────────────╰X─┤  vnentropy

    """
    try:
        return load(quantum_circuit, format="qiskit", measurements=measurements)
    except ValueError as e:
        if e.args[0].split(".")[0] == "Converter does not exist":
            raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e
        raise e


def from_qiskit_op(qiskit_op, params=None, wires=None):
    """Loads Qiskit `SparsePauliOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`_
    objects by using the converter in the PennyLane-Qiskit plugin.

    Args:
        qiskit_op (qiskit.quantum_info.SparsePauliOp): the ``SparsePauliOp`` to be converted
        params (Any): optional assignment of coefficient values for the ``SparsePauliOp``; see the
            `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp#assign_parameters>`_
            to learn more about the expected format of these parameters
        wires (Sequence | None): optional assignment of wires for the converted ``SparsePauliOp``;
            if the original ``SparsePauliOp`` acted on :math:`N` qubits, then this must be a
            sequence of length :math:`N`

    Returns:
        Operator: The equivalent PennyLane operator.

    .. note::

        The wire ordering convention differs between PennyLane and Qiskit: PennyLane wires are
        enumerated from left to right, while the Qiskit convention is to enumerate from right to
        left. This means a ``SparsePauliOp`` term defined by the string ``"XYZ"`` applies ``Z`` on
        wire 0, ``Y`` on wire 1, and ``X`` on wire 2. For more details, see the
        `String representation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`_
        section of the Qiskit documentation for the ``Pauli`` class.

    **Example**

    Consider the following script which creates a Qiskit ``SparsePauliOp``:

    .. code-block:: python

        from qiskit.quantum_info import SparsePauliOp

        qiskit_op = SparsePauliOp(["II", "XY"])

    The ``SparsePauliOp`` contains two terms and acts over two qubits:

    >>> qiskit_op
    SparsePauliOp(['II', 'XY'],
                  coeffs=[1.+0.j, 1.+0.j])

    To convert the ``SparsePauliOp`` into a PennyLane :class:`Operator`, use:

    >>> import pennylane as qml
    >>> qml.from_qiskit_op(qiskit_op)
    I(0) + X(1) @ Y(0)

    .. details::
        :title: Usage Details

        You can convert a parameterized ``SparsePauliOp`` into a PennyLane operator by assigning
        literal values to each coefficient parameter. For example, the script

        .. code-block:: python

            import numpy as np
            from qiskit.circuit import Parameter

            a, b, c = [Parameter(var) for var in "abc"]
            param_qiskit_op = SparsePauliOp(["II", "XZ", "YX"], coeffs=np.array([a, b, c]))

        defines a ``SparsePauliOp`` with three coefficients (parameters):

        >>> param_qiskit_op
        SparsePauliOp(['II', 'XZ', 'YX'],
              coeffs=[ParameterExpression(1.0*a), ParameterExpression(1.0*b),
         ParameterExpression(1.0*c)])

        The ``SparsePauliOp`` can be converted into a PennyLane operator by calling the conversion
        function and specifying the value of each parameter using the ``params`` argument:

        >>> qml.from_qiskit_op(param_qiskit_op, params={a: 2, b: 3, c: 4})
        (
            (2+0j) * I(0)
          + (3+0j) * (X(1) @ Z(0))
          + (4+0j) * (Y(1) @ X(0))
        )

        Similarly, a custom wire mapping can be applied to a ``SparsePauliOp`` as follows:

        >>> wired_qiskit_op = SparsePauliOp("XYZ")
        >>> wired_qiskit_op
        SparsePauliOp(['XYZ'],
              coeffs=[1.+0.j])
        >>> qml.from_qiskit_op(wired_qiskit_op, wires=[3, 5, 7])
        Y(5) @ Z(3) @ X(7)
    """
    try:
        return load(qiskit_op, format="qiskit_op", params=params, wires=wires)
    except ValueError as e:
        if e.args[0].split(".")[0] == "Converter does not exist":
            raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e
        raise e


def from_qasm(quantum_circuit: str):
    """Loads quantum circuits from a QASM string using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    .. code-block:: python

        >>> hadamard_qasm = 'OPENQASM 2.0;' \\
        ...                 'include "qelib1.inc";' \\
        ...                 'qreg q[1];' \\
        ...                 'h q[0];'
        >>> my_circuit = qml.from_qasm(hadamard_qasm)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quantum_circuit (str): a QASM string containing a valid quantum circuit

    Returns:
        function: the PennyLane template created based on the QASM string
    """
    return load(quantum_circuit, format="qasm")


def from_qasm_file(qasm_filename: str):
    """Loads quantum circuits from a QASM file using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    >>> my_circuit = qml.from_qasm("hadamard_circuit.qasm")

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        qasm_filename (str): path to a QASM file containing a valid quantum circuit

    Returns:
        function: the PennyLane template created based on the QASM file
    """
    return load(qasm_filename, format="qasm_file")


def from_pyquil(pyquil_program):
    """Loads pyQuil Program objects by using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    >>> program = pyquil.Program()
    >>> program += pyquil.gates.H(0)
    >>> program += pyquil.gates.CNOT(0, 1)
    >>> my_circuit = qml.from_pyquil(program)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=[1, 0])
    >>>     return qml.expval(qml.Z(0))

    Args:
        pyquil_program (pyquil.Program): a program created in pyQuil

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(pyquil_program, format="pyquil_program")


def from_quil(quil: str):
    """Loads quantum circuits from a Quil string using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    .. code-block:: python

        >>> quil_str = 'H 0\\n'
        ...            'CNOT 0 1'
        >>> my_circuit = qml.from_quil(quil_str)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quil (str): a Quil string containing a valid quantum circuit

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(quil, format="quil")


def from_quil_file(quil_filename: str):
    """Loads quantum circuits from a Quil file using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    >>> my_circuit = qml.from_quil_file("teleportation.quil")

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quil_filename (str): path to a Quil file containing a valid quantum circuit

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(quil_filename, format="quil_file")
