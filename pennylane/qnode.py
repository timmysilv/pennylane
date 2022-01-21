# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the QNode class and qnode decorator.
"""
import functools
import inspect
import warnings

# pylint: disable=too-many-instance-attributes,too-many-arguments,protected-access
from collections.abc import Sequence

import autograd

import pennylane as qml
from pennylane import Device
from pennylane.interfaces.batch import SUPPORTED_INTERFACES, set_shots
from typing import List, Dict


def default_validator(tapes, devices):
    """Captures which tapes can be run on which devices.

    Args:
        tapes (list[QuantumTape]): the sequence of tapes to be run
        devices (list[Device]): the available devices to run them

    Returns:
        dict[Device, list[QuantumTape]]: a mapping from available devices to the tapes that can
        be run on them

    **Example**

    .. code-block:: python

        import pennylane as qml

        dev1 = qml.device("default.qubit", wires=2)
        dev2 = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape1:
            qml.Toffoli(wires=[0, 1, 2])

        with qml.tape.QuantumTape() as tape2:
            qml.CNOT(wires=[0, 1])

        tapes = [tape1, tape2]
        devs = [dev1, dev2]

    >>> qml.default_validator(tapes, devs)
    {<DefaultQubit device (wires=2, shots=None) at 0x7f43b6b58c10>: [<QuantumTape: wires=[0, 1], params=0>],
     <DefaultQubit device (wires=3, shots=None) at 0x7f43b6bab690>: [<QuantumTape: wires=[0, 1, 2], params=0>,
       <QuantumTape: wires=[0, 1], params=0>]}
    """
    map = {}

    for dev in devices:
        supported = []
        for tape in tapes:
            if tape.num_wires <= dev.num_wires:
                supported.append(tape)
        map[dev] = supported

    return map


def default_distributor(tapes: List, devices: List[qml.Device], validation: Dict[qml.Device, List]):
    """Distributes the sequence of tapes to compatible devices.

    Uses a naive approach of distributing circuits evenly between all compatible devices.
    Compatibility is determined by the ``validation`` argument.

    Args:
        tapes (list[QuantumTape]): the sequence of tapes to be run
        devices (list[Device]): the available devices to run them
        validation (dict[Device, list[QuantumTape]]): a mapping from available devices to the tapes
            that can be run on them

    Returns:
        tuple[dict[Device, list[QuantumTape]], dict[Device, list[int]]]: a mapping from available
        devices to the tapes that should be run on them, as well as a mapping from available devices
        to the order of the contained tapes relative to the input ``tapes`` list

    **Example**

    .. code-block:: python

        import pennylane as qml

        dev1 = qml.device("default.qubit", wires=2)
        dev2 = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape1:
            qml.Toffoli(wires=[0, 1, 2])

        with qml.tape.QuantumTape() as tape2:
            qml.CNOT(wires=[0, 1])

        tapes = [tape1, tape2]
        devs = [dev1, dev2]

        validation = qml.default_validator(tapes, devs)

    >>> qml.default_distributor(tapes, devs, validation)
    ({<DefaultQubit device (wires=2, shots=None) at 0x7fe4f4169310>: [<QuantumTape: wires=[0, 1], params=0>],
      <DefaultQubit device (wires=3, shots=None) at 0x7fe5a00f9a10>: [<QuantumTape: wires=[0, 1], params=0>,
       <QuantumTape: wires=[0, 1, 2], params=0>]},
     {<DefaultQubit device (wires=2, shots=None) at 0x7fe4f4169310>: [1],
      <DefaultQubit device (wires=3, shots=None) at 0x7fe5a00f9a10>: [1, 0]})
    # TODO: this isn't working as expected - the 3-wire device shouldn't have the 2-wire circuit.
    """
    distributed = False
    distribution = {device: [] for device in devices}
    distribution_positions = {device: [] for device in devices}
    available_devices = devices.copy()

    while not distributed:
        for device in available_devices:
            t = validation[device]
            if len(t) == 0:
                available_devices.remove(device)
            else:
                tape = t.pop()
                distribution[device].append(tape)
                distribution_positions[device].append(tapes.index(tape))
        if len(available_devices) == 0:
            distributed = True

    return distribution, distribution_positions


def executor(tapes: List, devices: List[qml.Device], **execute_kwargs):
    validation = default_validator(tapes, devices)
    distribution, distribution_positions = default_distributor(tapes, devices, validation)

    results = [None] * len(tapes)

    for device, tapes in distribution.items():
        res = qml.execute(tapes, device, **execute_kwargs)
        positions = distribution_positions[device]

        for pos, r in zip(positions, res):
            results[pos] = r

    return results


class QNode:
    """Represents a quantum node in the hybrid computational graph.

    A *quantum node* contains a :ref:`quantum function <intro_vcirc_qfunc>`,
    corresponding to a collection of :ref:`variational circuits <glossary_variational_circuit>`
    and the computational devices they can be executed on.

    The QNode calls the quantum function to construct a collection of :class:`~.QuantumTape`
    instances representing quantum circuits.

    Args:
        func (callable): a quantum function
        device (Device or Sequence[Device]): a PennyLane-compatible device or sequence of unique
            devices
        interface (str): The interface that will be used for classical backpropagation.
            This affects the types of objects that can be passed to/returned from the QNode:

            * ``"autograd"``: Allows autograd to backpropagate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``"torch"``: Allows PyTorch to backpropogate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``"tf"``: Allows TensorFlow in eager mode to backpropogate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``"jax"``: Allows JAX to backpropogate
              through the QNode. The QNode accepts and returns
              JAX ``DeviceArray`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str or .gradient_transform): The method used to differentiate circuits in
            the created QNode. Can either be a :class:`~.gradient_transform`, which includes all
            quantum gradient transforms in the :mod:`qml.gradients <.gradients>` module, or a
            string. The following strings are allowed:

            * ``"best"``: Best available method. Uses device-based gradients or classical
              backpropagation for circuits executed on supporting devices, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"device"``: Queries the executing device directly for the gradient.
              Only allowed on devices that provide their own gradient computation.

            * ``"backprop"``: Use classical backpropagation. Only allowed on
              simulator devices that are classically end-to-end differentiable,
              for example :class:`default.qubit <~.DefaultQubit>`. Note that
              the returned QNode can only be used with the machine-learning
              framework supported by the device.

            * ``"adjoint"``: Uses an `adjoint method <https://arxiv.org/abs/2009.02823>`__ that
              reverses through the circuit after a forward pass by iteratively applying the inverse
              (adjoint) gate. Only allowed on supported simulator devices such as
              :class:`default.qubit <~.DefaultQubit>`.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum operation
              arguments.

            * ``None``: QNode cannot be differentiated. Works the same as ``interface=None``.

        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method. Further decompositions required for device execution are performed by the
              device prior to circuit execution.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.

            The ``gradient`` strategy typically results in a reduction in quantum device evaluations
            required during optimization, at the expense of an increase in classical preprocessing.
        max_expansion (int): The number of times each circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``). Only applies
            if devices are queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass.
        cache (bool or dict or Cache): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
            If ``True``, a cache with corresponding ``cachesize`` is created for each batch
            execution. If ``False``, no caching is used. You may also pass your own cache
            to be used; this can be any object that implements the special methods
            ``__getitem__()``, ``__setitem__()``, and ``__delitem__()``, such as a dictionary.
        cachesize (int): The size of any auto-created caches. Only applies when ``cache=True``.
        max_diff (int): If ``diff_method`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.

    Keyword Args:
        **kwargs: Any additional keyword arguments provided are passed to the differentiation
            method. Please refer to the :mod:`qml.gradients <.gradients>` module for details
            on supported options for your chosen gradient transform.

    **Example**

    QNodes can be created by decorating a quantum function:

    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return expval(qml.PauliZ(0))

    or by instantiating the class directly:

    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = qml.QNode(circuit, dev)
    """

    def __init__(
        self,
        func,
        device,
        interface="autograd",
        diff_method="best",
        expansion_strategy="gradient",
        max_expansion=10,
        mode="best",
        cache=True,
        cachesize=10000,
        max_diff=1,
        **gradient_kwargs,
    ):
        if interface not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {SUPPORTED_INTERFACES}."
            )

        self.devices = device if isinstance(device, Sequence) else [device]

        for device in self.devices:
            if not isinstance(device, Device):
                raise qml.QuantumFunctionError(
                    "Invalid device. Device must be a valid PennyLane device."
                )

        if "shots" in inspect.signature(func).parameters:
            warnings.warn(
                "Detected 'shots' as an argument to the given quantum function. "
                "The 'shots' argument name is reserved for overriding the number of shots "
                "taken by the device. Its use outside of this context should be avoided.",
                UserWarning,
            )
            self._qfunc_uses_shots_arg = True
        else:
            self._qfunc_uses_shots_arg = False

        # input arguments
        self.func = func
        self._interface = interface
        self.diff_method = diff_method
        self.expansion_strategy = expansion_strategy
        self.max_expansion = max_expansion

        # execution keyword arguments
        self.execute_kwargs = {
            "mode": mode,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
            "max_expansion": max_expansion,
        }

        if self.expansion_strategy == "device":
            self.execute_kwargs["expand_fn"] = None  # Why not "device"?
        elif self.expansion_strategy == "gradient":
            self.execute_kwargs["expand_fn"] = "gradient"
        else:
            raise ValueError("Invalid expansion strategy")

        # internal data attributes
        self._original_tape = None
        self._tapes = []
        self._qfunc_output = None
        self.gradient_kwargs = gradient_kwargs
        self._qnode_transform = None
        self._processing_fn = None

        functools.update_wrapper(self, func)

    def __repr__(self):
        """String representation."""
        detail = "<QNode: devices='{}', interface='{}', diff_method='{}'>"
        return detail.format(
            ", ".join(device.short_name for device in self.devices),
            self.interface,
            self.diff_method,
        )

    @property
    def interface(self):
        """The interface used by the QNode"""
        return self._interface

    @interface.setter
    def interface(self, value):
        if value not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {value}. Interface must be one of {SUPPORTED_INTERFACES}."
            )

        self._interface = value

    @property
    def device(self):
        if len(self.devices) == 1:
            return self.devices[0]
        else:
            raise ValueError("This QNode contains multiple devices.")

    @property
    def tape(self):
        """The quantum tape"""
        return self._original_tape

    @property
    def tapes(self):
        """The collection of quantum tapes"""
        return self._tapes

    qtape = tape  # for backwards compatibility

    def construct_original_tape(self, args, kwargs):
        """Construct the original tape from the input quantum function"""

        if self.interface == "autograd":
            # HOTFIX: to maintain backwards compatibility existing PennyLane code and demos, here we treat
            # all inputs that do not explicitly specify `requires_grad=False`
            # as trainable. This should be removed at some point, forcing users
            # to specify `requires_grad=True` for trainable parameters.
            args = [
                qml.numpy.array(a, requires_grad=True) if not hasattr(a, "requires_grad") else a
                for a in args
            ]

        self._original_tape = qml.tape.JacobianTape()

        with self.tape:
            self._qfunc_output = self.func(*args, **kwargs)

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

        if not isinstance(self._qfunc_output, Sequence):
            measurement_processes = (self._qfunc_output,)
        else:
            measurement_processes = self._qfunc_output

        if not all(isinstance(m, qml.measure.MeasurementProcess) for m in measurement_processes):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        if not all(ret == m for ret, m in zip(measurement_processes, self.tape.measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.tape.operations + self.tape.observables:

            if getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires:
                # check here only if enough wires
                if len(obj.wires) != self.device.num_wires:
                    raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

    def construct(self, args, kwargs):
        """Construct the collection of quantum tapes in the QNode"""
        self.construct_original_tape(args, kwargs)

        if self._qnode_transform is not None:
            self._tapes, self._processing_fn = self._qnode_transform(self._original_tape)
        else:
            self._tapes = [self._original_tape]
            self._processing_fn = lambda x: x[0]

    def __call__(self, *args, **kwargs):
        override_shots = kwargs.pop("shots", False) if not self._qfunc_uses_shots_arg else False
        execute_fn = kwargs.pop("executor", None) or executor

        # construct the tape
        self.construct(args, kwargs)

        execute_kwargs = {
            **dict(
            gradient_fn=self.diff_method,
            interface=self.interface,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,),
            **self.execute_kwargs
        }

        res = execute_fn(
            self.tapes,
            self.devices,
            **execute_kwargs
        )

        if autograd.isinstance(res, (tuple, list)) and len(res) == 1:
            # If a device batch transform was applied, we need to 'unpack'
            # the returned tuple/list to a float.
            #
            # Note that we use autograd.isinstance, because on the backwards pass
            # with Autograd, lists and tuples are converted to autograd.box.SequenceBox.
            # autograd.isinstance is a 'safer' isinstance check that supports
            # autograd backwards passes.
            #
            # TODO: find a more explicit way of determining that a batch transform
            # was applied.

            res = res[0]

        res = self._processing_fn(res)

        if isinstance(self._qfunc_output, Sequence) or (
            self.tape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        return qml.math.squeeze(res)


qnode = lambda device, **kwargs: functools.partial(QNode, device=device, **kwargs)
qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
