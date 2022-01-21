# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the gradients.validate module."""
import pytest

import pennylane as qml
from pennylane.gradients.validate import (
    _validate_backprop_method,
    _validate_device_method,
    _validate_parameter_shift,
    get_best_method,
    get_gradient_fn,
    _validate_adjoint_method,
)


def test_validate_device_method(monkeypatch):
    """Test that the method for validating the device diff method
    tape works as expected"""
    dev = qml.device("default.qubit", wires=1)

    with pytest.raises(
        qml.QuantumFunctionError,
        match="does not provide a native method for computing the jacobian",
    ):
        _validate_device_method(dev)

    monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)
    method, diff_options, device = _validate_device_method(dev)

    assert method == "device"
    assert device is dev


def test_validate_backprop_method_invalid_device():
    """Test that the method for validating the backprop diff method
    tape raises an exception if the device does not support backprop."""
    dev = qml.device("default.gaussian", wires=1)

    with pytest.raises(qml.QuantumFunctionError, match="does not support native computations"):
        _validate_backprop_method(dev, None)


def test_validate_backprop_method_invalid_interface(monkeypatch):
    """Test that the method for validating the backprop diff method
    tape raises an exception if the wrong interface is provided"""
    dev = qml.device("default.qubit", wires=1)
    test_interface = "something"

    monkeypatch.setitem(dev._capabilities, "passthru_interface", test_interface)

    with pytest.raises(qml.QuantumFunctionError, match=f"when using the {test_interface}"):
        _validate_backprop_method(dev, None)


def test_validate_backprop_method(monkeypatch):
    """Test that the method for validating the backprop diff method
    tape works as expected"""
    dev = qml.device("default.qubit", wires=1)
    test_interface = "something"
    monkeypatch.setitem(dev._capabilities, "passthru_interface", test_interface)

    method, diff_options, device = _validate_backprop_method(dev, "something")

    assert method == "backprop"
    assert device is dev


def test_validate_backprop_child_method(monkeypatch):
    """Test that the method for validating the backprop diff method
    tape works as expected if a child device supports backprop"""
    dev = qml.device("default.qubit", wires=1)
    test_interface = "something"

    orig_capabilities = dev.capabilities().copy()
    orig_capabilities["passthru_devices"] = {test_interface: "default.gaussian"}
    monkeypatch.setattr(dev, "capabilities", lambda: orig_capabilities)

    method, diff_options, device = _validate_backprop_method(dev, test_interface)

    assert method == "backprop"
    assert isinstance(device, qml.devices.DefaultGaussian)


def test_validate_backprop_child_method_wrong_interface(monkeypatch):
    """Test that the method for validating the backprop diff method
    tape raises an error if a child device supports backprop but using a different interface"""
    dev = qml.device("default.qubit", wires=1)
    test_interface = "something"

    orig_capabilities = dev.capabilities().copy()
    orig_capabilities["passthru_devices"] = {test_interface: "default.gaussian"}
    monkeypatch.setattr(dev, "capabilities", lambda: orig_capabilities)

    with pytest.raises(qml.QuantumFunctionError, match=r"when using the \['something'\] interface"):
        _validate_backprop_method(dev, "another_interface")


def test_parameter_shift_qubit_device():
    """Test that the _validate_parameter_shift method
    returns the correct gradient transform for qubit devices."""
    dev = qml.device("default.qubit", wires=1)
    gradient_fn = _validate_parameter_shift(dev)
    assert gradient_fn[0] is qml.gradients.param_shift


def test_parameter_shift_cv_device():
    """Test that the _validate_parameter_shift method
    returns the correct gradient transform for cv devices."""
    dev = qml.device("default.gaussian", wires=1)
    gradient_fn = _validate_parameter_shift(dev)
    assert gradient_fn[0] is qml.gradients.param_shift_cv
    assert gradient_fn[1] == {"dev": dev}


def test_parameter_shift_tape_unknown_model(monkeypatch):
    """Test that an unknown model raises an exception"""

    def capabilities(cls):
        capabilities = cls._capabilities
        capabilities.update(model="None")
        return capabilities

    monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
    dev = qml.device("default.qubit", wires=1)

    with pytest.raises(qml.QuantumFunctionError, match="does not support the parameter-shift rule"):
        _validate_parameter_shift(dev)


def test_best_method(monkeypatch):
    """Test that the method for determining the best diff method
    for a given device and interface works correctly"""
    dev = qml.device("default.qubit", wires=1)
    monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
    monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

    # device is top priority
    res = get_best_method(dev, "another_interface")
    assert res == ("device", {}, dev)

    # backprop is next priority
    monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)
    res = get_best_method(dev, "some_interface")
    assert res == ("backprop", {}, dev)

    # The next fallback is parameter-shift.
    res = get_best_method(dev, "another_interface")
    assert res == (qml.gradients.param_shift, {}, dev)

    # finally, if both fail, finite differences is the fallback
    def capabilities(cls):
        capabilities = cls._capabilities
        capabilities.update(model="None")
        return capabilities

    monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
    res = get_best_method(dev, "another_interface")
    assert res == (qml.gradients.finite_diff, {}, dev)


def test_diff_method(mocker):
    """Test that a user-supplied diff method correctly returns the right
    diff method."""
    dev = qml.device("default.qubit", wires=1)

    mock_best = mocker.patch("pennylane.gradients.validate.get_best_method")
    mock_best.return_value = ("best", {}, dev)

    mock_backprop = mocker.patch("pennylane.gradients.validate._validate_backprop_method")
    mock_backprop.return_value = ("backprop", {}, dev)

    mock_device = mocker.patch("pennylane.gradients.validate._validate_device_method")
    mock_device.return_value = ("device", {}, dev)

    g_fn, g_kwargs, dev = get_gradient_fn(dev, interface="autograd", diff_method="best")
    assert g_fn == "best"

    g_fn, g_kwargs, dev = get_gradient_fn(dev, interface="autograd", diff_method="backprop")
    assert g_fn == "backprop"
    mock_backprop.assert_called_once()

    g_fn, g_kwargs, dev = get_gradient_fn(dev, interface="autograd", diff_method="device")
    assert g_fn == "device"
    mock_device.assert_called_once()

    g_fn, g_kwargs, dev = get_gradient_fn(dev, interface="autograd", diff_method="finite-diff")
    assert g_fn is qml.gradients.finite_diff

    g_fn, g_kwargs, dev = get_gradient_fn(dev, interface="autograd", diff_method="parameter-shift")
    assert g_fn is qml.gradients.param_shift

    # check that get_best_method was only ever called once
    mock_best.assert_called_once()


def test_validate_adjoint_invalid_device():
    """Test if a ValueError is raised when an invalid device is provided to
    _validate_adjoint_method"""

    dev = qml.device("default.gaussian", wires=1)

    with pytest.raises(ValueError, match="The default.gaussian device does not"):
        _validate_adjoint_method(dev)

def test_validate_adjoint_finite_shots():
    """Test that a UserWarning is raised when device has finite shots"""

    dev = qml.device("default.qubit", wires=1, shots=1)

    with pytest.warns(
        UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
    ):
        _validate_adjoint_method(dev)
