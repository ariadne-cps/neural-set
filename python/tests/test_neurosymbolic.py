# /***************************************************************************
#  *            python/tests/test_neurosymbolic.py
#  *
#  *  Copyright  2026  Luca Geretti
#  *
#  ****************************************************************************/

# /*
#  *  This file is part of Ariadne Neuro-Symbolic.
#  *
#  *  Ariadne Neuro-Symbolic is free software: you can redistribute it and/or
#  *  modify it under the terms of the GNU General Public License as published
#  *  by the Free Software Foundation, either version 3 of the License, or
#  *  (at your option) any later version.
#  *
#  *  Ariadne Neuro-Symbolic is distributed in the hope that it will be useful,
#  *  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  *  GNU General Public License for more details.
#  *
#  *  You should have received a copy of the GNU General Public License
#  *  along with Ariadne Neuro-Symbolic. If not, see <https://www.gnu.org/licenses/>.
#  */

import math

import pytest

from pyariadne import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralNetwork,
    NeuralParameterSnapshot,
    NeuralValidatedInterval,
    NeuralValidatedEvaluator,
    neuro_symbolic_validated_bounds,
    neuro_symbolic_version,
    NeuralExecutionSession,
)


def model(activation):
    return NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, activation)]),
        [1.25, -0.5],
    )


def test_version():
    assert neuro_symbolic_version() == "0.1.0"


def test_exact_affine_corner_oracle():
    snap = NeuralParameterSnapshot(
        NeuralNetwork(2, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)]),
        [2, -3, 0.5],
    )
    result = NeuralValidatedEvaluator(snap).bounds([
        NeuralValidatedInterval(-1, 2),
        NeuralValidatedInterval(4, 5),
    ])
    assert len(result) == 1
    assert -17 < result[0].lower <= -16.5
    assert -7.5 <= result[0].upper < -7


@pytest.mark.parametrize("activation", [
    NeuralActivation.Identity,
    NeuralActivation.Relu,
    NeuralActivation.Tanh,
    NeuralActivation.Sigmoid,
    NeuralActivation.Softplus,
])
def test_all_activations_contain_points(activation):
    snap = model(activation)
    bounds = neuro_symbolic_validated_bounds(
        snap, [NeuralValidatedInterval(-3, 4)])[0]
    cpu = NeuralExecutionSession(snap)
    assert math.isfinite(bounds.lower)
    assert math.isfinite(bounds.upper)
    for x in (-3, -2, -1, 0, 1, 2, 3, 4):
        y = cpu.infer([x])[0]
        # Sampling is falsification support, not a proof of enclosure.
        assert bounds.lower <= y <= bounds.upper


def test_bad_input_box():
    snap = model(NeuralActivation.Identity)
    for bad in ([], [NeuralValidatedInterval(2, 1)],
                [NeuralValidatedInterval(0, math.inf)]):
        with pytest.raises(ValueError):
            neuro_symbolic_validated_bounds(snap, bad)
