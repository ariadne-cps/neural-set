# /***************************************************************************
#  *            python/tests/test_neural_set.py
#  *
#  *  Copyright  2026  Luca Geretti
#  *
#  ****************************************************************************/

# /*
#  *  This file is part of Ariadne Neural Set.
#  *
#  *  Ariadne Neural Set is free software: you can redistribute it and/or
#  *  modify it under the terms of the GNU General Public License as published
#  *  by the Free Software Foundation, either version 3 of the License, or
#  *  (at your option) any later version.
#  *
#  *  Ariadne Neural Set is distributed in the hope that it will be useful,
#  *  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  *  GNU General Public License for more details.
#  *
#  *  You should have received a copy of the GNU General Public License
#  *  along with Ariadne Neural Set. If not, see <https://www.gnu.org/licenses/>.
#  */

import math

import pytest

from pyariadne_neural_set import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralNetwork,
    NeuralParameterSnapshot,
    FloatDPExactBox,
    FloatDPExactInterval,
    FloatDP,
    FloatDPBounds,
    FloatDPBall,
    FloatDPBoundsVector,
    FloatDPBallVector,
    DoublePrecision,
    NeuralValidatedEvaluator,
    neural_validated_bounds,
    neural_set_version,
    NeuralExecutionSession,
)


def box(*pairs):
    dp = DoublePrecision()
    return FloatDPExactBox([
        FloatDPExactInterval(FloatDP(left, dp), FloatDP(right, dp))
        for left, right in pairs
    ])


def endpoints(interval):
    return float(interval.lower_bound().raw().get_d()), float(interval.upper_bound().raw().get_d())


def model(activation):
    return NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, activation)]),
        [1.25, -0.5],
    )


def test_version():
    assert neural_set_version() == "0.1.0"


def test_exact_affine_corner_oracle():
    snap = NeuralParameterSnapshot(
        NeuralNetwork(2, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)]),
        [2, -3, 0.5],
    )
    result = NeuralValidatedEvaluator(snap).bounds(
        box((-1, 2), (4, 5)))
    assert result.dimension() == 1
    low, high = endpoints(result[0])
    assert -17 < low <= -16.5
    assert -7.5 <= high < -7


@pytest.mark.parametrize("activation", [
    NeuralActivation.Identity,
    NeuralActivation.Relu,
    NeuralActivation.Tanh,
    NeuralActivation.Sigmoid,
    NeuralActivation.Softplus,
])
def test_all_activations_contain_points(activation):
    snap = model(activation)
    bounds = neural_validated_bounds(
        snap, box((-3, 4)))[0]
    cpu = NeuralExecutionSession(snap)
    low, high = endpoints(bounds)
    assert math.isfinite(low)
    assert math.isfinite(high)
    for x in (-3, -2, -1, 0, 1, 2, 3, 4):
        y = cpu.infer([x])[0]
        # Sampling is falsification support, not a proof of enclosure.
        assert low <= y <= high


def test_bad_input_box():
    snap = model(NeuralActivation.Identity)

    # Construct a zero-dimensional box using the dimension constructor.
    # A list-based FloatDPExactBox([]) cannot infer interval precision.
    with pytest.raises(ValueError):
        neural_validated_bounds(snap, FloatDPExactBox(0))

    # Wrong dimension must be rejected even for a valid, nonempty box.
    with pytest.raises(ValueError):
        neural_validated_bounds(snap, box((0, 1), (0, 1)))

    # The evaluator must reject reversed endpoints, not normalize them.
    with pytest.raises(ValueError):
        neural_validated_bounds(snap, box((2, 1)))


def test_bounds_and_ball_vector_inputs():
    snap = model(NeuralActivation.Identity)
    evaluator = NeuralValidatedEvaluator(snap)
    dp = DoublePrecision()
    interval = FloatDPBounds(FloatDP(-2, dp), FloatDP(3, dp))
    bounds = evaluator.bounds(FloatDPBoundsVector([interval]))
    balls = evaluator.bounds(FloatDPBallVector([FloatDPBall(interval)]))
    assert len(bounds) == 1
    assert len(balls) == 1
    assert bounds[0].lower().raw().get_d() <= -3.0
    assert bounds[0].upper().raw().get_d() >= 3.25
    assert balls[0].lower().raw().get_d() <= bounds[0].lower().raw().get_d()
    assert balls[0].upper().raw().get_d() >= bounds[0].upper().raw().get_d()
