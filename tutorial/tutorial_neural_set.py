# ***************************************************************************
# *            tutorial/tutorial_neural_set.py
# *
# *  Copyright  2026  Luca Geretti
# *
# ***************************************************************************
#
# This file is part of Ariadne Neural Set.
#
# Ariadne Neural Set is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ariadne Neural Set is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ariadne Neural Set. If not, see <https://www.gnu.org/licenses/>.

from pyariadne_neural_set import (
    DoublePrecision,
    FloatDP,
    FloatDPBall,
    FloatDPBallVector,
    FloatDPBounds,
    FloatDPBoundsVector,
    FloatDPExactBox,
    FloatDPExactInterval,
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralNetwork,
    NeuralParameterSnapshot,
    NeuralValidatedEvaluator,
    SweeperDP,
    ThresholdSweeperDP,
    ValidatedTaylorModelDP,
    neural_set_version,
)


def snapshot(activation, weight, bias):
    return NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, activation)]),
        [weight, bias],
    )


def main():
    print("NeuralSet", neural_set_version())
    dp = DoublePrecision()

    # 1. A smooth frozen network; binary64 parameters are exact dyadic reals
    #    in validated evaluation.
    smooth = snapshot(NeuralActivation.Tanh, 2.0, 0.0)
    smooth_evaluator = NeuralValidatedEvaluator(smooth)
    input_box = FloatDPExactBox([
        FloatDPExactInterval(FloatDP(-1, dp), FloatDP(1, dp)),
    ])
    boxed = smooth_evaluator.bounds(input_box)
    print("Box -> Tanh enclosure:", boxed)

    # 2. Bounds and Ball inputs, including the complete ball radius.
    bounds_input = FloatDPBoundsVector([
        FloatDPBounds(FloatDP(-1, dp), FloatDP(1, dp)),
    ])
    ball_input = FloatDPBallVector([FloatDPBall(bounds_input[0])])
    bounded = smooth_evaluator.bounds(bounds_input)
    balled = smooth_evaluator.bounds(ball_input)
    print("Bounds -> Tanh enclosure:", bounded)
    print("Ball -> Tanh enclosure:", balled)

    # 3. Existing validated Taylor models maintain parameter dependence.
    sweeper = SweeperDP(ThresholdSweeperDP(dp, 1e-12))
    x = ValidatedTaylorModelDP.coordinate(1, 0, sweeper)
    smooth_taylor = smooth_evaluator.taylor([x])
    print("Tanh Taylor enclosure:", smooth_taylor[0].range())

    # 4. The default ReLU representation is one constrained image set.
    #    Constraints encode Taylor remainder and exact complementarity.
    relu = snapshot(NeuralActivation.Relu, 1.0, 0.0)
    relu_evaluator = NeuralValidatedEvaluator(relu)
    implicit = relu_evaluator.constrained_taylor([x])
    print("Implicit ReLU parameters:", implicit.number_of_parameters())
    print("Implicit ReLU constraints:", implicit.number_of_constraints())
    assert implicit.number_of_parameters() == 3
    assert implicit.number_of_constraints() == 4

    # 5. Explicit branches are optional; solvers are never called by default.
    branches = relu_evaluator.branched_taylor([x], 4)
    print("Explicit ReLU branches:", len(branches))
    assert len(branches) == 2

    # 6. All five Neural activation types support validated bounds.
    for activation in (
        NeuralActivation.Identity,
        NeuralActivation.Relu,
        NeuralActivation.Tanh,
        NeuralActivation.Sigmoid,
        NeuralActivation.Softplus,
    ):
        model = snapshot(activation, 1.0, 0.0)
        outputs = NeuralValidatedEvaluator(model).bounds(bounds_input)
        print("Activation enclosure:", outputs)


if __name__ == "__main__":
    main()
