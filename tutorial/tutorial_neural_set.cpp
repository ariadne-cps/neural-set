/***************************************************************************
 *            tutorial_neural_set.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neural Set.
 *
 *  Ariadne Neural Set is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neural Set is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neural Set. If not, see <https://www.gnu.org/licenses/>.
 */

/** @file tutorial/tutorial_neural_set.cpp
 *  @brief Demonstrate validated neural bounds and Taylor-model evaluation.
 */

#include "ariadne-neural-set.hpp"
#include "algebra/sweeper.hpp"
#include "numeric/builtin.hpp"
#include "numeric/floats.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Ariadne;

int main() {
    std::cout << "NeuralSet " << neural_set_version() << '\n';

    // 1. A smooth frozen network. Every stored binary64 coefficient denotes
    //    its exact dyadic real value in validated evaluation.
    NeuralParameterSnapshot smooth(
        NeuralNetwork(1,{{1,NeuralActivation::Tanh}}),{2.0,0.0});
    NeuralValidatedEvaluator smooth_evaluator(smooth);
    FloatDPExactBox input_box({
        FloatDPExactInterval(
            FloatDP(ExactDouble(-1.0),dp),
            FloatDP(ExactDouble(1.0),dp))
    });
    auto const boxed=smooth_evaluator.bounds(input_box);
    std::cout << "Box -> Tanh enclosure: " << boxed << '\n';

    // 2. Both Bounds and Ball inputs work through the same validated
    //    numerical engine. Ball-to-Bounds includes the full ball radius.
    FloatDPBoundsVector bounds_input({
        FloatDPBounds(ExactDouble(-1.0),ExactDouble(1.0),dp)
    });
    FloatDPBallVector ball_input({FloatDPBall(bounds_input[0])});
    auto const bounded=smooth_evaluator.bounds(bounds_input);
    auto const balled=smooth_evaluator.bounds(ball_input);
    std::cout << "Bounds -> Tanh enclosure: " << bounded << '\n';
    std::cout << "Ball -> Tanh enclosure: " << balled << '\n';

    // 3. Native Function Taylor models retain parameter dependence.
    const Sweeper<FloatDP> sweeper=ThresholdSweeper<FloatDP>(dp,1e-12);
    auto const x=ValidatedTaylorModelDP::coordinate(1u,0u,sweeper);
    auto const smooth_taylor=smooth_evaluator.taylor(
        Vector<ValidatedTaylorModelDP>{x});
    std::cout << "Tanh Taylor enclosure: " << smooth_taylor[0].range() << '\n';

    // 4. The default ReLU representation is ONE constrained image set.
    //    The constraints encode the input Taylor error and the exact
    //    complementarity relation y>=0, y>=z, y*(y-z)==0.
    NeuralParameterSnapshot relu(
        NeuralNetwork(1,{{1,NeuralActivation::Relu}}),{1.0,0.0});
    NeuralValidatedEvaluator relu_evaluator(relu);
    auto const implicit=relu_evaluator.constrained_taylor(
        Vector<ValidatedTaylorModelDP>{x});
    std::cout << "Implicit ReLU parameters: "
              << implicit.number_of_parameters() << '\n';
    std::cout << "Implicit ReLU constraints: "
              << implicit.number_of_constraints() << '\n';
    if (implicit.number_of_parameters()!=3u ||
        implicit.number_of_constraints()!=4u)
        throw std::runtime_error("Unexpected implicit ReLU model");

    // 5. The alternative enumerates explicit activation patterns.
    auto const branches=relu_evaluator.branched_taylor(
        Vector<ValidatedTaylorModelDP>{x},4u);
    std::cout << "Explicit ReLU branches: " << branches.size() << '\n';
    if (branches.size()!=2u)
        throw std::runtime_error("Unexpected explicit ReLU branches");
    // Neither method invokes ConstraintSolver automatically. A caller may
    // explicitly use reduce() on a constrained image set when needed.

    // 6. All five Neural activations work with validated bounds.
    for (auto activation : {NeuralActivation::Identity,NeuralActivation::Relu,
                            NeuralActivation::Tanh,NeuralActivation::Sigmoid,
                            NeuralActivation::Softplus}) {
        NeuralParameterSnapshot model(
            NeuralNetwork(1,{{1,activation}}),{1.0,0.0});
        auto const outputs=NeuralValidatedEvaluator(model).bounds(bounds_input);
        std::cout << "Activation enclosure: " << outputs << '\n';
    }
    return 0;
}
