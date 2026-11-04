/***************************************************************************
 *            tutorial/tutorial_neurosymbolic.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neurosymbolic.
 *
 *  Ariadne Neurosymbolic is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neurosymbolic is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neurosymbolic. If not, see <https://www.gnu.org/licenses/>.
 */

#include "ariadne-neurosymbolic.hpp"
#include "numeric/real.hpp"
#include "numeric/builtin.hpp"
#include "numeric/floats.hpp"

#include <iostream>

int main() {
    using namespace Ariadne;
    NeuralNetwork model(1,{{1,NeuralActivation::Tanh}});
    NeuralParameterSnapshot snapshot(model,{2.0,0.0});
    auto const enclosure=NeurosymbolicValidatedEvaluator(snapshot).bounds(
        FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(-1.0),dp),FloatDP(ExactDouble(1.0),dp))}));
    std::cout << "Neurosymbolic " << neurosymbolic_version() << '\n';
    std::cout << "Validated output: [" << enclosure[0].lower_bound()
              << ", " << enclosure[0].upper_bound() << "]\n";
    RealVariable x("x");
    auto const expressions=neurosymbolic_expand(snapshot,{x});
    std::cout << "Expanded expressions: " << expressions.size() << '\n';
    auto const df=derivative(expressions.at(0),x);
    (void)df;
    return 0;
}
