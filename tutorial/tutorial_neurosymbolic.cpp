/***************************************************************************
 *            tutorial/tutorial_neurosymbolic.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neuro-Symbolic.
 *
 *  Ariadne Neuro-Symbolic is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neuro-Symbolic is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neuro-Symbolic. If not, see <https://www.gnu.org/licenses/>.
 */

#include "ariadne-neuro-symbolic.hpp"
#include "numeric/real.hpp"

#include <iostream>

int main() {
    using namespace Ariadne;
    NeuralNetwork model(1,{{1,NeuralActivation::Tanh}});
    NeuralParameterSnapshot snapshot(model,{2.0,0.0});
    auto const enclosure=NeuralValidatedEvaluator(snapshot).bounds({{-1.0,1.0}});
    std::cout << "Neuro-Symbolic " << neuro_symbolic_version() << '\n';
    std::cout << "Validated output: [" << enclosure.at(0).lower
              << ", " << enclosure.at(0).upper << "]\n";
    RealVariable x("x");
    auto const expressions=neuro_symbolic_expand(snapshot,{x});
    std::cout << "Expanded expressions: " << expressions.size() << '\n';
    auto const df=derivative(expressions.at(0),x);
    (void)df;
    return 0;
}
