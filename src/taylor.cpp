/***************************************************************************
 *            src/taylor.cpp
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

#include "neurosymbolic/validated.hpp"
#include "numeric/builtin.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace Ariadne {
namespace {

FloatDPBounds exact_taylor_parameter(double value) {
    // A stored binary64 coefficient denotes an exact dyadic real.
    return FloatDPBounds(ExactDouble(value),dp);
}

void check_finite_taylor(ValidatedTaylorModelDP const& model) {
    auto const enclosure=model.range();
    if (!std::isfinite(enclosure.lower_bound().raw().get_d()) ||
        !std::isfinite(enclosure.upper_bound().raw().get_d()))
        throw std::overflow_error("Nonfinite validated neural Taylor enclosure");
}

ValidatedTaylorModelDP activate_taylor(
    ValidatedTaylorModelDP const& value, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Identity: return value;
        case NeuralActivation::Relu:
            return max(value,value.create_constant(exact_taylor_parameter(0.0)));
        case NeuralActivation::Tanh: return tanh(value);
        case NeuralActivation::Sigmoid: return rec(1+exp(-value));
        case NeuralActivation::Softplus: return log(1+exp(value));
        default: throw std::invalid_argument("Unsupported neural activation");
    }
}

} // namespace

Vector<ValidatedTaylorModelDP> NeurosymbolicValidatedEvaluator::taylor(
    Vector<ValidatedTaylorModelDP> const& input) const {
    auto const& network=snapshot_.network();
    if (input.size()!=network.input_dimension())
        throw std::invalid_argument("Wrong number of neural Taylor inputs");

    auto const parameter_dimension=input[0].argument_size();
    std::vector<ValidatedTaylorModelDP> state;
    state.reserve(input.size());
    for (SizeType i=0; i<input.size(); ++i) {
        if (input[i].argument_size()!=parameter_dimension)
            throw std::invalid_argument("Incompatible neural Taylor parameter dimensions");
        check_finite_taylor(input[i]);
        state.push_back(input[i]);
    }

    auto const& parameters=snapshot_.values();
    for (auto const& layer : network.layers()) {
        std::vector<ValidatedTaylorModelDP> output;
        output.reserve(layer.outputs);
        for (std::size_t o=0; o<layer.outputs; ++o) {
            auto sum=state[0].create_constant(
                exact_taylor_parameter(parameters[layer.biases_offset+o]));
            for (std::size_t i=0; i<layer.inputs; ++i) {
                sum+=state[i]*exact_taylor_parameter(
                    parameters[layer.weights_offset+o*layer.inputs+i]);
                check_finite_taylor(sum);
            }
            auto activated=activate_taylor(sum,layer.activation);
            check_finite_taylor(activated);
            output.push_back(std::move(activated));
        }
        state.swap(output);
    }

    Vector<ValidatedTaylorModelDP> result(state.size(),state[0]);
    for (SizeType i=0; i<state.size(); ++i)
        result[i]=state[i];
    return result;
}

} // namespace Ariadne
