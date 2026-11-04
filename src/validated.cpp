/***************************************************************************
 *            src/validated.cpp
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
#include "numeric/floats.hpp"
#include "numeric/builtin.hpp"
#include "interval/interval.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Ariadne {
namespace {

FloatDPBounds exact(double value) {
    // Preserve the exact binary64 parameter as a singleton enclosure.
    return FloatDPBounds(ExactDouble(value),dp);
}

FloatDPBounds checked(FloatDPBounds value) {
    if (!std::isfinite(value.lower_raw().get_d()) ||
        !std::isfinite(value.upper_raw().get_d()) ||
        value.lower_raw().get_d() > value.upper_raw().get_d())
        throw std::overflow_error("Nonfinite validated neural interval result");
    return value;
}

FloatDPBounds activate(FloatDPBounds const& interval, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Identity:  return interval;
        case NeuralActivation::Relu:      return max(interval, exact(0.0));
        case NeuralActivation::Tanh:      return tanh(interval);
        case NeuralActivation::Sigmoid:   return sigmoid(interval);
        case NeuralActivation::Softplus:  return softplus(interval);
        default: throw std::invalid_argument("Unsupported neural activation");
    }
}

} // namespace

NeurosymbolicValidatedEvaluator::NeurosymbolicValidatedEvaluator(NeuralParameterSnapshot snapshot)
    : snapshot_(std::move(snapshot)) { }

NeuralParameterSnapshot const& NeurosymbolicValidatedEvaluator::snapshot() const noexcept {
    return snapshot_;
}

FloatDPBoundsVector NeurosymbolicValidatedEvaluator::bounds(
    FloatDPBoundsVector const& input) const {
    auto const& network=snapshot_.network();
    if (input.size()!=network.input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    std::vector<FloatDPBounds> state;
    state.reserve(input.size());
    for (SizeType i=0; i<input.size(); ++i) {
        auto const& x=input[i];
        if (!std::isfinite(x.lower_raw().get_d()) ||
            !std::isfinite(x.upper_raw().get_d()) ||
            x.lower_raw().get_d()>x.upper_raw().get_d())
            throw std::invalid_argument("Invalid validated input bounds");
        state.push_back(x);
    }

    auto const& values=snapshot_.values();
    for (auto const& layer : network.layers()) {
        std::vector<FloatDPBounds> output;
        output.reserve(layer.outputs);
        for (std::size_t o=0; o<layer.outputs; ++o) {
            auto sum=exact(values[layer.biases_offset+o]);
            for (std::size_t i=0; i<layer.inputs; ++i) {
                auto const w=exact(values[layer.weights_offset+o*layer.inputs+i]);
                sum=checked(sum+checked(w*state[i]));
            }
            output.push_back(checked(activate(sum,layer.activation)));
        }
        state.swap(output);
    }

    FloatDPBoundsVector result(state.size(),dp);
    for (SizeType i=0; i<state.size(); ++i)
        result[i]=state[i];
    return result;
}

FloatDPBoundsVector NeurosymbolicValidatedEvaluator::bounds(
    FloatDPBallVector const& input) const {
    if (input.size()!=snapshot_.network().input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");
    // Conversion includes the full radius with outward-rounded endpoints.
    FloatDPBoundsVector input_bounds(input.size(),dp);
    for (SizeType i=0; i<input.size(); ++i)
        input_bounds[i]=FloatDPBounds(input[i]);
    return this->bounds(input_bounds);
}

FloatDPUpperBox NeurosymbolicValidatedEvaluator::bounds(
    FloatDPExactBox const& input_box) const {
    if (input_box.dimension()!=snapshot_.network().input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    FloatDPBoundsVector input(input_box.dimension(),dp);
    for (SizeType i=0; i<input_box.dimension(); ++i) {
        auto const& interval=input_box[i];
        auto const lower=interval.lower_bound();
        auto const upper=interval.upper_bound();
        if (!std::isfinite(lower.get_d()) || !std::isfinite(upper.get_d()) ||
            lower.get_d()>upper.get_d())
            throw std::invalid_argument("Invalid validated input interval");
        input[i]=FloatDPBounds(lower,upper);
    }

    auto const output=this->bounds(input);
    FloatDPUpperBox result(output.size());
    for (SizeType i=0; i<output.size(); ++i)
        result[i]=make_interval(output[i]);
    return result;
}

FloatDPUpperBox neurosymbolic_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    FloatDPExactBox const& input_box) {
    return NeurosymbolicValidatedEvaluator(snapshot).bounds(input_box);
}

} // namespace Ariadne
