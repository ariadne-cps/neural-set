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
#include "interval/interval.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Ariadne {
namespace {

using BoundsType = FloatDPBounds;

BoundsType exact(double value) {
    // A FloatDP initialized from double copies its exact IEEE binary64
    // value: no decimal conversion or rounding of parameters is involved.
    return BoundsType(FloatDP(value));
}

BoundsType checked(BoundsType value) {
    if (!std::isfinite(value.lower_raw().get_d()) ||
        !std::isfinite(value.upper_raw().get_d()) ||
        value.lower_raw().get_d() > value.upper_raw().get_d())
        throw std::overflow_error("Nonfinite validated neural interval result");
    return value;
}

BoundsType activate(BoundsType const& interval, NeuralActivation activation) {
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

FloatDPUpperBox NeurosymbolicValidatedEvaluator::bounds(
    FloatDPExactBox const& input_box) const {
    auto const& network = snapshot_.network();
    if (input_box.dimension() != network.input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    std::vector<BoundsType> state;
    state.reserve(input_box.dimension());
    for (std::size_t i = 0u; i < input_box.dimension(); ++i) {
        auto const& interval=input_box[i];
        auto const lower=interval.lower_bound();
        auto const upper=interval.upper_bound();
        if (!std::isfinite(lower.get_d()) || !std::isfinite(upper.get_d()) ||
            lower.get_d() > upper.get_d())
            throw std::invalid_argument("Invalid validated input interval");
        state.emplace_back(lower, upper);
    }

    auto const& values = snapshot_.values();
    for (auto const& layer : network.layers()) {
        std::vector<BoundsType> output;
        output.reserve(layer.outputs);
        for (std::size_t o=0; o<layer.outputs; ++o) {
            auto sum=exact(values[layer.biases_offset+o]);
            for (std::size_t i=0; i<layer.inputs; ++i) {
                const auto w=exact(values[layer.weights_offset+o*layer.inputs+i]);
                sum=checked(sum+checked(w*state[i]));
            }
            output.push_back(checked(activate(sum,layer.activation)));
        }
        state.swap(output);
    }
    FloatDPUpperBox result(state.size());
    for (std::size_t i = 0u; i < state.size(); ++i)
        result[i]=make_interval(state[i]);
    return result;
}

FloatDPUpperBox neurosymbolic_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    FloatDPExactBox const& input_box) {
    return NeurosymbolicValidatedEvaluator(snapshot).bounds(input_box);
}

} // namespace Ariadne
