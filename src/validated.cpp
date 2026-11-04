/***************************************************************************
 *            src/validated.cpp
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

#include "neuro_symbolic/validated.hpp"
#include "numeric/floats.hpp"

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

NeuralValidatedEvaluator::NeuralValidatedEvaluator(NeuralParameterSnapshot snapshot)
    : snapshot_(std::move(snapshot)) { }

NeuralParameterSnapshot const& NeuralValidatedEvaluator::snapshot() const noexcept {
    return snapshot_;
}

std::vector<NeuralValidatedInterval> NeuralValidatedEvaluator::bounds(
    std::vector<NeuralValidatedInterval> const& input_box) const {
    auto const& network = snapshot_.network();
    if (input_box.size() != network.input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    std::vector<BoundsType> state;
    state.reserve(input_box.size());
    for (auto const& x : input_box) {
        if (!std::isfinite(x.lower) || !std::isfinite(x.upper) ||
            x.lower > x.upper)
            throw std::invalid_argument("Invalid validated input interval");
        state.emplace_back(FloatDP(x.lower), FloatDP(x.upper));
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
    std::vector<NeuralValidatedInterval> result;
    result.reserve(state.size());
    for (auto const& value : state)
        result.push_back({value.lower_raw().get_d(),
                          value.upper_raw().get_d()});
    return result;
}

std::vector<NeuralValidatedInterval> neuro_symbolic_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    std::vector<NeuralValidatedInterval> const& input_box) {
    return NeuralValidatedEvaluator(snapshot).bounds(input_box);
}

} // namespace Ariadne
