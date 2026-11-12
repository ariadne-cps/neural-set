/***************************************************************************
 *            validated.cpp
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

/** @file src/validated.cpp
 *  @brief Implement outward-rounded bounds for frozen neural networks.
 */

#include "neural-set/validated.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "interval/interval.hpp"
#include "numeric/builtin.hpp"
#include "numeric/floats.hpp"

namespace Ariadne {
namespace {

FloatDPBounds exact(double value) {
    // Preserve the exact binary64 parameter as a singleton enclosure.
    return FloatDPBounds(ExactDouble(value), dp);
}

FloatDPBounds checked(FloatDPBounds value) {
    if (!std::isfinite(value.lower_raw().get_d()) ||
        !std::isfinite(value.upper_raw().get_d()))
        throw std::overflow_error("Nonfinite validated neural interval result");
    return value;
}

FloatDPBounds activate(const FloatDPBounds& interval, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Relu:
            return max(interval, exact(0.0));
        case NeuralActivation::Tanh:
            return tanh(interval);
        case NeuralActivation::Sigmoid:
            return sigmoid(interval);
        case NeuralActivation::Softplus:
            return softplus(interval);
        // NeuralNetwork validates all activations before an immutable
        // snapshot is created; Identity is the sole remaining activation.
        default:
            return interval;
    }
}

} // namespace

NeuralValidatedEvaluator::NeuralValidatedEvaluator(NeuralParameterSnapshot snapshot)
    : snapshot_(std::move(snapshot)) {}

const NeuralParameterSnapshot& NeuralValidatedEvaluator::snapshot() const noexcept {
    return snapshot_;
}

FloatDPBoundsVector NeuralValidatedEvaluator::bounds(
    const FloatDPBoundsVector& input) const {
    const auto& network = snapshot_.network();
    if (input.size() != network.input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    std::vector<FloatDPBounds> state;
    state.reserve(input.size());
    for (SizeType i = 0; i < input.size(); ++i) {
        const auto& x = input[i];
        if (!std::isfinite(x.lower_raw().get_d()) ||
            !std::isfinite(x.upper_raw().get_d()) ||
            x.lower_raw().get_d() > x.upper_raw().get_d())
            throw std::invalid_argument("Invalid validated input bounds");
        state.push_back(x);
    }

    const auto& values = snapshot_.values();
    for (const auto& layer: network.layers()) {
        std::vector<FloatDPBounds> output;
        output.reserve(layer.outputs);
        for (std::size_t o = 0; o < layer.outputs; ++o) {
            auto sum = exact(values[layer.biases_offset + o]);
            for (std::size_t i = 0; i < layer.inputs; ++i) {
                const auto w = exact(values[layer.weights_offset + o * layer.inputs + i]);
                sum = checked(sum + checked(w * state[i]));
            }
            output.push_back(checked(activate(sum, layer.activation)));
        }
        state.swap(output);
    }

    FloatDPBoundsVector result(state.size(), dp);
    for (SizeType i = 0; i < state.size(); ++i)
        result[i] = state[i];
    return result;
}

FloatDPBoundsVector NeuralValidatedEvaluator::bounds(
    const FloatDPBallVector& input) const {
    if (input.size() != snapshot_.network().input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");
    // Conversion includes the full radius with outward-rounded endpoints.
    FloatDPBoundsVector input_bounds(input.size(), dp);
    for (SizeType i = 0; i < input.size(); ++i)
        input_bounds[i] = FloatDPBounds(input[i]);
    return this->bounds(input_bounds);
}

FloatDPUpperBox NeuralValidatedEvaluator::bounds(
    const FloatDPExactBox& input_box) const {
    if (input_box.dimension() != snapshot_.network().input_dimension())
        throw std::invalid_argument("Wrong number of validated neural inputs");

    FloatDPBoundsVector input(input_box.dimension(), dp);
    for (SizeType i = 0; i < input_box.dimension(); ++i) {
        const auto& interval = input_box[i];
        const auto lower = interval.lower_bound();
        const auto upper = interval.upper_bound();
        if (!std::isfinite(lower.get_d()) || !std::isfinite(upper.get_d()) ||
            lower.get_d() > upper.get_d())
            throw std::invalid_argument("Invalid validated input interval");
        input[i] = FloatDPBounds(lower, upper);
    }

    const auto output = this->bounds(input);
    FloatDPUpperBox result(output.size());
    for (SizeType i = 0; i < output.size(); ++i)
        result[i] = make_interval(output[i]);
    return result;
}

FloatDPUpperBox neural_validated_bounds(
    const NeuralParameterSnapshot& snapshot,
    const FloatDPExactBox& input_box) {
    return NeuralValidatedEvaluator(snapshot).bounds(input_box);
}

} // namespace Ariadne
