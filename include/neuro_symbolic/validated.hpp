/***************************************************************************
 *            include/neuro_symbolic/validated.hpp
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

/** @file neuro_symbolic/validated.hpp
 *  @brief Rigorous interval output enclosures of frozen neural networks.
 *
 *  All weights and inputs represent their exact binary64 real values.
 *  Unlike Neural's NativeCpu backend, this evaluator uses Ariadne Numeric's
 *  outwardly rounded FloatDPBounds operations without approximate fallbacks.
 */
#ifndef ARIADNE_NEURO_SYMBOLIC_VALIDATED_HPP
#define ARIADNE_NEURO_SYMBOLIC_VALIDATED_HPP

#include "neural/network.hpp"

#include <vector>

namespace Ariadne {

/** @brief A closed interval with exact binary64 endpoints. */
struct NeuralValidatedInterval {
    double lower;
    double upper;
};

/** @brief Validated interval-bound propagation of a frozen dense network.
 *
 *  Supports Identity, ReLU, Tanh, Sigmoid and Softplus. Dependency loss
 *  may cause overestimation. All intermediate results must remain finite.
 *  The mathematical meaning is exact binary64 coefficients and real inputs
 *  within the closed input box (not bitwise CPU inference semantics).
 */
class NeuralValidatedEvaluator {
public:
    explicit NeuralValidatedEvaluator(NeuralParameterSnapshot snapshot);

    /** @brief Immutable snapshot used by the evaluator. */
    NeuralParameterSnapshot const& snapshot() const noexcept;

    /** @brief Enclose every real-valued network output for an input box.
     *
     *  @throws std::invalid_argument for wrong dimension, nonfinite or
     *  reversed input endpoints.
     *  @throws std::overflow_error for nonfinite intermediate enclosures.
     */
    std::vector<NeuralValidatedInterval> bounds(
        std::vector<NeuralValidatedInterval> const& input_box) const;

private:
    NeuralParameterSnapshot snapshot_;
};

/** @brief Convenience wrapper for one validated evaluation. */
std::vector<NeuralValidatedInterval> neuro_symbolic_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    std::vector<NeuralValidatedInterval> const& input_box);

} // namespace Ariadne

#endif // ARIADNE_NEURO_SYMBOLIC_VALIDATED_HPP
