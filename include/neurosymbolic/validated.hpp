/***************************************************************************
 *            include/neurosymbolic/validated.hpp
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

/** @file neurosymbolic/validated.hpp
 *  @brief Rigorous interval output enclosures of frozen neural networks.
 *
 *  The input is Ariadne's existing FloatDPExactBox (a Box of exact
 *  intervals); the result is FloatDPUpperBox, its validated enclosure.
 *  All weights and inputs represent their exact binary64 real values.
 *  Unlike Neural's NativeCpu backend, this evaluator uses Ariadne Numeric's
 *  outwardly rounded FloatDPBounds operations without approximate fallbacks.
 */
#ifndef ARIADNE_NEUROSYMBOLIC_VALIDATED_HPP
#define ARIADNE_NEUROSYMBOLIC_VALIDATED_HPP

#include "neural/network.hpp"
#include "geometry/box.hpp"

namespace Ariadne {

/** @brief Validated interval-bound propagation of a frozen dense network.
 *
 *  Supports Identity, ReLU, Tanh, Sigmoid and Softplus. Dependency loss
 *  may cause overestimation. All intermediate results must remain finite.
 *  The mathematical meaning is exact binary64 coefficients and real inputs
 *  within the closed input box (not bitwise CPU inference semantics).
 */
class NeurosymbolicValidatedEvaluator {
public:
    explicit NeurosymbolicValidatedEvaluator(NeuralParameterSnapshot snapshot);

    /** @brief Immutable snapshot used by the evaluator. */
    NeuralParameterSnapshot const& snapshot() const noexcept;

    /** @brief Enclose every real-valued network output for an input box.
     *
     *  @throws std::invalid_argument for wrong dimension, nonfinite or
     *  reversed input endpoints.
     *  @throws std::overflow_error for nonfinite intermediate enclosures.
     */
    FloatDPUpperBox bounds(FloatDPExactBox const& input_box) const;

private:
    NeuralParameterSnapshot snapshot_;
};

/** @brief Convenience wrapper for one validated evaluation. */
FloatDPUpperBox neurosymbolic_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    FloatDPExactBox const& input_box);

} // namespace Ariadne

#endif // ARIADNE_NEUROSYMBOLIC_VALIDATED_HPP
