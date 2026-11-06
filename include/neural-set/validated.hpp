/***************************************************************************
 *            include/neural-set/validated.hpp
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

/** @file neural-set/validated.hpp
 *  @brief Rigorous interval output enclosures of frozen neural networks.
 *
 *  Accepts native FloatDPBoundsVector, FloatDPBallVector or FloatDPExactBox.
 *  The outputs are FloatDPBoundsVector or FloatDPUpperBox.
 *  Weights are exact binary64 reals; inputs describe validated sets.
 *  Unlike Neural's NativeCpu backend, this evaluator uses Ariadne Numeric's
 *  outwardly rounded FloatDPBounds operations without approximate fallbacks.
 */
#ifndef ARIADNE_NEURAL_SET_VALIDATED_HPP
#define ARIADNE_NEURAL_SET_VALIDATED_HPP

#include "ann/network.hpp"
#include "geometry/box.hpp"
#include "algebra/vector.hpp"
#include "algebra/linear_algebra.decl.hpp"
#include "numeric/float_ball.hpp"
#include "numeric/float_bounds.hpp"
#include "function/taylor_model.hpp"
#include "set/function_set.hpp"

namespace Ariadne {

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
    FloatDPUpperBox bounds(FloatDPExactBox const& input_box) const;

    /** @brief Propagate validated bounds through the network. */
    FloatDPBoundsVector bounds(FloatDPBoundsVector const& input) const;

    /** @brief Propagate full balls, including their radii, as bounds. */
    FloatDPBoundsVector bounds(FloatDPBallVector const& input) const;

    /** @brief Propagate validated Taylor models, retaining parameter dependencies.
     *  Each input model must have the same number of independent variables.
     *  ReLU uses Function\'s validated max enclosure, without branch splitting.
     */
    Vector<ValidatedTaylorModelDP> taylor(Vector<ValidatedTaylorModelDP> const& input) const;

    /** @brief Implicit neural graph as one validated constrained image set.
     *  This is the default representation. Original Taylor parameters,
     *  uncertain inputs and ReLU outputs are represented using a shared
     *  extended parameter domain; every ReLU is encoded by nonnegativity,
     *  dominance and complementarity constraints.
     *  A Taylor input's remainder is represented by a bounded auxiliary
     *  input variable. No constraint solver is invoked automatically.
     */
    ValidatedConstrainedImageSet constrained_taylor(
        Vector<ValidatedTaylorModelDP> const& input) const;

    /** @brief Alternative explicit activation-branch representation.
     *  Each branch is a validated constrained image set on the original
     *  Taylor parameter box; ambiguous ReLUs may double the branch count.
     *  @throws std::length_error if max_branches would be exceeded.
     */
    List<ValidatedConstrainedImageSet> branched_taylor(
        Vector<ValidatedTaylorModelDP> const& input,
        SizeType max_branches=1024u) const;

private:
    NeuralParameterSnapshot snapshot_;
};

/** @brief Convenience wrapper for one validated evaluation. */
FloatDPUpperBox neural_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    FloatDPExactBox const& input_box);

} // namespace Ariadne

#endif // ARIADNE_NEURAL_SET_VALIDATED_HPP
