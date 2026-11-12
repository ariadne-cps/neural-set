/***************************************************************************
 *            test_taylor.cpp
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

/** @file tests/test_taylor.cpp
 *  @brief Test neural Taylor models, constraints and activation branching.
 */

#include <limits>
#include <stdexcept>

#include "algebra/sweeper.hpp"
#include "numeric/builtin.hpp"

#include "ariadne-neural-set.hpp"
#include "ariadne-testing.hpp"

using namespace Ariadne;

class TestTaylor : public Testing::ContextFixture {
  public:
    using ContextFixture::ContextFixture;

    void test_affine_map_preserves_dependencies() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(2u, 0u, swp);
        const auto y = ValidatedTaylorModelDP::coordinate(2u, 1u, swp);
        NeuralParameterSnapshot model(
            NeuralNetwork(2, {
                                 {1, NeuralActivation::Identity}
        }),
            {2.0, -3.0, 0.5});
        NeuralValidatedEvaluator evaluator(model);
        Vector<ValidatedTaylorModelDP> input{x, y};
        const auto output = evaluator.taylor(input);
        ARIADNE_TESTING_EXPECT_EQUAL(output.size(), SizeType{1});
        ARIADNE_TESTING_EXPECT_EQUAL(output[0].argument_size(), SizeType{2});
        // Evaluate on a legal parameter point in the unit box.
        Vector<FloatDPBounds> point{
            FloatDPBounds(ExactDouble(0.25), dp),
            FloatDPBounds(ExactDouble(-0.5), dp)};
        const auto actual = evaluate(output[0], point);
        ARIADNE_TESTING_PROVE(actual.lower_raw().get_d() <= 2.5);
        ARIADNE_TESTING_PROVE(actual.upper_raw().get_d() >= 2.5);
        ARIADNE_TESTING_PROVE(actual.lower_raw().get_d() > 2.499999999999);
        ARIADNE_TESTING_PROVE(actual.upper_raw().get_d() < 2.500000000001);
        // The output must retain both parameter derivatives.
        const auto dx = derivative(output[0], 0u).range();
        const auto dy = derivative(output[0], 1u).range();
        ARIADNE_TESTING_PROVE(dx.lower_bound().raw().get_d() <= 2.0);
        ARIADNE_TESTING_PROVE(dx.upper_bound().raw().get_d() >= 2.0);
        ARIADNE_TESTING_PROVE(dy.lower_bound().raw().get_d() <= -3.0);
        ARIADNE_TESTING_PROVE(dy.upper_bound().raw().get_d() >= -3.0);

        // A parameter-dimension mismatch must not silently discard inputs.
        const auto incompatible = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                                      (void)evaluator.taylor(Vector<ValidatedTaylorModelDP>{x, incompatible}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                                      (void)evaluator.taylor(Vector<ValidatedTaylorModelDP>{x}));
    }

    void test_all_activations_are_validated() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        for (auto activation: {NeuralActivation::Identity,
                               NeuralActivation::Relu,
                               NeuralActivation::Tanh,
                               NeuralActivation::Sigmoid,
                               NeuralActivation::Softplus}) {
            NeuralParameterSnapshot model(
                NeuralNetwork(1, {
                                     {1, activation}
            }),
                {1.0, 0.0});
            NeuralValidatedEvaluator evaluator(model);
            const auto output = evaluator.taylor(Vector<ValidatedTaylorModelDP>{x});
            ARIADNE_TESTING_EXPECT_EQUAL(output.size(), SizeType{1});
            ARIADNE_TESTING_EXPECT_EQUAL(output[0].argument_size(), SizeType{1});
            for (ExactDouble point_value: {-0.5_x, 0.0_x, 0.5_x}) {
                const FloatDPBounds point(point_value, dp);
                FloatDPBounds reference = point;
                switch (activation) {
                    case NeuralActivation::Identity:
                        break;
                    case NeuralActivation::Relu:
                        reference = max(point, FloatDPBounds(0, dp));
                        break;
                    case NeuralActivation::Tanh:
                        reference = tanh(point);
                        break;
                    case NeuralActivation::Sigmoid:
                        reference = sigmoid(point);
                        break;
                    case NeuralActivation::Softplus:
                        reference = softplus(point);
                        break;
                    default:
                        throw std::invalid_argument("Unsupported activation");
                }
                const auto actual = evaluate(output[0], Vector<FloatDPBounds>{point});
                // Independent Numeric evaluation must be contained in the
                // Taylor enclosure, including the ReLU kink at zero.
                ARIADNE_TESTING_PROVE(actual.lower_raw().get_d() <= reference.lower_raw().get_d());
                ARIADNE_TESTING_PROVE(actual.upper_raw().get_d() >= reference.upper_raw().get_d());
            }
        }
    }

    void test_constrained_relu_branches() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        NeuralParameterSnapshot model(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 0.0});
        NeuralValidatedEvaluator evaluator(model);
        const auto branches = evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x});
        ARIADNE_TESTING_EXPECT_EQUAL(branches.size(), SizeType{2});
        for (const auto& branch: branches) {
            ARIADNE_TESTING_EXPECT_EQUAL(branch.dimension(), SizeType{1});
            ARIADNE_TESTING_EXPECT_EQUAL(branch.number_of_parameters(), SizeType{1});
            ARIADNE_TESTING_EXPECT_EQUAL(branch.number_of_constraints(), SizeType{1});
        }
        // A strictly negative parameter admits only the inactive branch,
        // and a positive parameter admits only the active branch.
        for (ExactDouble sample: {-0.75_x, 0.0_x, 0.75_x}) {
            const auto point = Vector<FloatDPBounds>{FloatDPBounds(sample, dp)};
            const auto expected = max(point[0], FloatDPBounds(0, dp));
            bool covered = false;
            for (const auto& branch: branches) {
                const auto guard = branch.constraint(0u);
                const auto guard_value = guard.function()(point);
                if (guard_value.upper_raw().get_d() < guard.lower_bound().get(dp).lower_raw().get_d() ||
                    guard_value.lower_raw().get_d() > guard.upper_bound().get(dp).upper_raw().get_d())
                    continue;
                const auto output = branch.function()(point)[0u];
                if (output.lower_raw().get_d() <= expected.lower_raw().get_d() &&
                    output.upper_raw().get_d() >= expected.upper_raw().get_d())
                    covered = true;
            }
            ARIADNE_TESTING_PROVE(covered);
        }
        ARIADNE_TESTING_EXPECT_THROWS(std::length_error,
                                      (void)evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 1u));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                                      (void)evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 0u));
    }

    void test_remainder_safe_relu_guards() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        x.set_error(1u);
        NeuralParameterSnapshot model(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 0.0});
        const auto branches = NeuralValidatedEvaluator(model).branched_taylor(
            Vector<ValidatedTaylorModelDP>{x});
        ARIADNE_TESTING_EXPECT_EQUAL(branches.size(), SizeType{2});
        // At parameter s=+0.5 an admissible input is s-1=-0.5.
        // Both branches must remain possible because the Taylor error is 1.
        const Vector<FloatDPBounds> point{FloatDPBounds(ExactDouble(0.5), dp)};
        for (const auto& branch: branches) {
            const auto guard = branch.constraint(0u);
            const auto guard_value = guard.function()(point);
            ARIADNE_TESTING_PROVE(
                guard_value.upper_raw().get_d() >= guard.lower_bound().get(dp).lower_raw().get_d());
            ARIADNE_TESTING_PROVE(
                guard_value.lower_raw().get_d() <= guard.upper_bound().get(dp).upper_raw().get_d());
        }
    }

    void test_stable_relu_and_multineuron_branch_limit() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        NeuralParameterSnapshot stable(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 3.0});
        const auto stable_branches = NeuralValidatedEvaluator(stable).branched_taylor(
            Vector<ValidatedTaylorModelDP>{x});
        ARIADNE_TESTING_EXPECT_EQUAL(stable_branches.size(), SizeType{1});
        ARIADNE_TESTING_EXPECT_EQUAL(stable_branches[0].number_of_constraints(), SizeType{0});
        NeuralParameterSnapshot two_neuron_model(
            NeuralNetwork(1, {
                                 {2, NeuralActivation::Relu}
        }),
            {1.0, -1.0, 0.0, 0.0});
        NeuralValidatedEvaluator evaluator(two_neuron_model);
        const auto branches = evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 4u);
        ARIADNE_TESTING_EXPECT_EQUAL(branches.size(), SizeType{4});
        for (const auto& branch: branches)
            ARIADNE_TESTING_EXPECT_EQUAL(branch.number_of_constraints(), SizeType{2});
        ARIADNE_TESTING_EXPECT_THROWS(std::length_error,
                                      (void)evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 3u));
    }

    void test_implicit_relu_graph() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        NeuralParameterSnapshot model(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 0.0});
        NeuralValidatedEvaluator evaluator(model);
        const auto graph = evaluator.constrained_taylor(
            Vector<ValidatedTaylorModelDP>{x});
        // Parameters: Taylor coordinate s, uncertain input x, ReLU output y.
        ARIADNE_TESTING_EXPECT_EQUAL(graph.number_of_parameters(), SizeType{3});
        ARIADNE_TESTING_EXPECT_EQUAL(graph.dimension(), SizeType{1});
        ARIADNE_TESTING_EXPECT_EQUAL(graph.number_of_constraints(), SizeType{4});

        // Both ReLU regimes and the kink must satisfy every constraint.
        for (ExactDouble s: {-0.75_x, 0.0_x, 0.75_x}) {
            const double xv = s.get_d();
            const double yv = xv > 0.0 ? xv : 0.0;
            Vector<FloatDPBounds> point{
                FloatDPBounds(s, dp),
                FloatDPBounds(s, dp),
                FloatDPBounds(ExactDouble(yv), dp)};
            for (SizeType i = 0u; i < graph.number_of_constraints(); ++i) {
                const auto& c = graph.constraint(i);
                const auto v = c.function()(point);
                ARIADNE_TESTING_PROVE(
                    v.upper_raw().get_d() >= c.lower_bound().get(dp).lower_raw().get_d());
                ARIADNE_TESTING_PROVE(
                    v.lower_raw().get_d() <= c.upper_bound().get(dp).upper_raw().get_d());
            }
            const auto output = graph.function()(point);
            ARIADNE_TESTING_PROVE(output[0].lower_raw().get_d() <= yv);
            ARIADNE_TESTING_PROVE(output[0].upper_raw().get_d() >= yv);
        }
    }

    void test_implicit_relu_remainder_and_depth() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        auto uncertain = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        uncertain.set_error(1u);
        NeuralParameterSnapshot one(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 0.0});
        const auto graph = NeuralValidatedEvaluator(one).constrained_taylor(
            Vector<ValidatedTaylorModelDP>{uncertain});
        ARIADNE_TESTING_EXPECT_EQUAL(graph.number_of_constraints(), SizeType{4});
        // At s=+0.5, the remainder permits both x=+0.5 and x=-0.5.
        for (double xv: {-0.5, 0.5}) {
            const double yv = xv > 0.0 ? xv : 0.0;
            const Vector<FloatDPBounds> point{
                FloatDPBounds(ExactDouble(0.5), dp),
                FloatDPBounds(ExactDouble(xv), dp),
                FloatDPBounds(ExactDouble(yv), dp)};
            for (SizeType i = 0u; i < graph.number_of_constraints(); ++i) {
                const auto c = graph.constraint(i);
                const auto v = c.function()(point);
                ARIADNE_TESTING_PROVE(
                    v.upper_raw().get_d() >= c.lower_bound().get(dp).lower_raw().get_d());
                ARIADNE_TESTING_PROVE(
                    v.lower_raw().get_d() <= c.upper_bound().get(dp).upper_raw().get_d());
            }
        }
        NeuralParameterSnapshot two_neuron_model(
            NeuralNetwork(1, {
                                 {2, NeuralActivation::Relu}
        }),
            {1.0, -1.0, 0.0, 0.0});
        const auto two_graph = NeuralValidatedEvaluator(two_neuron_model)
                                   .constrained_taylor(Vector<ValidatedTaylorModelDP>{uncertain});
        ARIADNE_TESTING_EXPECT_EQUAL(two_graph.number_of_parameters(), SizeType{4});
        ARIADNE_TESTING_EXPECT_EQUAL(two_graph.number_of_constraints(), SizeType{7});
        ARIADNE_TESTING_EXPECT_EQUAL(two_graph.dimension(), SizeType{2});
    }

    void test_branched_and_constrained_input_contracts() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        const auto other = ValidatedTaylorModelDP::coordinate(2u, 0u, swp);
        NeuralValidatedEvaluator evaluator(NeuralParameterSnapshot(
            NeuralNetwork(2, {
                                 {1, NeuralActivation::Identity}
        }),
            {1.0, 1.0, 0.0}));
        const Vector<ValidatedTaylorModelDP> empty(0u, x);
        const Vector<ValidatedTaylorModelDP> short_input{x};
        const Vector<ValidatedTaylorModelDP> mismatch{x, other};
        for (const auto& input: std::vector<Vector<ValidatedTaylorModelDP>>{empty, short_input, mismatch}) {
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                                          (void)evaluator.branched_taylor(input));
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                                          (void)evaluator.constrained_taylor(input));
        }
        NeuralValidatedEvaluator negative(NeuralParameterSnapshot(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, -3.0}));
        const auto branches = negative.branched_taylor(Vector<ValidatedTaylorModelDP>{x});
        ARIADNE_TESTING_EXPECT_EQUAL(branches.size(), 1u);
        ARIADNE_TESTING_EXPECT_EQUAL(branches[0].number_of_constraints(), 0u);
        const auto zero = branches[0].function()(Vector<FloatDPBounds>{FloatDPBounds(0, dp)})[0u];
        ARIADNE_TESTING_EXPECT_EQUAL(zero.lower_raw().get_d(), 0.0);
        ARIADNE_TESTING_EXPECT_EQUAL(zero.upper_raw().get_d(), 0.0);

        // The first branch splits at the second layer, while the next one
        // cancels to zero: the final insertion must still enforce the cap.
        NeuralValidatedEvaluator mixed(NeuralParameterSnapshot(
            NeuralNetwork(1, {
                                 {2, NeuralActivation::Relu},
                                 {1, NeuralActivation::Relu}
        }),
            {1.0, 1.0, 0.0, 2.0, -1.0, 1.0, -2.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::length_error,
                                      (void)mixed.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 2u));
        ARIADNE_TESTING_EXPECT_EQUAL(
            mixed.branched_taylor(Vector<ValidatedTaylorModelDP>{x}, 3u).size(), 3u);
    }

    void test_smooth_activation_graphs() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        for (auto activation: {NeuralActivation::Identity, NeuralActivation::Tanh,
                               NeuralActivation::Sigmoid, NeuralActivation::Softplus}) {
            NeuralValidatedEvaluator evaluator(NeuralParameterSnapshot(
                NeuralNetwork(1, {
                                     {1, activation}
            }),
                {1.0, 0.0}));
            const auto branches = evaluator.branched_taylor(Vector<ValidatedTaylorModelDP>{x});
            const auto graph = evaluator.constrained_taylor(Vector<ValidatedTaylorModelDP>{x});
            ARIADNE_TESTING_EXPECT_EQUAL(branches.size(), 1u);
            ARIADNE_TESTING_EXPECT_EQUAL(branches[0].number_of_constraints(), 0u);
            ARIADNE_TESTING_EXPECT_EQUAL(graph.number_of_parameters(), 2u);
            ARIADNE_TESTING_EXPECT_EQUAL(graph.number_of_constraints(), 1u);
            const auto reference = evaluator.bounds(FloatDPBoundsVector{FloatDPBounds(0, dp)})[0u];
            for (const auto& value: std::vector<FloatDPBounds>{
                     branches[0].function()(Vector<FloatDPBounds>{FloatDPBounds(0, dp)})[0u],
                     graph.function()(Vector<FloatDPBounds>{FloatDPBounds(0, dp), FloatDPBounds(0, dp)})[0u]}) {
                ARIADNE_TESTING_PROVE(value.lower_raw().get_d() <= reference.lower_raw().get_d());
                ARIADNE_TESTING_PROVE(value.upper_raw().get_d() >= reference.upper_raw().get_d());
            }
        }
    }

    void test_nonfinite_taylor_and_graph_overflow() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        const auto x = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        NeuralValidatedEvaluator evaluator(NeuralParameterSnapshot(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Identity}
        }),
            {1.0, 0.0}));
        for (double value: {-std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()}) {
            const auto nonfinite = ValidatedTaylorModelDP::constant(
                1u, FloatDPBounds(ExactDouble(value), dp), swp);
            ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
                                          (void)evaluator.taylor(Vector<ValidatedTaylorModelDP>{nonfinite}));
        }
        for (double weight: {-1e308, 1e308}) {
            NeuralValidatedEvaluator overflow(NeuralParameterSnapshot(
                NeuralNetwork(1, {
                                     {1, NeuralActivation::Identity}
            }),
                {weight, 0.0}));
            const auto huge = x.create_constant(FloatDPBounds(ExactDouble(2.0), dp));
            ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
                                          (void)overflow.constrained_taylor(Vector<ValidatedTaylorModelDP>{huge}));
        }
        auto upper_overflow = x;
        upper_overflow.set_value(FloatDP(ExactDouble(1e308), dp));
        upper_overflow.set_gradient(0u, FloatDP(ExactDouble(1e308), dp));
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
                                      (void)evaluator.taylor(Vector<ValidatedTaylorModelDP>{upper_overflow}));
    }

    void test_input_remainder_is_preserved() {
        const Sweeper<FloatDP> swp = ThresholdSweeper<FloatDP>(dp, 1e-12);
        auto uncertain = ValidatedTaylorModelDP::coordinate(1u, 0u, swp);
        uncertain.set_error(1u);
        NeuralParameterSnapshot model(
            NeuralNetwork(1, {
                                 {1, NeuralActivation::Identity}
        }),
            {2.0, 0.5});
        const auto output = NeuralValidatedEvaluator(model).taylor(
            Vector<ValidatedTaylorModelDP>{uncertain});
        const auto value = evaluate(output[0],
                                    Vector<FloatDPBounds>{FloatDPBounds(0, dp)});
        ARIADNE_TESTING_PROVE(value.lower_raw().get_d() <= -1.5);
        ARIADNE_TESTING_PROVE(value.upper_raw().get_d() >= 2.5);
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("taylor");
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_affine_map_preserves_dependencies);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_all_activations_are_validated);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_input_remainder_is_preserved);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_constrained_relu_branches);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_remainder_safe_relu_guards);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_stable_relu_and_multineuron_branch_limit);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_implicit_relu_graph);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_implicit_relu_remainder_and_depth);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_branched_and_constrained_input_contracts);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_smooth_activation_graphs);
    ARIADNE_TESTING_ADD_TEST(TestTaylor, test_nonfinite_taylor_and_graph_overflow);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
