/***************************************************************************
 *            tests/test_validated.cpp
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

#include "ariadne-neural-set.hpp"
#include "numeric/builtin.hpp"
#include "numeric/floats.hpp"
#include "ann/backend.hpp"
#include "ariadne-testing.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace Ariadne;

class TestValidated : public Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_affine_oracle_and_backend_isolation() {
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{1,NeuralActivation::Identity}}),
            {2.0,-3.0,0.5});
        auto const bounds=NeuralValidatedEvaluator(model).bounds(FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(-1.0),dp),FloatDP(ExactDouble(2.0),dp)), FloatDPExactInterval(FloatDP(ExactDouble(4.0),dp),FloatDP(ExactDouble(5.0),dp))}));
        ARIADNE_TESTING_EXPECT_EQUAL(bounds.dimension(),std::size_t{1});
        ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()<=-16.5);
        ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()>=-7.5);
        ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()>-17.0);
        ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()<-7.0);
        // The base Neural module must not advertize validated bounds.
        ARIADNE_TESTING_PROVE(!neural_backend_available(
            NeuralBackendKind::AriadneValidated));
        ARIADNE_TESTING_PROVE(!neural_backend_supports(
            NeuralBackendKind::NativeCpu,NeuralOperation::ValidatedBounds));
    }

    void test_activations_and_sampled_inclusion() {
        for (auto activation : {NeuralActivation::Identity,
                                NeuralActivation::Relu,
                                NeuralActivation::Tanh,
                                NeuralActivation::Sigmoid,
                                NeuralActivation::Softplus}) {
            NeuralParameterSnapshot model(
                NeuralNetwork(1,{{1,activation}}),
                {1.25,-0.5});
            auto const bounds=neural_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(-3.0),dp),FloatDP(ExactDouble(4.0),dp))}));
            ARIADNE_TESTING_EXPECT_EQUAL(bounds.dimension(),std::size_t{1});
            ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()<=bounds[0].upper_bound().raw().get_d());
            NeuralExecutionSession cpu(model);
            for (double x : {-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0}) {
                auto const y=cpu.infer({x}).at(0);
                ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()<=y);
                ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()>=y);
            }
            if (activation==NeuralActivation::Relu) {
                ARIADNE_TESTING_EXPECT_EQUAL(bounds[0].lower_bound().raw().get_d(),0.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()>=4.5);
            }
            if (activation==NeuralActivation::Sigmoid) {
                ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()>=0.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()<=1.0);
            }
            if (activation==NeuralActivation::Tanh) {
                ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()>=-1.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()<=1.0);
            }
        }
    }

    void test_ball_and_bounds_vectors() {
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{1,NeuralActivation::Identity}}),
            {2.0,-3.0,0.5});
        NeuralValidatedEvaluator evaluator(model);
        FloatDPBoundsVector interval_input({
            FloatDPBounds(ExactDouble(-1.0),ExactDouble(2.0),dp),
            FloatDPBounds(ExactDouble(4.0),ExactDouble(5.0),dp)
        });
        FloatDPBallVector ball_input({
            FloatDPBall(interval_input[0]),FloatDPBall(interval_input[1])
        });
        auto const from_bounds=evaluator.bounds(interval_input);
        auto const from_balls=evaluator.bounds(ball_input);
        ARIADNE_TESTING_EXPECT_EQUAL(from_bounds.size(),SizeType{1});
        ARIADNE_TESTING_EXPECT_EQUAL(from_balls.size(),SizeType{1});
        ARIADNE_TESTING_PROVE(from_bounds[0].lower_raw().get_d()<=-16.5);
        ARIADNE_TESTING_PROVE(from_bounds[0].upper_raw().get_d()>=-7.5);
        ARIADNE_TESTING_PROVE(from_balls[0].lower_raw().get_d()<=-16.5);
        ARIADNE_TESTING_PROVE(from_balls[0].upper_raw().get_d()>=-7.5);
        // A ball's radius must be included, not discarded.
        ARIADNE_TESTING_PROVE(from_balls[0].lower_raw().get_d()
                             <=from_bounds[0].lower_raw().get_d());
        ARIADNE_TESTING_PROVE(from_balls[0].upper_raw().get_d()
                             >=from_bounds[0].upper_raw().get_d());

        for (auto activation : {NeuralActivation::Identity,NeuralActivation::Relu,
                                NeuralActivation::Tanh,NeuralActivation::Sigmoid,
                                NeuralActivation::Softplus}) {
            NeuralParameterSnapshot one(
                NeuralNetwork(1,{{1,activation}}),{1.25,-0.5});
            NeuralValidatedEvaluator one_evaluator(one);
            FloatDPBoundsVector bounds_input({
                FloatDPBounds(ExactDouble(-3.0),ExactDouble(4.0),dp)
            });
            FloatDPBallVector balls_input({FloatDPBall(bounds_input[0])});
            auto const bounded=one_evaluator.bounds(bounds_input);
            auto const balled=one_evaluator.bounds(balls_input);
            NeuralExecutionSession cpu(one);
            for (double x : {-3.0,-1.0,0.0,1.0,4.0}) {
                double y=cpu.infer({x}).at(0);
                ARIADNE_TESTING_PROVE(bounded[0].lower_raw().get_d()<=y);
                ARIADNE_TESTING_PROVE(bounded[0].upper_raw().get_d()>=y);
                ARIADNE_TESTING_PROVE(balled[0].lower_raw().get_d()<=y);
                ARIADNE_TESTING_PROVE(balled[0].upper_raw().get_d()>=y);
            }
        }
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)evaluator.bounds(FloatDPBoundsVector()));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)evaluator.bounds(FloatDPBallVector()));
    }

    void test_composed_network_and_invalid_boxes() {
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{2,NeuralActivation::Relu},
                             {1,NeuralActivation::Identity}}),
            {1.0,-1.0,-1.0,1.0,0.0,0.0,1.0,1.0,0.0});
        auto const bounds=neural_validated_bounds(
            model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(1.0),dp),FloatDP(ExactDouble(3.0),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(2.0),dp))}));
        ARIADNE_TESTING_EXPECT_EQUAL(bounds.dimension(),std::size_t{1});
        NeuralExecutionSession cpu(model);
        for (int i=0;i<=20;++i)
            for (int j=0;j<=20;++j) {
                double x=1.0+static_cast<double>(i)/10.0;
                double y=static_cast<double>(j)/10.0;
                double z=cpu.infer({x,y}).at(0);
                ARIADNE_TESTING_PROVE(bounds[0].lower_bound().raw().get_d()<=z);
                ARIADNE_TESTING_PROVE(bounds[0].upper_bound().raw().get_d()>=z);
            }

        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(2.0),dp),FloatDP(ExactDouble(1.0),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(std::numeric_limits<double>::infinity()),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(std::numeric_limits<double>::quiet_NaN()),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        NeuralParameterSnapshot overflow(
            NeuralNetwork(1,{{1,NeuralActivation::Identity}}),
            {1e308,0.0});
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
            (void)neural_validated_bounds(
                overflow,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(1e308),dp),FloatDP(ExactDouble(1e308),dp))})));
    }
};

int main(int argc,char** argv) {
    ARIADNE_TESTING_SUITE("validated");
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_affine_oracle_and_backend_isolation);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_activations_and_sampled_inclusion);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_composed_network_and_invalid_boxes);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_ball_and_bounds_vectors);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
