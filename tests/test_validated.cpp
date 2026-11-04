/***************************************************************************
 *            tests/test_validated.cpp
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

#include "ariadne-neurosymbolic.hpp"
#include "numeric/builtin.hpp"
#include "numeric/floats.hpp"
#include "neural/backend.hpp"
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
        auto const bounds=NeurosymbolicValidatedEvaluator(model).bounds(FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(-1.0),dp),FloatDP(ExactDouble(2.0),dp)), FloatDPExactInterval(FloatDP(ExactDouble(4.0),dp),FloatDP(ExactDouble(5.0),dp))}));
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
            auto const bounds=neurosymbolic_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(-3.0),dp),FloatDP(ExactDouble(4.0),dp))}));
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

    void test_composed_network_and_invalid_boxes() {
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{2,NeuralActivation::Relu},
                             {1,NeuralActivation::Identity}}),
            {1.0,-1.0,-1.0,1.0,0.0,0.0,1.0,1.0,0.0});
        auto const bounds=neurosymbolic_validated_bounds(
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
            (void)neurosymbolic_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neurosymbolic_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(2.0),dp),FloatDP(ExactDouble(1.0),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neurosymbolic_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(std::numeric_limits<double>::infinity()),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neurosymbolic_validated_bounds(model,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(std::numeric_limits<double>::quiet_NaN()),dp)),FloatDPExactInterval(FloatDP(ExactDouble(0.0),dp),FloatDP(ExactDouble(1.0),dp))})));
        NeuralParameterSnapshot overflow(
            NeuralNetwork(1,{{1,NeuralActivation::Identity}}),
            {1e308,0.0});
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
            (void)neurosymbolic_validated_bounds(
                overflow,FloatDPExactBox({FloatDPExactInterval(FloatDP(ExactDouble(1e308),dp),FloatDP(ExactDouble(1e308),dp))})));
    }
};

int main(int argc,char** argv) {
    ARIADNE_TESTING_SUITE("validated");
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_affine_oracle_and_backend_isolation);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_activations_and_sampled_inclusion);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_composed_network_and_invalid_boxes);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
