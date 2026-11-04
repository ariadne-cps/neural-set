/***************************************************************************
 *            tests/test_symbolic.cpp
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

#include "ariadne-neuro-symbolic.hpp"
#include "ariadne-testing.hpp"
#include "numeric/builtin.hpp"
#include "numeric/real.hpp"

#include <stdexcept>
#include <vector>

using namespace Ariadne;

class TestSymbolic : public Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_affine_exact_coefficients_and_derivatives() {
        RealVariable x("x"), y("y");
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{1,NeuralActivation::Identity}}),
            {2.0,-3.0,0.5});
        auto const expr=neuro_symbolic_expand(model,{x,y});
        ARIADNE_TESTING_EXPECT_EQUAL(expr.size(),std::size_t{1});
        // Mathematical real arithmetic; both derivatives are constant.
        ARIADNE_TESTING_PROVE(identical(
            simplify(derivative(expr[0],x)),RealExpression(2)));
        ARIADNE_TESTING_PROVE(identical(
            simplify(derivative(expr[0],y)),RealExpression(-3)));
        Map<Identifier,Real> point;
        point[x.name()]=Real(1);
        point[y.name()]=Real(2);
        ARIADNE_TESTING_EXPECT_EQUAL(
            evaluate(expr[0],point),Real(ExactDouble(-3.5)));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neuro_symbolic_expand(model,{x}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neuro_symbolic_expand(model,{x,x}));
    }

    void test_supported_nonlinear_activations() {
        RealVariable x("x");
        for (auto activation : {NeuralActivation::Identity,
                                NeuralActivation::Relu,
                                NeuralActivation::Tanh,
                                NeuralActivation::Sigmoid,
                                NeuralActivation::Softplus}) {
            NeuralParameterSnapshot model(
                NeuralNetwork(1,{{1,activation}}),
                {1.0,0.0});
            auto expressions=neuro_symbolic_expand(model,{x});
            ARIADNE_TESTING_EXPECT_EQUAL(expressions.size(),std::size_t{1});
        }
        NeuralParameterSnapshot smooth(
            NeuralNetwork(1,{{1,NeuralActivation::Tanh}}),
            {1.0,0.0});
        auto const expr=neuro_symbolic_expand(smooth,{x});
        ARIADNE_TESTING_PROVE(
            identical(simplify(derivative(tanh(RealExpression(x)),x)),
                      1-sqr(tanh(RealExpression(x)))));
        ARIADNE_TESTING_PROVE(expr[0].kind()!=OperatorKind::CONSTANT);
    }
};

int main(int argc,char** argv) {
    ARIADNE_TESTING_SUITE("symbolic");
    ARIADNE_TESTING_ADD_TEST(TestSymbolic,test_affine_exact_coefficients_and_derivatives);
    ARIADNE_TESTING_ADD_TEST(TestSymbolic,test_supported_nonlinear_activations);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
