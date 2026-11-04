/***************************************************************************
 *            tests/test_neuro_symbolic.cpp
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

#include "neuro_symbolic/neuro_symbolic.hpp"
#include "ariadne-testing.hpp"

#include <string_view>

class TestNeuroSymbolic : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;
    void test_version() {
        ARIADNE_TESTING_EXPECT_EQUAL(
            std::string_view(Ariadne::neuro_symbolic_version()),
            std::string_view("0.1.0"));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("neuro-symbolic");
    ARIADNE_TESTING_ADD_TEST(TestNeuroSymbolic,test_version);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
