#include "test/Test.hpp"

#define CATCH_CONFIG_RUNNER
#include "catch_amalgamated.hpp"

#include <array>
#include <numeric>
#include <random>

#define NODE(...) std::make_shared<ASTTNode>(__VA_ARGS__)
#define INTERPRET(...)                                                                             \
    granite::interpret<my::math::Vec3,                                        /**/                 \
                       granite::Space::Space3D,                               /**/                 \
                       granite::Integrator::ClassicRungeKutta,                /**/                 \
                       granite::BorderMode::Block,                            /**/                 \
                       granite::CurvatureMode::Off, granite::AbortMode::Time, /**/                 \
                       granite::UpLiftMode::Off,                              /**/                 \
                       false,                                                 /**/                 \
                       false,                                                 /**/                 \
                       false,                                                 /**/                 \
                       false,                                                 /**/                 \
                       false,                                                 /**/                 \
                       false>(__VA_ARGS__)

#define DEFAULT_ENV                                                                                \
    std::unordered_map<std::string, ASTNodeType> env;                                              \
    env.insert(std::make_pair("f_0", ASTNodeType::Reference_F0));                                  \
    env.insert(std::make_pair("f_1", ASTNodeType::Reference_F1));                                  \
    env.insert(std::make_pair("f_2", ASTNodeType::Reference_F2));                                  \
    env.insert(std::make_pair("f_3", ASTNodeType::Reference_F3));                                  \
    env.insert(std::make_pair("f_4", ASTNodeType::Reference_F4));                                  \
    env.insert(std::make_pair("f_5", ASTNodeType::Reference_F5));                                  \
    env.insert(std::make_pair("f_6", ASTNodeType::Reference_F6));                                  \
    env.insert(std::make_pair("f_7", ASTNodeType::Reference_F7));                                  \
    env.insert(std::make_pair("c_0", ASTNodeType::Reference_C0));                                  \
    env.insert(std::make_pair("c_1", ASTNodeType::Reference_C1));                                  \
    env.insert(std::make_pair("c_2", ASTNodeType::Reference_C2));                                  \
    env.insert(std::make_pair("c_3", ASTNodeType::Reference_C3));

#define NUM_SHUFFLE_TESTS 8

TEST_CASE("Parser")
{
#define PYGRANITE_PARSER_TEST(prog, cmp)                                                           \
    {                                                                                              \
        std::shuffle(ref.begin(), ref.end(), generator);                                           \
        std::stringstream ss;                                                                      \
        ss << parse(prog);                                                                         \
        REQUIRE(ss.str() == cmp);                                                                  \
    }

#define PYGRANITE_PARSER_TEST_FAIL(prog)                                                           \
    {                                                                                              \
        std::shuffle(ref.begin(), ref.end(), generator);                                           \
        REQUIRE_THROWS(parse(prog));                                                               \
    }

    using namespace granite;

    std::array<float, PYGRANITE_LANGUAGE_MAX_REF> ref;
    std::random_device random_device;
    std::mt19937 generator(random_device());

    std::iota(ref.begin(), ref.end(), 1);

    // ! POSITIVE TEST CASES
    PYGRANITE_PARSER_TEST("x / y * z", "((x/y)*z)")
    PYGRANITE_PARSER_TEST("x / (y * z)", "(x/(y*z))")
    PYGRANITE_PARSER_TEST("x + y * -1.0", "(x+(y*-1))")
    PYGRANITE_PARSER_TEST("x * y * -1.0", "((x*y)*-1)")
    PYGRANITE_PARSER_TEST("atan2(36.0 * 3.141 / 7, 9)", "atan2(((36*3.141)/7), 9)")
    PYGRANITE_PARSER_TEST("f_2 / (0.622 * 6.112 * exp(17.67 * f_1 / (243.5 + f_1))) * c_0",
                          "((f_2/((0.622*6.112)*exp(((17.67*f_1)/(243.5+f_1)))))*c_0)")
    PYGRANITE_PARSER_TEST("abc", "abc")
    PYGRANITE_PARSER_TEST("Abc", "Abc")
    PYGRANITE_PARSER_TEST("A0bc", "A0bc")
    PYGRANITE_PARSER_TEST("_abc", "_abc")

    // ! NEGATIVE TEST CASES
    PYGRANITE_PARSER_TEST_FAIL("f6 + b")
    PYGRANITE_PARSER_TEST_FAIL("a + f0")
    PYGRANITE_PARSER_TEST_FAIL("c0 + f0")
    PYGRANITE_PARSER_TEST_FAIL("16 * (f7 + 2")
    PYGRANITE_PARSER_TEST_FAIL("0a1 + 2")
    PYGRANITE_PARSER_TEST_FAIL("0_a1 + 2")

#undef PYGRANITE_PARSER_TEST
#undef PYGRANITE_PARSER_TEST_FAIL
}

TEST_CASE("Interpreter")
{
#define PYGRANITE_INTERPRET_TEST(code, res)                                                        \
    {                                                                                              \
        auto program = parse(code);                                                                \
        auto flat_program = flatten(replace_environment(program, env));                            \
        REQUIRE(granite::interpret(program, ref.data()) == res);                                   \
        REQUIRE(INTERPRET(flat_program.data(), flat_program.size(), ref.data(), size_t(0)) ==      \
                res);                                                                              \
    }

    using namespace granite;

    DEFAULT_ENV

    std::array<float, PYGRANITE_LANGUAGE_MAX_REF> ref;
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::iota(ref.begin(), ref.end(), 1);

    // ! TEST CASES
    // Values:
    // x = 1 ; y = 2 ; z = 3 ; cv = 4 ; f0 = 5 ; ... ; f7 = 12 ; c0 = 13 ; ... ; c3 = 16
    PYGRANITE_INTERPRET_TEST("x + y", 3.f)
    PYGRANITE_INTERPRET_TEST("z * y", 6.f)
    PYGRANITE_INTERPRET_TEST("x / y", 0.5f)
    PYGRANITE_INTERPRET_TEST("x - y", -1.0f)
    PYGRANITE_INTERPRET_TEST("x - y * -1", 3.0f)
    PYGRANITE_INTERPRET_TEST("ln(x) * (1 - 12 * exp(23) + f_7)", 0.f)
    PYGRANITE_INTERPRET_TEST("f_7 * 2 + x", 25.f)
    PYGRANITE_INTERPRET_TEST("c_3 * exp(0)", 16)
    PYGRANITE_INTERPRET_TEST("u + v", 17 + 18)

#undef PYGRANITE_INTERPRET_TEST
}

TEST_CASE("Flatten")
{
#define PYGRANITE_FLATTEN_TEST(code)                                                               \
    {                                                                                              \
        auto program = parse(code);                                                                \
        auto flat_program = flatten(replace_environment(program, env));                            \
        for (size_t i = 0; i < NUM_SHUFFLE_TESTS; ++i)                                             \
        {                                                                                          \
            std::shuffle(ref.begin(), ref.end(), generator);                                       \
            REQUIRE(granite::interpret(program, ref.data()) ==                                     \
                    INTERPRET(flat_program.data(), flat_program.size(), ref.data(), size_t(0)));   \
        }                                                                                          \
    }

    using namespace granite;

    DEFAULT_ENV

    std::array<float, PYGRANITE_LANGUAGE_MAX_REF> ref;
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::iota(ref.begin(), ref.end(), 1);

    // ! TEST CASES
    PYGRANITE_FLATTEN_TEST("x + y * f_0 + exp(9 * f_4)")
    PYGRANITE_FLATTEN_TEST("x * sqrt(y) * f_2 + ln(9.2 * c_0)")
    PYGRANITE_FLATTEN_TEST("x + z * f_7 + exp(9 * c_3)")

#undef PYGRANITE_FLATTEN_TEST
}

void test::test_run()
{
    const char * argv[]{"./test"};
    int result = Catch::Session().run(1, argv);
    std::cout << "CATCH2 < EXIT CODE | " << result << " >" << std::endl;
}