#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "my/util/Util.cuh"

namespace granite
{

/**
 * @brief enum representing the different possible ast node types
 */
enum class ASTNodeType : uint32_t
{
    // empty node or not existing node
    None = 0,

    // particle position
    Reference_X = 1,
    Reference_Y = 2,
    Reference_Z = 3,

    // particle position
    Reference_U = 33,
    Reference_V = 34,
    Reference_W = 35,

    // curvature at that position
    Reference_CV = 4,

    // the 8 additional volume values
    Reference_F0 = 5,
    Reference_F1 = 6,
    Reference_F2 = 7,
    Reference_F3 = 8,
    Reference_F4 = 9,
    Reference_F5 = 10,
    Reference_F6 = 11,
    Reference_F7 = 12,

    Reference_C0 = 13,
    Reference_C1 = 14,
    Reference_C2 = 15,
    Reference_C3 = 16,

    // constant value
    Constant = 17,

    // binary operations
    Addition = 18,
    Multiplication = 19,
    Substraction = 20,
    Division = 21,

    // unitary operations (such as build in functions)
    Operation_SQRT = 22,
    Operation_EXP = 23,
    Operation_LN = 24,
    Operation_SINE = 25,
    Operation_COSINE = 26,
    Operation_TANGENS = 27,
    Operation_ARCSINE = 28,
    Operation_ARCCOSINE = 29,
    Operation_ARCTANGENS = 30,
    Operation_ARCTANGENS_2 = 31,

    // ONLY on host ! (needs to be compiled away)
    Identifier = 32
};

constexpr size_t PYGRANITE_LANGUAGE_MAX_REF = 19;

/**
 * @brief struct representing an inline AST node.
 */
struct ASTANode
{
    ASTNodeType Type;
    float Value{0.f};
    int32_t Left{-1};
    int32_t Right{-1};
};

/**
 * @brief struct representing an actual ast node
 */
struct ASTTNode
{
    ASTNodeType Type;
    float Value{0.f};
    std::string Id{""};
    std::shared_ptr<ASTTNode> Left{nullptr};
    std::shared_ptr<ASTTNode> Right{nullptr};

    ASTTNode(ASTNodeType Type, float Value, std::shared_ptr<ASTTNode> Left,
             std::shared_ptr<ASTTNode> Right)
    {
        this->Type = Type;
        this->Value = Value;
        this->Left = Left;
        this->Right = Right;
    }

    ASTTNode(ASTNodeType Type, std::shared_ptr<ASTTNode> Left, std::shared_ptr<ASTTNode> Right)
    {
        this->Type = Type;
        this->Value = 0.f;
        this->Left = Left;
        this->Right = Right;
    }

    ASTTNode(float Value)
    {
        this->Type = ASTNodeType::Constant;
        this->Value = Value;
        this->Left = nullptr;
        this->Right = nullptr;
    }

    ASTTNode(ASTNodeType Type)
    {
        this->Type = Type;
        this->Value = 0.f;
        this->Left = nullptr;
        this->Right = nullptr;
    }
    
    ASTTNode(std::string Id)
    {
        this->Id = Id;
        this->Type = ASTNodeType::Identifier;
        this->Value = 0.f;
        this->Left = nullptr;
        this->Right = nullptr;
    }
};

} // namespace granite