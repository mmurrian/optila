#pragma once

#include <cmath>
#include <tuple>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"

namespace optila::Operation {

template <std::size_t Index, typename Expr>
constexpr decltype(auto) evalScalarOperand(Expr&& expr) {
  const auto& operand = expr.template operand<Index>();
  return operand();
}

template <std::size_t Index, typename Expr>
constexpr decltype(auto) evalMatrixOperand(std::size_t i, std::size_t j,
                                           Expr&& expr) {
  const auto& operand = expr.template operand<Index>();
  return operand(i, j);
}

struct ScalarAddition {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return evalScalarOperand<0>(expr) + evalScalarOperand<1>(expr);
  };
};

struct Addition {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return evalMatrixOperand<0>(i, j, expr) + evalMatrixOperand<1>(i, j, expr);
  };
};
struct ScalarSubtraction {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return evalScalarOperand<0>(expr) - evalScalarOperand<1>(expr);
  };
};
struct Subtraction {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return evalMatrixOperand<0>(i, j, expr) - evalMatrixOperand<1>(i, j, expr);
  };
};
// Matrix multiplication
struct Multiplication {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    if constexpr (details::is_matrix_v<decltype(expr.template operand<0>())> &&
                  details::is_matrix_v<decltype(expr.template operand<1>())>) {
      for (std::size_t k = 0; k < expr.template operand<0>().num_cols(); ++k) {
        result +=
            evalMatrixOperand<0>(i, k, expr) * evalMatrixOperand<1>(k, j, expr);
      }
    } else if constexpr (details::is_matrix_v<
                             decltype(expr.template operand<0>())>) {
      result = evalMatrixOperand<0>(i, j, expr) * evalScalarOperand<1>(expr);
    } else {  // details::is_matrix_v<decltype(expr.template operand<1>())>
      result = evalScalarOperand<0>(expr) * evalMatrixOperand<1>(i, j, expr);
    }
    return result;
  };
};
struct ScalarMultiplication {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return evalScalarOperand<0>(expr) * evalScalarOperand<1>(expr);
  };
};
struct MatrixScalarDivision {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return evalMatrixOperand<0>(i, j, expr) / evalScalarOperand<1>(expr);
  };
};
struct ScalarDivision {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return evalScalarOperand<0>(expr) / evalScalarOperand<1>(expr);
  };
};
struct DotProduct {
  static constexpr auto apply_scalar = [](auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    for (std::size_t i = 0; i < expr.template operand<0>().num_rows(); ++i) {
      result +=
          evalMatrixOperand<0>(i, 0, expr) * evalMatrixOperand<1>(i, 0, expr);
    }
    return result;
  };
};
struct CrossProduct {};
struct OuterProduct {};
struct Transpose {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return evalMatrixOperand<0>(j, i, expr);
  };
};

struct Determinant {};
struct Trace {};
struct Inverse {};
struct Adjoint {};
struct Cofactor {};
struct Rank {};
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols>
struct SubmatrixExtraction {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return evalMatrixOperand<0>(i + StartRow, j + StartCol, expr);
  };
};
struct Concatenation {};
struct SquareRoot {
  // FIXME: sqrt is not constexpr
  template <typename Expr>
  static auto apply_scalar(Expr&& expr) {
    using std::sqrt;
    return sqrt(evalScalarOperand<0>(expr));
  }
};
struct ElementWiseOperation {};

struct StrictEquality {
  static constexpr auto apply_scalar = [](auto&& expr) {
    using Lhs = std::decay_t<decltype(expr.template operand<0>())>;
    using Rhs = std::decay_t<decltype(expr.template operand<1>())>;
    if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
      for (std::size_t i = 0; i < expr.template operand<0>().num_rows(); ++i) {
        for (std::size_t j = 0; j < expr.template operand<0>().num_cols();
             ++j) {
          if (evalMatrixOperand<0>(i, j, expr) !=
              evalMatrixOperand<1>(i, j, expr)) {
            return false;
          }
        }
      }
      return true;
    } else {
      return evalScalarOperand<0>(expr) == evalScalarOperand<1>(expr);
    }
  };
};

template <typename FromType, typename ToType>
struct StaticConversion {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return static_cast<ToType>(evalScalarOperand<0>(expr));
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return static_cast<ToType>(evalMatrixOperand<0>(i, j, expr));
  };
};

// Fill a matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantMatrix {
  static constexpr auto apply_matrix = [](std::size_t /*i*/, std::size_t /*j*/,
                                          auto&& expr) {
    return expr.template operand<0>()();
  };
};
// Fill a diagonal matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantDiagonal {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.template operand<0>()() : value_type{};
  };
};
// Put a vector on the diagonal of a matrix
struct DiagonalFromVector {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.template operand<0>()(i, 0) : value_type{};
  };
};
// Extract the diagonal of a matrix into a vector
struct DiagonalToVector {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return expr.template operand<0>()(i, i);
  };
};
// Extract the diagonal of a matrix into a diagonal matrix
struct DiagonalMatrix {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.template operand<0>()(i, i) : value_type{};
  };
};

struct Evaluate {
  static constexpr auto apply_scalar = [](auto&& expr) {
    return expr.template operand<0>()();
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    return expr.template operand<0>()(i, j);
  };
};

}  // namespace optila::Operation