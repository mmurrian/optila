#pragma once

#include <tuple>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_operation.h"

namespace optila::Operation {

// Operation state template structure, specialized for operations with and
// without state
template <typename Op, bool HasState, typename... States>
struct StateImpl;

template <typename Op, typename... States>
struct StateImpl<Op, true, States...> : public details::operation_state_tag {
  using operation_type = Op;
  typename operation_type::State state;

  std::tuple<States...> operands;
};

template <typename Op, typename... States>
struct StateImpl<Op, false, States...> {
  using operation_type = Op;

  std::tuple<States...> operands;
};

template <typename Op, typename... States>
struct State
    : public StateImpl<Op, details::has_operation_state_v<Op>, States...> {};

// Utility function to create the operation state structure
template <typename Op, typename... Operands>
struct BuildState {
  struct type {};
};

template <typename Op, typename... Operands>
struct BuildState<Expression<Op, Operands...>> {
  using type = State<Op, typename BuildState<std::decay_t<Operands>>::type...>;
};

template <typename T>
struct is_operand_state : std::false_type {};

template <typename Op, typename... States>
struct is_operand_state<State<Op, States...>> : std::true_type {};

template <typename T>
inline constexpr bool is_operand_state_v = is_operand_state<T>::value;

template <std::size_t Index, typename Expr, typename State>
constexpr decltype(auto) evalScalarOperand(Expr&& expr, State&& state) {
  const auto& operand = expr.template operand<Index>();
  const auto& operand_state = std::get<Index>(state.operands);
  if constexpr (is_operand_state_v<std::decay_t<decltype(operand_state)>>) {
    return operand(operand_state);
  } else {
    return operand();
  }
}

template <std::size_t Index, typename Expr, typename State>
constexpr decltype(auto) evalMatrixOperand(std::size_t i, std::size_t j,
                                           Expr&& expr, State&& state) {
  const auto& operand = expr.template operand<Index>();
  const auto& operand_state = std::get<Index>(state.operands);
  if constexpr (is_operand_state_v<std::decay_t<decltype(operand_state)>>) {
    return operand(i, j, operand_state);
  } else {
    return operand(i, j);
  }
}

struct ScalarAddition {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalScalarOperand<0>(expr, state) +
           evalScalarOperand<1>(expr, state);
  };
};

struct Addition {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand<0>(i, j, expr, state) +
           evalMatrixOperand<1>(i, j, expr, state);
  };
};
struct ScalarSubtraction {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalScalarOperand<0>(expr, state) -
           evalScalarOperand<1>(expr, state);
  };
};
struct Subtraction {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand<0>(i, j, expr, state) -
           evalMatrixOperand<1>(i, j, expr, state);
  };
};
// Matrix multiplication
struct Multiplication {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    if constexpr (details::is_matrix_v<decltype(expr.template operand<0>())> &&
                  details::is_matrix_v<decltype(expr.template operand<1>())>) {
      for (std::size_t k = 0; k < expr.template operand<0>().num_cols(); ++k) {
        result += evalMatrixOperand<0>(i, k, expr, state) *
                  evalMatrixOperand<1>(k, j, expr, state);
      }
    } else if constexpr (details::is_matrix_v<
                             decltype(expr.template operand<0>())>) {
      result = evalMatrixOperand<0>(i, j, expr, state) *
               evalScalarOperand<1>(expr, state);
    } else {  // details::is_matrix_v<decltype(expr.template operand<1>())>
      result = evalScalarOperand<0>(expr, state) *
               evalMatrixOperand<1>(i, j, expr, state);
    }
    return result;
  };
};
struct ScalarMultiplication {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalScalarOperand<0>(expr, state) *
           evalScalarOperand<1>(expr, state);
  };
};
struct ScalarDivision {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalScalarOperand<0>(expr, state) /
           evalScalarOperand<1>(expr, state);
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    constexpr bool rhs_is_scalar =
        details::is_scalar_v<decltype(expr.template operand<1>())>;
    static_assert(rhs_is_scalar, "The divisor must be a scalar");

    return evalMatrixOperand<0>(i, j, expr, state) /
           evalScalarOperand<1>(expr, state);
  };
};
struct DotProduct {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    for (std::size_t i = 0; i < expr.template operand<0>().num_rows(); ++i) {
      result += evalMatrixOperand<0>(i, 0, expr, state) *
                evalMatrixOperand<1>(i, 0, expr, state);
    }
    return result;
  };
};
struct CrossProduct {};
struct OuterProduct {};
struct Transpose {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand<0>(j, i, expr, state);
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
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand<0>(i + StartRow, j + StartCol, expr, state);
  };
};
struct Concatenation {};
struct SquareRoot {
  // FIXME: sqrt is not constexpr
  template <typename Expr, typename State>
  static auto apply_scalar(Expr&& expr, State&& state) {
    return sqrt(expr.template operand<0>()(std::get<0>(state.operands)));
  }
};
struct Norm {};
struct Normalization : public details::operation_state_tag {
  struct State {
    double norm;
  };

  template <typename Expr, typename State>
  static constexpr void precompute(Expr&& expr, State&& state) {
    // TODO: Ideally, this could be implemented after Expression and 'evaluate'
    // so that we can use the Norm operation on expr.template operand<0>().
    state.state.norm = 0;
    for (std::size_t i = 0; i < expr.num_rows(); ++i) {
      for (std::size_t j = 0; j < expr.num_cols(); ++j) {
        const auto& value = evalMatrixOperand<0>(i, j, expr, state);
        state.state.norm += value * value;
      }
    }
    state.state.norm = sqrt(state.state.norm);
  }

  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand<0>(i, j, expr, state) / state.state.norm;
  };
};
struct ElementWiseOperation {};

struct StrictEquality {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    using Lhs = std::decay_t<decltype(expr.template operand<0>())>;
    using Rhs = std::decay_t<decltype(expr.template operand<1>())>;
    if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
      for (std::size_t i = 0; i < expr.template operand<0>().num_rows(); ++i) {
        for (std::size_t j = 0; j < expr.template operand<0>().num_cols();
             ++j) {
          if (evalMatrixOperand<0>(i, j, expr, state) !=
              evalMatrixOperand<1>(i, j, expr, state)) {
            return false;
          }
        }
      }
      return true;
    } else {
      return evalScalarOperand<0>(expr, state) ==
             evalScalarOperand<1>(expr, state);
    }
  };
};

template <typename FromType, typename ToType>
struct StaticConversion {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return static_cast<ToType>(evalScalarOperand<0>(expr, state));
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return static_cast<ToType>(evalMatrixOperand<0>(i, j, expr, state));
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

}  // namespace optila::Operation