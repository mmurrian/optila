#pragma once

#include "details/optila_expression.h"
#include "optila_operation_impl.h"

namespace optila {

template <typename Expr, typename State>
constexpr void precompute(const Expr& expr, State& state) {
  static_assert(details::is_expression_literal_v<Expr>,
                "Expression literal expected");
  if constexpr (Expr::num_operands() > 0)
    if constexpr (details::is_operand_expression_literal_v<0, Expr>)
      precompute(expr.template operand<0>(), std::get<0>(state.operands));
  if constexpr (Expr::num_operands() > 1)
    if constexpr (details::is_operand_expression_literal_v<1, Expr>)
      precompute(expr.template operand<1>(), std::get<1>(state.operands));
  static_assert(Expr::num_operands() <= 2,
                "Only expressions with up to 2 operands are supported");
  if constexpr (details::has_operation_state_v<State>)
    Expr::operation_type::precompute(expr, state);
}

template <typename Expr, StorageOrder Order = StorageOrder::RowMajor,
          typename = std::enable_if_t<details::is_matrix_v<Expr>>>
constexpr auto evaluate(const Expr& expr) {
  using StateType = typename Operation::BuildState<std::decay_t<Expr>>::type;
  StateType state{};
  precompute(expr, state);

  Matrix<typename Expr::value_type, Expr::num_rows_static(),
         Expr::num_cols_static(), Order>
      result{};
  if constexpr (details::is_dynamic_expression_v<Expr>) {
    result.resize(expr.num_rows(), expr.num_cols());
  }
  for (std::size_t i = 0; i < result.num_rows(); ++i) {
    for (std::size_t j = 0; j < result.num_cols(); ++j) {
      result(i, j) = expr(i, j, state);
    }
  }
  return result;
}

template <typename Expr,
          typename = std::enable_if_t<details::is_scalar_v<Expr>>>
constexpr decltype(auto) evaluate(const Expr& expr) {
  using StateType = typename Operation::BuildState<std::decay_t<Expr>>::type;
  StateType state{};
  precompute(expr, state);

  return expr(state);
}

}  // namespace optila