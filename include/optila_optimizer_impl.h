// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <ratio>
#include <type_traits>

#include "optila_expression_traits_impl.h"

namespace optila {

namespace details {

template <typename... Operands>
struct CostHelper;

template <typename Head, typename... Tail>
struct CostHelper<Head, Tail...> {
  constexpr static std::size_t cost = Head::cost + CostHelper<Tail...>::cost;
};

template <>
struct CostHelper<> {
  constexpr static std::size_t cost = 0;
};

template <std::size_t Cost, bool LazyEvaluation, typename ActiveSubMatrix,
          typename... Operands>
struct OptimizedEvaluatorPolicy {
  constexpr static bool lazy_evaluation = LazyEvaluation;
  using active_sub_matrix = ActiveSubMatrix;
  constexpr static std::size_t cost = Cost + CostHelper<Operands...>::cost;

  constexpr static std::size_t num_operands = sizeof...(Operands);
  template <typename ParentExpr, std::size_t OperandIndex>
  using operand_policy_type =
      std::tuple_element_t<OperandIndex, typename std::tuple<Operands...>>;
};

template <std::size_t CoefficientRatio = 1,
          typename ActiveSubMatrix = MatrixBounds<0, 0, Dynamic, Dynamic>>
struct OptimizerState {
  constexpr static std::size_t operand_coefficient_ratio = CoefficientRatio;
  using active_sub_matrix = ActiveSubMatrix;
};

template <typename Expr, typename ParentState = OptimizerState<>>
constexpr decltype(auto) optimize_expression();

template <typename Expr, typename ParentState, std::size_t OperandIndex>
constexpr decltype(auto) optimize_operand() {
  using Operand = std::decay_t<ExpressionOperand_t<Expr, OperandIndex>>;
  using ExprTraits = ExpressionTraits<Expr>;
  constexpr std::size_t operand_coefficient_ratio =
      ParentState::operand_coefficient_ratio *
      std::get<OperandIndex>(ExprTraits::operand_coefficient_ratio);
  using operand_active_sub_matrix = std::tuple_element_t<
      OperandIndex, typename ExprTraits::template operand_active_sub_matrix<
                        typename ParentState::active_sub_matrix>>;
  return optimize_expression<
      Operand,
      OptimizerState<operand_coefficient_ratio, operand_active_sub_matrix>>();
}

template <typename Expr, typename ActiveSubMatrix>
constexpr std::size_t coefficient_count_hint() {
  using ExprTraits = ExpressionTraits<Expr>;
  if constexpr (std::is_same_v<typename ExprTraits::expression_type,
                               details::matrix_tag>) {
    // If the active submatrix is statically sized, we can calculate the
    // coefficient count at compile time. Otherwise, we have to use the
    // size hint from the expression.
    using active_bounds =
        MatrixBounds<0, 0, ActiveSubMatrix::num_cols_compile_time,
                     ActiveSubMatrix::num_cols_compile_time>;
    if constexpr (active_bounds::is_static()) {
      return active_bounds::num_rows_compile_time *
             active_bounds::num_cols_compile_time;
    } else {
      return ExprTraits::num_rows_hint * ExprTraits::num_cols_hint;
    }
  } else {
    return 1;
  }
}

template <typename Expr, typename ParentState, std::size_t... Is>
constexpr decltype(auto) optimize_expression_helper(
    std::index_sequence<Is...>) {
  using ExprTraits = ExpressionTraits<Expr>;

  constexpr std::size_t lazy_cost =
      (ExprTraits::operation_counts * ParentState::operand_coefficient_ratio *
       coefficient_count_hint<Expr, typename ParentState::active_sub_matrix>())
          .cost();
  constexpr auto lazy_path = OptimizedEvaluatorPolicy<
      lazy_cost, true, typename ParentState::active_sub_matrix,
      decltype(optimize_operand<Expr, ParentState, Is>())...>{};

  constexpr std::size_t eager_cost =
      (ExprTraits::operation_counts *
       coefficient_count_hint<Expr, typename ParentState::active_sub_matrix>())
          .cost();
  constexpr auto eager_path = OptimizedEvaluatorPolicy<
      eager_cost, false, typename ParentState::active_sub_matrix,
      decltype(optimize_operand<
               Expr, OptimizerState<1, typename ParentState::active_sub_matrix>,
               Is>())...>{};

  if constexpr (decltype(lazy_path)::cost > decltype(eager_path)::cost) {
    return eager_path;
  } else {
    return lazy_path;
  }
}

template <typename Expr, typename ParentState>
constexpr decltype(auto) optimize_expression() {
  static_assert(details::is_matrix_literal_v<Expr> ||
                details::is_scalar_literal_v<Expr> ||
                details::is_expression_literal_v<Expr>);
  if constexpr (details::is_matrix_literal_v<Expr> ||
                details::is_scalar_literal_v<Expr>) {
    return OptimizedEvaluatorPolicy<0, false,
                                    typename ParentState::active_sub_matrix>{};
  } else if constexpr (details::is_expression_literal_v<Expr>) {
    using ExprTraits = ExpressionTraits<Expr>;
    return optimize_expression_helper<Expr, ParentState>(
        std::make_index_sequence<std::tuple_size_v<
            decltype(ExprTraits::operand_coefficient_ratio)>>{});
  }
}

}  // namespace details

template <typename Expr>
using optimize_expression_t =
    decltype(details::optimize_expression<std::decay_t<Expr>>());

}  // namespace optila