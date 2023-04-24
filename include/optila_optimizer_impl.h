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

template <std::size_t Cost, bool LazyEvaluation, typename... Operands>
struct OptimizedEvaluatorPolicy {
  constexpr static bool lazy_evaluation = LazyEvaluation;
  constexpr static std::size_t cost = Cost + CostHelper<Operands...>::cost;

  constexpr static std::size_t num_operands = sizeof...(Operands);
  template <typename ParentExpr, std::size_t OperandIndex>
  using operand_policy_type =
      std::tuple_element_t<OperandIndex, typename std::tuple<Operands...>>;
};

template <std::size_t CoefficientRatio>
struct OptimizerState {
  constexpr static std::size_t operand_coefficient_ratio = CoefficientRatio;
};

template <typename Expr, typename ParentState = OptimizerState<1>>
constexpr decltype(auto) optimize_expression();

template <typename Policy, typename ParentState>
constexpr std::size_t get_policy_cost();

template <typename Policy, typename ParentState, std::size_t OperandIndex>
constexpr std::size_t get_operand_policy_cost() {
  using ExprTraits = ExpressionTraits<typename Policy::expression>;
  using OperandState =
      OptimizerState<ParentState::operand_coefficient_ratio *
                     std::get<OperandIndex>(
                         ExprTraits::operand_coefficient_ratio)>;
  using OperandPolicy =
      typename Policy::template operand_policy_type<typename Policy::expression,
                                                    OperandIndex>;
  return get_policy_cost<OperandPolicy, OperandState>();
}

template <typename Policy, typename ParentState, std::size_t... Is>
constexpr std::size_t get_policy_cost_helper(std::index_sequence<Is...>) {
  using ExprTraits = ExpressionTraits<typename Policy::expression>;
  std::size_t cost =
      (ExprTraits::operation_counts * ParentState::operand_coefficient_ratio)
          .cost();
  if constexpr (sizeof...(Is) > 0)
    cost += (get_operand_policy_cost<Policy, ParentState, Is>() + ...);
  return cost;
}

template <typename Policy, typename ParentState>
constexpr std::size_t get_policy_cost() {
  return get_policy_cost_helper<Policy, ParentState>(
      std::make_index_sequence<Policy::num_operands>{});
}

template <typename Expr, typename ParentState, std::size_t OperandIndex>
constexpr decltype(auto) optimize_operand() {
  using Operand = std::decay_t<ExpressionOperand_t<Expr, OperandIndex>>;
  using ExprTraits = ExpressionTraits<Expr>;
  constexpr std::size_t operand_coefficient_ratio =
      ParentState::operand_coefficient_ratio *
      std::get<OperandIndex>(ExprTraits::operand_coefficient_ratio);
  return optimize_expression<Operand,
                             OptimizerState<operand_coefficient_ratio>>();
}

template <typename Expr, typename ParentState, std::size_t... Is>
constexpr decltype(auto) optimize_expression_helper(
    std::index_sequence<Is...>) {
  using ExprTraits = ExpressionTraits<Expr>;
  // FIXME: Cost calculation is not yet correct. This needs to be
  // multiplied by the number of coefficients in the expression.
  constexpr std::size_t lazy_cost =
      (ExprTraits::operation_counts * ParentState::operand_coefficient_ratio)
          .cost();
  constexpr auto lazy_path = OptimizedEvaluatorPolicy<
      lazy_cost, true,
      decltype(optimize_operand<Expr, ParentState, Is>())...>{};

  // FIXME: Cost calculation is not yet correct. This needs to be
  // multiplied by the number of coefficients in the expression.
  constexpr std::size_t eager_cost = ExprTraits::operation_counts.cost();
  constexpr auto eager_path = OptimizedEvaluatorPolicy<
      eager_cost, false,
      decltype(optimize_operand<Expr, OptimizerState<1>, Is>())...>{};

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
    return OptimizedEvaluatorPolicy<0, false>{};
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