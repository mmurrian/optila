#pragma once

#include <ratio>
#include <tuple>

#include "optila_expression_traits_impl.h"

namespace optila {
template <std::size_t CoefficientRatio>
struct EvaluatorState {
  constexpr static std::size_t operand_coefficient_ratio = CoefficientRatio;
};

struct RootEvaluatorTag {};

template <typename ParentState = EvaluatorState<1>,
          typename ParentExpression = RootEvaluatorTag,
          std::size_t OperandIndex = 0>
struct DefaultEvaluatorPolicy;

template <typename ParentState, typename ParentExpression,
          std::size_t OperandIndex>
struct DefaultEvaluatorPolicy {
 private:
  using ExprTraits = ExpressionTraits<ParentExpression>;

 public:
  constexpr static std::size_t operand_coefficient_ratio =
      ParentState::operand_coefficient_ratio *
      std::get<OperandIndex>(ExprTraits::operand_coefficient_ratio);

  // TODO: Implement a more thoughtful lazy evaluation policy.
  constexpr static bool lazy_evaluation = operand_coefficient_ratio <= 1;

  // Provides the type for the operand policy of each operand for the current
  // Expression. Here, ParentExpression_ is the parent expression relative to
  // the operand. That is, it is the current Expression of the Evaluator that
  // has this EvaluatorPolicy.
  template <typename ParentExpression_, std::size_t OperandIndex_>
  using operand_policy_type =
      DefaultEvaluatorPolicy<EvaluatorState<operand_coefficient_ratio>,
                             ParentExpression_, OperandIndex_>;
};

template <typename RootState>
struct DefaultEvaluatorPolicy<RootState, RootEvaluatorTag, 0> {
 public:
  constexpr static std::size_t operand_coefficient_ratio =
      RootState::operand_coefficient_ratio;

  constexpr static bool lazy_evaluation = true;

  template <typename ParentExpr_, std::size_t OperandIndex_>
  using operand_policy_type =
      DefaultEvaluatorPolicy<RootState, ParentExpr_, OperandIndex_>;
};

struct LazyEvaluatorPolicy {
 public:
  constexpr static bool lazy_evaluation = true;

  template <typename ParentExpr_, std::size_t OperandIndex_>
  using operand_policy_type = LazyEvaluatorPolicy;
};

}  // namespace optila