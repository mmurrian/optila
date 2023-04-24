#pragma once

#include <type_traits>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_operation.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_traits_impl.h"

namespace optila {

template <typename ExprType, typename Operation, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Operation, typename... Operands>
class ExpressionImpl<details::matrix_tag, Operation, Operands...>
    : public details::matrix_tag {
  using Expr = Expression<Operation, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;

  using Derived = Expression<Operation, Operands...>;
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

 public:
  constexpr static auto num_rows_compile_time =
      ExprTraits::num_rows_compile_time;
  constexpr static auto num_cols_compile_time =
      ExprTraits::num_cols_compile_time;
  constexpr static auto num_rows_hint = ExprTraits::num_rows_hint;
  constexpr static auto num_cols_hint = ExprTraits::num_cols_hint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return std::apply(ExprTraits::num_rows, derived().operands());
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    return std::apply(ExprTraits::num_cols, derived().operands());
  }
};

// Partial specialization for scalar_tag
template <typename Operation, typename... Operands>
class ExpressionImpl<details::scalar_tag, Operation, Operands...>
    : public details::scalar_tag {};

// Expression inherits from Operation to provide storage for the operation
// in the rare case that it is stateful. Otherwise, empty base optimization
// (EBO) will ensure that the operation does not take up any additional space.
template <typename Operation, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionTraits<Expression<
                                Operation, Operands...>>::expression_type,
                            Operation, Operands...>,
      public Operation {
  using Expr = Expression<Operation, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;

  using operand_storage_type =
      std::tuple<details::safe_type_qualifiers_t<Operands>...>;

 public:
  using value_type = typename ExprTraits::value_type;
  using result_type = typename ExprTraits::result_type;

  constexpr explicit Expression(Operation&& operation, Operands&&... operands)
      : m_operands(std::forward<Operands>(operands)...),
        Operation(std::forward<Operation>(operation)) {
    using ExprTraits = ExpressionTraits<Expr>;
    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      ExprTraits::static_validate();
    } else {
      std::apply(ExprTraits::dynamic_validate, m_operands);
    }
  }

  using operation = Operation;

  // Return the tuple of operands by-value only if it is small and trivial.
  // Otherwise, return by const reference.
  using operand_return_type =
      details::efficient_type_qualifiers_t<operand_storage_type>;
  constexpr operand_return_type operands() const { return m_operands; }

 private:
  operand_storage_type m_operands;
};

// Deduction guide for Expression
template <typename Operation, typename... Operands,
          typename = std::enable_if_t<details::is_operation_v<Operation>>>
Expression(Operation&&, Operands&&...) -> Expression<Operation, Operands...>;

}  // namespace optila