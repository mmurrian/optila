// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_operation.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_traits_impl.h"

namespace optila {

template <typename ExprType, typename Op, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::matrix_tag, Op, Operands...>
    : public details::matrix_tag {
  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;

  using Derived = Expression<Op, Operands...>;
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
template <typename Op, typename... Operands>
class ExpressionImpl<details::scalar_tag, Op, Operands...>
    : public details::scalar_tag {};

// Expression inherits from Op to provide storage for the operation
// in the rare case that it is stateful. Otherwise, empty base optimization
// (EBO) will ensure that the operation does not take up any additional space.
template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionTraits<
                                Expression<Op, Operands...>>::expression_type,
                            Op, Operands...>,
      public Op {
  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;

  using operand_storage_type =
      std::tuple<details::safe_type_qualifiers_t<Operands>...>;
  using operand_return_type =
      details::efficient_type_qualifiers_t<operand_storage_type>;
  using operation_return_type = details::efficient_type_qualifiers_t<Op>;

 public:
  using value_type = typename ExprTraits::value_type;
  using result_type = typename ExprTraits::result_type;

  constexpr explicit Expression(Op&& operation, Operands&&... operands)
      : m_operands(std::forward<Operands>(operands)...),
        Op(std::forward<Op>(operation)) {
    using ExprTraits = ExpressionTraits<Expr>;
    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      ExprTraits::static_validate();
    } else {
      std::apply(ExprTraits::dynamic_validate, m_operands);
    }
  }

  constexpr operation_return_type operation() const { return *this; }
  constexpr operand_return_type operands() const { return m_operands; }

 private:
  operand_storage_type m_operands;
};

// Deduction guide for Expression
template <typename Op, typename... Operands,
          typename = std::enable_if_t<details::is_operation_v<Op>>>
Expression(Op&&, Operands&&...) -> Expression<Op, Operands...>;

}  // namespace optila