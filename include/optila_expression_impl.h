#pragma once

#include <type_traits>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_traits_impl.h"
#include "optila_scalar_impl.h"

namespace optila {

template <typename ExprType, typename Op, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::matrix_tag, Op, Operands...>
    : public details::matrix_tag {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

  using Derived = Expression<Op, Operands...>;
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

 public:
  using value_type = typename ExprTraits::value_type;
  using result_type =
      Matrix<value_type, ExprTraits::num_rows_compile_time,
             ExprTraits::num_cols_compile_time, DefaultMatrixPolicy>;
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
    : public details::scalar_tag {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

 public:
  using value_type = typename ExprTraits::value_type;
  using result_type = Scalar<value_type>;
};

template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionTraits<
                                Op, std::decay_t<Operands>...>::expression_type,
                            Op, Operands...> {
  using Base = ExpressionImpl<
      typename ExpressionTraits<Op, std::decay_t<Operands>...>::expression_type,
      Op, Operands...>;

  using operand_storage_type =
      std::tuple<details::safe_type_qualifiers_t<Operands>...>;

 public:
  using value_type = typename Base::value_type;
  using result_type = typename Base::result_type;

  constexpr explicit Expression(Operands&&... operands)
      : m_operands(std::forward<Operands>(operands)...) {
    using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;
    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      ExprTraits::static_validate();
    } else {
      std::apply(ExprTraits::dynamic_validate, m_operands);
    }
  }

  using operation = Op;

  // Return the tuple of operands by-value only if it is small and trivial.
  // Otherwise, return by const reference.
  using operand_return_type =
      details::efficient_type_qualifiers_t<operand_storage_type>;
  constexpr operand_return_type operands() const { return m_operands; }

 private:
  operand_storage_type m_operands;
};

}  // namespace optila