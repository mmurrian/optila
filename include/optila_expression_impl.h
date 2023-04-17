#pragma once

#include "details/optila_expression.h"
#include "optila_expression_validator_impl.h"
#include "optila_operation_impl.h"
#include "optila_result_helpers.h"

namespace optila {

template <typename ExprType, typename Op, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::matrix_tag, Op, Operands...>
    : public details::matrix_tag {
  using Derived = Expression<Op, Operands...>;
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }

 public:
  using state_type = typename Operation::BuildState<Derived>::type;
  constexpr decltype(auto) operator()(std::size_t i, std::size_t j,
                                      const state_type& state) const {
    return Op::apply_matrix(i, j, derived(), state);
  }

  static constexpr std::size_t num_rows_static() {
    return ResultNumRows<Op, Operands...>::value;
  }
  static constexpr std::size_t num_cols_static() {
    return ResultNumCols<Op, Operands...>::value;
  }
};

// Partial specialization for scalar_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::scalar_tag, Op, Operands...>
    : public details::scalar_tag {
  using Derived = Expression<Op, Operands...>;
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }

 public:
  using state_type = typename Operation::BuildState<Derived>::type;
  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(static_cast<const Derived&>(*this), state);
  }
};

template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionValidator<
                                Op, std::decay_t<Operands>...>::expression_type,
                            Op, Operands...> {
  // The Expression class forwards the deduced ExprType to ExpressionImpl.
 public:
  using value_type = typename ResultValueType<Op, Operands...>::type;
  using operation_type = Op;
  template <std::size_t index>
  using operand_type =
      std::tuple_element_t<index, std::tuple<std::decay_t<Operands>...>>;

  constexpr explicit Expression(Operands&&... operands)
      : operands_(std::forward<Operands>(operands)...) {}

  template <std::size_t index>
  constexpr const auto& operand() const {
    return std::get<index>(operands_);
  }
  [[nodiscard]] static constexpr std::size_t num_operands() {
    return std::tuple_size_v<operand_storage_type>;
  }

 private:
  using operand_storage_type =
      std::tuple<details::store_by_value_or_const_ref_t<Operands>...>;
  operand_storage_type operands_;
};

}  // namespace optila