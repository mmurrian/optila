#pragma once

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_validator_impl.h"
#include "optila_operation_impl.h"

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
  using value_type =
      typename ExpressionValidator<Op, std::decay_t<Operands>...>::value_type;
  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return Op::apply_matrix(i, j, derived());
  }

  static constexpr std::size_t num_rows_static() {
    return ExpressionValidator<Op,
                               std::decay_t<Operands>...>::num_rows_static();
  }
  static constexpr std::size_t num_cols_static() {
    return ExpressionValidator<Op,
                               std::decay_t<Operands>...>::num_cols_static();
  }
  [[nodiscard]] constexpr std::size_t num_rows() const {
    return std::apply(
        ExpressionValidator<Op, std::decay_t<Operands>...>::num_rows,
        derived().operands());
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    return std::apply(
        ExpressionValidator<Op, std::decay_t<Operands>...>::num_cols,
        derived().operands());
  }

  template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
            StorageOrder Order = StorageOrder::RowMajor>
  constexpr operator Matrix<ValueType, NumRows, NumCols, Order>() const {
    static_assert(details::is_matrix_v<Derived>, "Expression must be a matrix");
    static_assert(std::is_same_v<details::result_type_t<ValueType, value_type>,
                                 ValueType>,
                  "Incompatible value types");
    static_assert(
        Derived::num_rows_static() == NumRows || NumRows == Dynamic ||
            Derived::num_rows_static() == Dynamic,
        "Static row count mismatch between expression and result matrix");
    static_assert(
        Derived::num_cols_static() == NumCols || NumCols == Dynamic ||
            Derived::num_cols_static() == Dynamic,
        "Static column count mismatch between expression and result matrix");

    using ResultType = Matrix<value_type, NumRows, NumCols, Order>;
    ResultType result{};
    if constexpr (details::is_dynamic_expression_v<ResultType>) {
      result.resize(derived().num_rows(), derived().num_cols());
    } else if constexpr (details::is_dynamic_expression_v<Derived>) {
      assert(NumRows == derived().num_rows() &&
             NumCols == derived().num_cols());
    }
    for (std::size_t i = 0; i < result.num_rows(); ++i) {
      for (std::size_t j = 0; j < result.num_cols(); ++j) {
        result(i, j) = derived()(i, j);
      }
    }
    return result;
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
  using value_type =
      typename ExpressionValidator<Op, std::decay_t<Operands>...>::value_type;
  constexpr decltype(auto) operator()() const {
    return Op::apply_scalar(derived());
  }

  template <typename ValueType>
  constexpr operator Scalar<ValueType>() const {
    static_assert(details::is_scalar_v<Derived>, "Expression must be a scalar");
    static_assert(std::is_same_v<details::result_type_t<ValueType, value_type>,
                                 ValueType>,
                  "Incompatible value types");

    return Scalar<ValueType>(derived()());
  }
};

template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionValidator<
                                Op, std::decay_t<Operands>...>::expression_type,
                            Op, Operands...> {
  // The Expression class forwards the deduced ExprType to ExpressionImpl.
 public:
  using value_type =
      typename ExpressionValidator<Op, std::decay_t<Operands>...>::value_type;
  using operation_type = Op;
  template <std::size_t index>
  using operand_type =
      std::tuple_element_t<index, std::tuple<std::decay_t<Operands>...>>;

  constexpr explicit Expression(Operands&&... operands)
      : operands_(std::forward<Operands>(operands)...) {
    using Validator = ExpressionValidator<Op, std::decay_t<Operands>...>;
    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      Validator::static_validate();
    } else {
      std::apply(Validator::dynamic_validate, operands_);
    }
  }

  constexpr const auto& operands() const { return operands_; }

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