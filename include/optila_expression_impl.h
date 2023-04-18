#pragma once

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_validator_impl.h"
#include "optila_operation_impl.h"
#include "optila_scalar_impl.h"

namespace optila {

namespace details {

template <typename Expr, typename ValueType, std::size_t NumRows,
          std::size_t NumCols, StorageOrder Order = StorageOrder::RowMajor>
constexpr void evaluate_expression_into_matrix(
    const Expr& expr, Matrix<ValueType, NumRows, NumCols, Order>& result) {
  static_assert(details::is_matrix_v<Expr>, "Expression must be a matrix");
  static_assert(
      std::is_same_v<
          details::result_type_t<ValueType, typename Expr::value_type>,
          ValueType>,
      "Incompatible value types");
  static_assert(
      Expr::num_rows_static() == NumRows || NumRows == Dynamic ||
          Expr::num_rows_static() == Dynamic,
      "Static row count mismatch between expression and result matrix");
  static_assert(
      Expr::num_cols_static() == NumCols || NumCols == Dynamic ||
          Expr::num_cols_static() == Dynamic,
      "Static column count mismatch between expression and result matrix");

  if constexpr (details::is_dynamic_expression_v<decltype(result)>) {
    result.resize(expr.num_rows(), expr.num_cols());
  } else if constexpr (details::is_dynamic_expression_v<Expr>) {
    assert(NumRows == expr.num_rows() && NumCols == expr.num_cols());
  }
  for (std::size_t i = 0; i < result.num_rows(); ++i) {
    for (std::size_t j = 0; j < result.num_cols(); ++j) {
      result(i, j) = expr(i, j);
    }
  }
}

template <typename Expr, typename ValueType>
constexpr void evaluate_expression_into_scalar(const Expr& expr,
                                               Scalar<ValueType>& result) {
  static_assert(details::is_scalar_v<Expr>, "Expression must be a scalar");
  static_assert(
      std::is_same_v<
          details::result_type_t<ValueType, typename Expr::value_type>,
          ValueType>,
      "Incompatible value types");

  result = make_scalar(expr());
}

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
    using ResultType = Matrix<ValueType, NumRows, NumCols, Order>;
    ResultType result{};
    evaluate_expression_into_matrix(derived(), result);
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
  constexpr decltype(auto) operator()() const {
    return Op::apply_scalar(derived());
  }

  template <typename ValueType>
  constexpr operator Scalar<ValueType>() const {
    Scalar<ValueType> result{};
    evaluate_expression_into_scalar(derived(), result);
    return result;
  }
};

// Partial specialization for the scalar Evaluate operation
template <typename Lhs>
class ExpressionImpl<details::scalar_tag, Operation::Evaluate, Lhs>
    : public details::scalar_tag {
  using Derived = Expression<Operation::Evaluate, Lhs>;
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }

 public:
  constexpr decltype(auto) operator()() const {
    return derived().template operand<0>()();
  }

  template <typename ValueType>
  constexpr operator Scalar<ValueType>() const {
    // FIXME
    return optila::make_scalar<ValueType>(derived().template operand<0>()());
  }

 private:
  using result_type = Scalar<typename Lhs::value_type>;
  mutable result_type result{};
};

// Partial specialization for the matrix Evaluate operation
template <typename Lhs>
class ExpressionImpl<details::matrix_tag, Operation::Evaluate, Lhs>
    : public details::matrix_tag {
  using Derived = Expression<Operation::Evaluate, Lhs>;
  constexpr const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
  constexpr Derived& derived() { return static_cast<Derived&>(*this); }

 public:
  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return derived().template operand<0>()(i, j);
  }

  static constexpr std::size_t num_rows_static() {
    return std::decay_t<Lhs>::num_rows_static();
  }
  static constexpr std::size_t num_cols_static() {
    return std::decay_t<Lhs>::num_cols_static();
  }
  [[nodiscard]] constexpr std::size_t num_rows() const {
    return derived().template operand<0>().num_rows();
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    return derived().template operand<0>().num_cols();
  }

  template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
            StorageOrder Order = StorageOrder::RowMajor>
  constexpr operator Matrix<ValueType, NumRows, NumCols, Order>() const {
    // FIXME
    return derived().template operand<0>();
  }

 private:
  using result_type =
      Matrix<typename std::decay_t<Lhs>::value_type,
             std::decay_t<Lhs>::num_rows_static(),
             std::decay_t<Lhs>::num_cols_static(), StorageOrder::RowMajor>;
  mutable result_type result{};
};

}  // namespace details

template <typename Op, typename... Operands>
class Expression : public details::ExpressionImpl<
                       typename ExpressionValidator<
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