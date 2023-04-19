#pragma once

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_impl.h"
#include "optila_expression_traits_impl.h"

namespace optila {

template <typename Expr, typename Enable = void>
class Evaluator;

template <typename Expr>
class BaseEvaluator {};

template <typename Op, typename... Operands>
class BaseEvaluator<Expression<Op, Operands...>> {
 public:
  constexpr explicit BaseEvaluator(const Expression<Op, Operands...>& expr)
      : m_expr(expr), m_nested(m_expr.operands()) {}

  // Deleted constructor for rvalue references. Evaluators shall not be
  // constructed from temporary expressions. What this means is that this is
  // allowed:
  //
  // const auto expr = A + B;
  // Evaluator(expr).evaluate()
  //
  // and this is not allowed:
  // Evaluator(A + B).evaluate()
  //
  // To be clear, in the disallowed example, the lifetime of  A + B ends after
  // the constructor and is dead by the time evaluate() is called.
  //
  // Users shouldn't be directly interacting with Evaluators anyways so this is
  // more of a note for myself.
  BaseEvaluator(Expression<Op, Operands...>&&) = delete;

 protected:
  constexpr decltype(auto) expr() const { return m_expr; }
  constexpr decltype(auto) operands() const {
    return details::make_tuple_ref(m_nested);
  }

 private:
  const Expression<Op, Operands...>& m_expr;
  std::tuple<Evaluator<std::decay_t<Operands>>...> m_nested;
};

template <typename Op, typename... Operands>
class Evaluator<
    Expression<Op, Operands...>,
    std::enable_if_t<details::is_scalar_v<Expression<Op, Operands...>>>>
    : public BaseEvaluator<Expression<Op, Operands...>>,
      public details::scalar_tag {
  using Base = BaseEvaluator<Expression<Op, Operands...>>;

  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

 public:
  using Base::Base;

  using result_type = Scalar<typename ExprTraits::value_type>;

  constexpr decltype(auto) operator()() const {
    return std::apply(Op::to_scalar, Base::operands());
  }

  constexpr result_type evaluate() const {
    result_type result{};
    this->evaluate_into(result);
    return result;
  }

  template <typename OtherValueType>
  constexpr void evaluate_into(Scalar<OtherValueType>& dest) const {
    using CommonValueType =
        details::common_value_type_t<typename ExprTraits::value_type,
                                     OtherValueType>;
    dest = Scalar<OtherValueType>((*this)());
  }
};

template <typename Op, typename... Operands>
class Evaluator<
    Expression<Op, Operands...>,
    std::enable_if_t<details::is_matrix_v<Expression<Op, Operands...>>>>
    : public BaseEvaluator<Expression<Op, Operands...>>,
      public details::matrix_tag {
  using Base = BaseEvaluator<Expression<Op, Operands...>>;

  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

 public:
  using Base::Base;

  using result_type = Matrix<typename ExprTraits::value_type,
                             std::decay_t<ExprTraits>::num_rows_static(),
                             std::decay_t<ExprTraits>::num_cols_static(),
                             StorageOrder::RowMajor>;

  constexpr static std::size_t num_rows_static() {
    return std::decay_t<ExprTraits>::num_rows_static();
  }

  constexpr static std::size_t num_cols_static() {
    return std::decay_t<ExprTraits>::num_cols_static();
  }

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return Base::expr().num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return Base::expr().num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return std::apply(
        [i, j](auto&&... operands) {
          return Op::to_matrix_element(
              i, j, std::forward<decltype(operands)>(operands)...);
        },
        Base::operands());
  }

  constexpr result_type evaluate() const {
    result_type result{};
    this->evaluate_into(result);
    return result;
  }

  template <typename OtherValueType, std::size_t OtherNumRows,
            std::size_t OtherNumCols, StorageOrder OtherOrder>
  constexpr void evaluate_into(Matrix<OtherValueType, OtherNumRows,
                                      OtherNumCols, OtherOrder>& dest) const {
    using CommonValueType =
        details::common_value_type_t<typename ExprTraits::value_type,
                                     OtherValueType>;

    if constexpr (details::is_dynamic_expression_v<decltype(dest)>) {
      dest.resize(num_rows(), num_cols());
    }
    for (std::size_t i = 0; i < Base::expr().num_rows(); ++i) {
      for (std::size_t j = 0; j < Base::expr().num_cols(); ++j) {
        dest(i, j) = (*this)(i, j);
      }
    }
  }
};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          StorageOrder Order>
class Evaluator<Matrix<ValueType, NumRows, NumCols, Order>>
    : public BaseEvaluator<Matrix<ValueType, NumRows, NumCols, Order>>,
      public details::matrix_tag {
 public:
  using result_type = Matrix<ValueType, NumRows, NumCols, Order>;

  constexpr explicit Evaluator(const result_type& value) : value_(value) {}

  constexpr static std::size_t num_rows_static() { return NumRows; }

  constexpr static std::size_t num_cols_static() { return NumCols; }

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return value_.num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return value_.num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return value_(i, j);
  }

  constexpr const result_type& evaluate() const { return value_; }

  template <typename OtherValueType, std::size_t OtherNumRows,
            std::size_t OtherNumCols, StorageOrder OtherOrder>
  constexpr void evaluate_into(Matrix<OtherValueType, OtherNumRows,
                                      OtherNumCols, OtherOrder>& dest) const {
    dest = value_;
  }

 private:
  const result_type& value_;
};

template <typename ValueType>
class Evaluator<Scalar<ValueType>> : public BaseEvaluator<Scalar<ValueType>>,
                                     public details::scalar_tag {
 public:
  using result_type = Scalar<ValueType>;

  constexpr explicit Evaluator(const result_type& value) : value_(value) {}

  constexpr decltype(auto) operator()() const { return value_(); }

  constexpr ValueType evaluate() const { return value_(); }

  template <typename OtherValueType>
  constexpr void evaluate_into(OtherValueType& dest) const {
    using CommonValueType =
        details::common_value_type_t<ValueType, OtherValueType>;
    dest = static_cast<CommonValueType>(value_());
  }

 private:
  const result_type& value_;
};

template <typename Expr,
          typename = std::enable_if_t<details::is_expression_literal_v<Expr>>>
Evaluator(const Expr& expr) -> Evaluator<Expr>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          StorageOrder Order>
Evaluator(const Matrix<ValueType, NumRows, NumCols, Order>& expr)
    -> Evaluator<Matrix<ValueType, NumRows, NumCols, Order>>;

template <typename ValueType>
Evaluator(const Scalar<ValueType>& expr) -> Evaluator<Scalar<ValueType>>;

}  // namespace optila