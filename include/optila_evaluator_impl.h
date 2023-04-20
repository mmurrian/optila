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
  // Accept and store the expression by value if is it small and trivial.
  // Otherwise, accept and store it by const reference.
  //
  // IMPORTANT: This must match the return type behavior of Expression::expr().
  // If Expression::expr() returns by value then this must store by value as
  // well. Otherwise, this would store a dangling reference (the value is a
  // temporary).
  using expression_storage_type =
      details::efficient_type_qualifiers_t<Expression<Op, Operands...>>;

  using operands_storage_type =
      std::tuple<Evaluator<std::decay_t<Operands>>...>;

 public:
  constexpr explicit BaseEvaluator(expression_storage_type expr)
      : m_expr(std::move(expr)), m_nested(m_expr.operands()) {}

  // Deleted constructor for rvalue references. Evaluators shall not be
  // constructed from temporary expressions. What this means is that this is
  // allowed:
  //
  // const auto expr = A + B;
  // Evaluator(expr).evaluate()
  //
  // and this is not allowed:
  //
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

  using operands_return_type =
      details::efficient_type_qualifiers_t<operands_storage_type>;
  constexpr operands_return_type operands() const { return m_nested; }

 private:
  expression_storage_type m_expr;
  operands_storage_type m_nested;
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
#ifndef OPTILA_ENABLE_IMPLICIT_CONVERSIONS
    static_assert(
        std::is_same_v<typename ExprTraits::value_type, OtherValueType>,
        "Implicit conversions are disabled. Use explicit conversions to "
        "convert between types.");
#endif
    dest = Scalar<OtherValueType>(static_cast<OtherValueType>((*this)()));
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
                             std::decay_t<ExprTraits>::num_rows_compile_time,
                             std::decay_t<ExprTraits>::num_cols_compile_time,
                             DefaultMatrixPolicy>;

  constexpr static std::size_t num_rows_compile_time =
      std::decay_t<ExprTraits>::num_rows_compile_time;
  constexpr static std::size_t num_cols_compile_time =
      std::decay_t<ExprTraits>::num_cols_compile_time;
  constexpr static std::size_t num_rows_hint =
      std::decay_t<ExprTraits>::num_rows_hint;
  constexpr static std::size_t num_cols_hint =
      std::decay_t<ExprTraits>::num_cols_hint;

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
            std::size_t OtherNumCols, typename OtherPolicy>
  constexpr void evaluate_into(Matrix<OtherValueType, OtherNumRows,
                                      OtherNumCols, OtherPolicy>& dest) const {
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
          typename Policy>
class Evaluator<Matrix<ValueType, NumRows, NumCols, Policy>>
    : public BaseEvaluator<Matrix<ValueType, NumRows, NumCols, Policy>>,
      public details::matrix_tag {
  using matrix_storage_type = details::efficient_type_qualifiers_t<
      Matrix<ValueType, NumRows, NumCols, Policy>>;

 public:
  using result_type = Matrix<ValueType, NumRows, NumCols, Policy>;

  constexpr explicit Evaluator(matrix_storage_type value)
      : m_value(std::move(value)) {}
  constexpr explicit Evaluator(result_type&&) = delete;

  constexpr static auto num_rows_compile_time = NumRows;
  constexpr static auto num_cols_compile_time = NumCols;
  constexpr static auto num_rows_hint = Policy::NumRowsHint;
  constexpr static auto num_cols_hint = Policy::NumColsHint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return m_value.num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return m_value.num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return m_value(i, j);
  }

  constexpr matrix_storage_type evaluate() const { return m_value; }

  template <typename OtherValueType, std::size_t OtherNumRows,
            std::size_t OtherNumCols, typename OtherPolicy>
  constexpr void evaluate_into(Matrix<OtherValueType, OtherNumRows,
                                      OtherNumCols, OtherPolicy>& dest) const {
    dest = m_value;
  }

 private:
  matrix_storage_type m_value;
};

template <typename ValueType>
class Evaluator<Scalar<ValueType>> : public BaseEvaluator<Scalar<ValueType>>,
                                     public details::scalar_tag {
  using scalar_storage_type =
      details::efficient_type_qualifiers_t<Scalar<ValueType>>;

 public:
  using result_type = Scalar<ValueType>;

  constexpr explicit Evaluator(scalar_storage_type value)
      : m_value(std::move(value)) {}
  constexpr Evaluator(result_type&&) = delete;

  constexpr decltype(auto) operator()() const { return m_value(); }

  constexpr ValueType evaluate() const { return m_value(); }

  template <typename OtherValueType>
  constexpr void evaluate_into(OtherValueType& dest) const {
    using CommonValueType =
        details::common_value_type_t<ValueType, OtherValueType>;
    dest = static_cast<CommonValueType>(m_value());
  }

 private:
  scalar_storage_type m_value;
};

template <typename Expr,
          typename = std::enable_if_t<details::is_expression_literal_v<Expr>>>
Evaluator(const Expr& expr) -> Evaluator<Expr>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename Policy>
Evaluator(const Matrix<ValueType, NumRows, NumCols, Policy>& expr)
    -> Evaluator<Matrix<ValueType, NumRows, NumCols, Policy>>;

template <typename ValueType>
Evaluator(const Scalar<ValueType>& expr) -> Evaluator<Scalar<ValueType>>;

}  // namespace optila