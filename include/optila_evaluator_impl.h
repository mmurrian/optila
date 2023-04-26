#pragma once

#include "details/optila_evaluator.h"
#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_impl.h"
#include "optila_expression_traits_impl.h"

namespace optila {

template <typename Expr, typename Strategy = details::lazy_strategy_tag,
          typename EvaluatorPolicyChain = void, typename Enable = void>
class Evaluator;

template <typename Expr, typename Strategy, typename EvaluatorPolicyChain>
class LazyEvaluatorBase {};

template <typename Op, typename... Operands, typename Strategy,
          typename EvaluatorPolicyChain>
class LazyEvaluatorBase<Expression<Op, Operands...>, Strategy,
                        EvaluatorPolicyChain> {
  using Expr = Expression<Op, Operands...>;
  // Accept and store the expression by value if is it small and trivial.
  // Otherwise, accept and store it by const reference.
  //
  // IMPORTANT: This must match the return type behavior of Expression::expr().
  // If Expression::expr() returns by value then this must store by value as
  // well. Otherwise, this would store a dangling reference (the value is a
  // temporary).
  using expression_storage_type =
      details::efficient_type_qualifiers_t<Expression<Op, Operands...>>;

  template <typename OperandsTuple, std::size_t... Is>
  constexpr static decltype(auto) make_nested_evaluators(
      OperandsTuple&& operands, std::index_sequence<Is...>) {
    return std::make_tuple(
        Evaluator<
            std::decay_t<std::tuple_element_t<Is, std::decay_t<OperandsTuple>>>,
            details::next_strategy_t<Strategy>,
            typename EvaluatorPolicyChain::template operand_policy_type<Expr,
                                                                        Is>>(
            std::get<Is>(operands))...);
  }

  template <typename OperandsTuple>
  constexpr static decltype(auto) make_nested_evaluators(
      OperandsTuple&& operands) {
    return make_nested_evaluators(
        std::forward<OperandsTuple>(operands),
        std::make_index_sequence<
            std::tuple_size_v<std::decay_t<OperandsTuple>>>{});
  }

  using operands_storage_type = std::decay_t<decltype(make_nested_evaluators(
      std::declval<expression_storage_type>().operands()))>;

 public:
  constexpr explicit LazyEvaluatorBase(expression_storage_type expr)
      : m_expr(std::move(expr)),
        m_nested(make_nested_evaluators(m_expr.operands())) {}

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
  LazyEvaluatorBase(Expression<Op, Operands...>&&) = delete;

 protected:
  using operation_return_type = details::efficient_type_qualifiers_t<Op>;
  constexpr operation_return_type operation() const {
    static_assert(std::is_base_of_v<Op, Expr>);
    return static_cast<operation_return_type>(m_expr);
  }
  constexpr expression_storage_type expr() const { return m_expr; }

  using operands_return_type =
      details::efficient_type_qualifiers_t<operands_storage_type>;
  constexpr operands_return_type operands() const { return m_nested; }

 private:
  expression_storage_type m_expr;
  operands_storage_type m_nested;
};

// Matrix expression lazy evaluator.
template <typename Op, typename... Operands, typename Strategy,
          typename EvaluatorPolicyChain>
class Evaluator<
    Expression<Op, Operands...>, Strategy, EvaluatorPolicyChain,
    std::enable_if_t<
        details::is_matrix_v<Expression<Op, Operands...>> &&
        details::use_lazy_strategy_v<Strategy, EvaluatorPolicyChain>>>
    : public LazyEvaluatorBase<Expression<Op, Operands...>, Strategy,
                               EvaluatorPolicyChain>,
      public details::matrix_tag {
  using Base = LazyEvaluatorBase<Expression<Op, Operands...>, Strategy,
                                 EvaluatorPolicyChain>;

  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;
  using result_type = typename ExprTraits::result_type;

 public:
  using Base::Base;

  constexpr static auto num_rows_compile_time =
      std::decay_t<ExprTraits>::num_rows_compile_time;
  constexpr static auto num_cols_compile_time =
      std::decay_t<ExprTraits>::num_cols_compile_time;
  constexpr static auto num_rows_hint = std::decay_t<ExprTraits>::num_rows_hint;
  constexpr static auto num_cols_hint = std::decay_t<ExprTraits>::num_cols_hint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return Base::expr().num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return Base::expr().num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return std::apply(
        [i, j, this](auto&&... operands) {
          return Base::operation().to_matrix_element(
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

    using active_bounds = typename EvaluatorPolicyChain::active_sub_matrix;
    using expr_bounds = MatrixBounds<0, 0, ExprTraits::num_rows_compile_time,
                                     ExprTraits::num_cols_compile_time>;

    const auto sub_matrix =
        intersect(active_bounds{}, expr_bounds(0, 0, num_rows(), num_cols()));
    for (std::size_t i = sub_matrix.row0(); i < sub_matrix.row_end(); ++i) {
      for (std::size_t j = sub_matrix.col0(); j < sub_matrix.col_end(); ++j) {
        dest(i, j) = (*this)(i, j);
      }
    }
  }
};

// Matrix expression eager evaluator.
template <typename Op, typename... Operands, typename Strategy,
          typename EvaluatorPolicyChain>
class Evaluator<
    Expression<Op, Operands...>, Strategy, EvaluatorPolicyChain,
    std::enable_if_t<
        details::is_matrix_v<Expression<Op, Operands...>> &&
        details::use_eager_strategy_v<Strategy, EvaluatorPolicyChain>>>
    : public details::matrix_tag {
  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;
  using result_type = typename ExprTraits::result_type;

  using expression_storage_type = details::efficient_type_qualifiers_t<Expr>;

  Evaluator<
      std::decay_t<Expr>,
      details::once_and_then_strategy<details::lazy_strategy_tag, Strategy>,
      EvaluatorPolicyChain>
      m_evaluator;

  result_type m_result;

 public:
  constexpr Evaluator(expression_storage_type expr) : m_evaluator(expr) {
    m_evaluator.evaluate_into(m_result);
  }

  constexpr static auto num_rows_compile_time =
      result_type::num_rows_compile_time;
  constexpr static auto num_cols_compile_time =
      result_type::num_cols_compile_time;
  constexpr static auto num_rows_hint = result_type::num_rows_hint;
  constexpr static auto num_cols_hint = result_type::num_cols_hint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return m_result.num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return m_result.num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return m_result(i, j);
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
    dest = m_result;
  }
};

// Scalar expression lazy evaluator.
template <typename Op, typename... Operands, typename Strategy,
          typename EvaluatorPolicyChain>
class Evaluator<
    Expression<Op, Operands...>, Strategy, EvaluatorPolicyChain,
    std::enable_if_t<
        details::is_scalar_v<Expression<Op, Operands...>> &&
        details::use_lazy_strategy_v<Strategy, EvaluatorPolicyChain>>>
    : public LazyEvaluatorBase<Expression<Op, Operands...>, Strategy,
                               EvaluatorPolicyChain>,
      public details::scalar_tag {
  using Base = LazyEvaluatorBase<Expression<Op, Operands...>, Strategy,
                                 EvaluatorPolicyChain>;

  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;
  using result_type = typename ExprTraits::result_type;

 public:
  using Base::Base;

  constexpr decltype(auto) operator()() const {
    return std::apply(Base::operation().to_scalar, Base::operands());
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

// Scalar expression eager evaluator.
template <typename Op, typename... Operands, typename Strategy,
          typename EvaluatorPolicyChain>
class Evaluator<
    Expression<Op, Operands...>, Strategy, EvaluatorPolicyChain,
    std::enable_if_t<
        details::is_scalar_v<Expression<Op, Operands...>> &&
        details::use_eager_strategy_v<Strategy, EvaluatorPolicyChain>>>
    : public details::scalar_tag {
  using Expr = Expression<Op, Operands...>;
  using ExprTraits = ExpressionTraits<Expr>;
  using value_type = typename ExprTraits::value_type;
  using result_type = typename ExprTraits::result_type;
  using expression_type = details::efficient_type_qualifiers_t<Expr>;

  Evaluator<
      std::decay_t<Expr>,
      details::once_and_then_strategy<details::lazy_strategy_tag, Strategy>,
      EvaluatorPolicyChain>
      m_evaluator;

  result_type m_result;

 public:
  constexpr Evaluator(expression_type expr) : m_evaluator(expr) {
    m_evaluator.evaluate_into(m_result);
  }

  constexpr decltype(auto) operator()() const { return m_result(); }

  constexpr result_type evaluate() const { return m_result; }

  template <typename OtherValueType>
  constexpr void evaluate_into(Scalar<OtherValueType>& dest) const {
    using CommonValueType =
        details::common_value_type_t<value_type, OtherValueType>;
    dest = Scalar<OtherValueType>(static_cast<CommonValueType>(m_result()));
  }
};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename MatrixPolicy, typename Strategy,
          typename EvaluatorPolicyChain>
class Evaluator<Matrix<ValueType, NumRows, NumCols, MatrixPolicy>, Strategy,
                EvaluatorPolicyChain> : public details::matrix_tag {
  using Expr = Matrix<ValueType, NumRows, NumCols, MatrixPolicy>;
  using ExprTraits = ExpressionTraits<Expr>;
  using result_type = typename ExprTraits::result_type;
  using result_storage_type = details::efficient_type_qualifiers_t<result_type>;

 public:
  constexpr explicit Evaluator(result_storage_type value)
      : m_value(std::move(value)) {}
  constexpr explicit Evaluator(result_type&&) = delete;

  constexpr static auto num_rows_compile_time =
      std::decay_t<ExprTraits>::num_rows_compile_time;
  constexpr static auto num_cols_compile_time =
      std::decay_t<ExprTraits>::num_cols_compile_time;
  constexpr static auto num_rows_hint = std::decay_t<ExprTraits>::num_rows_hint;
  constexpr static auto num_cols_hint = std::decay_t<ExprTraits>::num_cols_hint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    return m_value.num_rows();
  }

  [[nodiscard]] constexpr std::size_t num_cols() const {
    return m_value.num_cols();
  }

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j) const {
    return m_value(i, j);
  }

  constexpr result_storage_type evaluate() const { return m_value; }

  template <typename OtherValueType, std::size_t OtherNumRows,
            std::size_t OtherNumCols, typename OtherPolicy>
  constexpr void evaluate_into(Matrix<OtherValueType, OtherNumRows,
                                      OtherNumCols, OtherPolicy>& dest) const {
    dest = m_value;
  }

 private:
  result_storage_type m_value;
};

template <typename ValueType, typename Strategy, typename EvaluatorPolicyChain>
class Evaluator<Scalar<ValueType>, Strategy, EvaluatorPolicyChain>
    : public details::scalar_tag {
  using Expr = Scalar<ValueType>;
  using ExprTraits = ExpressionTraits<Expr>;
  using result_type = typename ExprTraits::result_type;
  using result_storage_type = details::efficient_type_qualifiers_t<result_type>;

 public:
  constexpr explicit Evaluator(result_storage_type value)
      : m_value(std::move(value)) {}
  constexpr Evaluator(result_type&&) = delete;

  constexpr decltype(auto) operator()() const { return m_value(); }

  constexpr result_type evaluate() const { return m_value; }

  template <typename OtherValueType>
  constexpr void evaluate_into(Scalar<OtherValueType>& dest) const {
    using CommonValueType =
        details::common_value_type_t<ValueType, OtherValueType>;
    dest = Scalar<OtherValueType>(static_cast<CommonValueType>(m_value()));
  }

 private:
  result_storage_type m_value;
};

template <typename Expr, typename Strategy = details::lazy_strategy_tag,
          typename EvaluatorPolicyChain = void,
          typename = std::enable_if_t<details::is_expression_literal_v<Expr>>>
Evaluator(const Expr& expr) -> Evaluator<Expr, Strategy, EvaluatorPolicyChain>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename MatrixPolicy>
Evaluator(const Matrix<ValueType, NumRows, NumCols, MatrixPolicy>& expr)
    -> Evaluator<Matrix<ValueType, NumRows, NumCols, MatrixPolicy>>;

template <typename ValueType>
Evaluator(const Scalar<ValueType>& expr) -> Evaluator<Scalar<ValueType>>;

}  // namespace optila