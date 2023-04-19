#pragma once

#include <utility>

#include "details/optila_expression.h"
#include "details/optila_scalar.h"
#include "optila_evaluator_impl.h"

namespace optila {

template <typename ValueType>
class Scalar : public details::scalar_tag {
 public:
  using value_type = ValueType;

  constexpr Scalar() = default;

  constexpr explicit Scalar(value_type value) : value_(std::move(value)) {}

  template <typename Expr,
            typename = std::enable_if_t<
                details::is_expression_literal_v<std::decay_t<Expr>> &&
                details::is_scalar_v<Expr>>>
  constexpr Scalar(Expr&& expr) {
    // Accept l-value and r-value expressions but do not std::forward<Expr> to
    // the evaluator. The evaluator does not accept r-value expressions and will
    // not manage the lifetime of the expression.
    Evaluator<std::decay_t<Expr>>(expr).evaluate_into(*this);
  }

  constexpr operator decltype(auto)() const { return value_; }
  constexpr decltype(auto) operator()() const { return value_; }

 private:
  ValueType value_;
};

template <typename ValueType>
constexpr Scalar<ValueType> make_scalar(ValueType&& args) {
  return Scalar<ValueType>(std::forward<ValueType>(args));
}

// Deduction guides for Scalar
template <typename ValueType,
          typename = std::enable_if_t<std::is_arithmetic_v<ValueType>>>
Scalar(ValueType) -> Scalar<ValueType>;

template <typename Expr,
          typename = std::enable_if_t<
              details::is_expression_literal_v<std::decay_t<Expr>> &&
              details::is_scalar_v<Expr>>>
Scalar(Expr&& expr) -> Scalar<typename std::decay_t<Expr>::value_type>;

}  // namespace optila