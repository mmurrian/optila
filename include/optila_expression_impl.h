#pragma once

#include <type_traits>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_traits_impl.h"
#include "optila_operation_impl.h"
#include "optila_scalar_impl.h"

namespace optila {

template <typename ExprType, typename Op, typename... Operands>
struct expression_result_type;

template <typename Op, typename... Operands>
struct expression_result_type<details::scalar_tag, Op, Operands...> {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;
  using type = Scalar<typename ExprTraits::value_type>;
};

template <typename Op, typename... Operands>
struct expression_result_type<details::matrix_tag, Op, Operands...> {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;
  using type = Matrix<typename ExprTraits::value_type,
                      std::decay_t<ExprTraits>::num_rows_static(),
                      std::decay_t<ExprTraits>::num_cols_static(),
                      StorageOrder::RowMajor>;
};

template <typename ExprType, typename Op, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::matrix_tag, Op, Operands...>
    : public details::matrix_tag {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

 public:
  using value_type = typename ExprTraits::value_type;
  using operation_type = Op;
  template <std::size_t index>
  using operand_type =
      std::tuple_element_t<index, std::tuple<std::decay_t<Operands>...>>;

  constexpr explicit ExpressionImpl(Operands&&... operands)
      : storage_(std::forward<Operands>(operands)...) {
    // Instantiating this type will cause a compile-time error if it is
    // non-constexpr and the expression is used in a constexpr context. This
    // would happen eventually if the expression truly cannot be evaluated in a
    // constexpr context, but it is better to fail early (and hopefully provide
    // a more helpful error message).
    [[maybe_unused]] Op check_for_nonconstexpr_operation_in_constexpr_context{};

    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      ExprTraits::static_validate();
    } else {
      std::apply(ExprTraits::dynamic_validate, storage_.operands);
    }

    if constexpr (!ExprTraits::lazy_evaluation) {
      storage_.result = uncached_evaluate();
    }
  }

  constexpr value_type operator()(std::size_t i, std::size_t j) const {
    return cached_evaluate(i, j);
  }

  static constexpr std::size_t num_rows_static() {
    return ExpressionTraits<Op, std::decay_t<Operands>...>::num_rows_static();
  }
  static constexpr std::size_t num_cols_static() {
    return ExpressionTraits<Op, std::decay_t<Operands>...>::num_cols_static();
  }
  [[nodiscard]] constexpr std::size_t num_rows() const {
    return std::apply(ExpressionTraits<Op, std::decay_t<Operands>...>::num_rows,
                      storage_.operands);
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    return std::apply(ExpressionTraits<Op, std::decay_t<Operands>...>::num_cols,
                      storage_.operands);
  }

  template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
            StorageOrder Order = StorageOrder::RowMajor>
  constexpr operator Matrix<ValueType, NumRows, NumCols, Order>() const {
    return {cached_evaluate()};
  }

 private:
  constexpr value_type uncached_evaluate(std::size_t i, std::size_t j) const {
    return std::apply(
        [i, j](auto&&... operands) {
          return Op::to_matrix_element(
              i, j, std::forward<decltype(operands)>(operands)...);
        },
        storage_.operands);
  }
  constexpr auto uncached_evaluate() const {
    result_storage_type result;
    if constexpr (details::is_dynamic_expression_v<result_storage_type>) {
      result.resize(num_rows(), num_cols());
    }
    for (std::size_t i = 0; i < result.num_rows(); ++i) {
      for (std::size_t j = 0; j < result.num_cols(); ++j) {
        result(i, j) = uncached_evaluate(i, j);
      }
    }
    return result;
  }
  constexpr value_type cached_evaluate(std::size_t i, std::size_t j) const {
    if constexpr (ExprTraits::lazy_evaluation) {
      return uncached_evaluate(i, j);
    } else {
      return storage_.result(i, j);
    }
  }
  constexpr decltype(auto) cached_evaluate() const {
    if constexpr (ExprTraits::lazy_evaluation) {
      return uncached_evaluate();
    } else {
      return storage_.result;
    }
  }

  using result_storage_type =
      typename expression_result_type<typename ExprTraits::expression_type, Op,
                                      Operands...>::type;
  using operand_storage_type =
      std::tuple<details::store_by_value_or_const_ref_t<Operands>...>;

  struct eager_expression_storage {
    constexpr explicit eager_expression_storage(Operands&&... operands)
        : operands(std::forward<Operands>(operands)...) {}

    result_storage_type result{};
    operand_storage_type operands;
  };
  struct lazy_expression_storage {
    constexpr explicit lazy_expression_storage(Operands&&... operands)
        : operands(std::forward<Operands>(operands)...) {}

    operand_storage_type operands;
  };

  using storage_type =
      std::conditional_t<ExprTraits::lazy_evaluation, lazy_expression_storage,
                         eager_expression_storage>;
  storage_type storage_;
};

// Partial specialization for scalar_tag
template <typename Op, typename... Operands>
class ExpressionImpl<details::scalar_tag, Op, Operands...>
    : public details::scalar_tag {
  using ExprTraits = ExpressionTraits<Op, std::decay_t<Operands>...>;

 public:
  using value_type = typename ExprTraits::value_type;
  using operation_type = Op;
  template <std::size_t index>
  using operand_type =
      std::tuple_element_t<index, std::tuple<std::decay_t<Operands>...>>;

  constexpr explicit ExpressionImpl(Operands&&... operands)
      : storage_(std::forward<Operands>(operands)...) {
    if constexpr (std::conjunction_v<
                      details::is_static_expression<Operands>...>) {
      ExprTraits::static_validate();
    } else {
      std::apply(ExprTraits::dynamic_validate, storage_.operands);
    }

    if constexpr (!ExprTraits::lazy_evaluation) {
      storage_.result = Scalar<value_type>(uncached_evaluate());
    }
  }

  constexpr value_type operator()() const { return cached_evaluate(); }

  template <typename ValueType>
  constexpr operator Scalar<ValueType>() const {
    return Scalar<ValueType>(cached_evaluate());
  }

 private:
  constexpr decltype(auto) uncached_evaluate() const {
    return std::apply(Op::to_scalar, storage_.operands);
  }
  constexpr decltype(auto) cached_evaluate() const {
    if constexpr (ExprTraits::lazy_evaluation) {
      return uncached_evaluate();
    } else {
      return storage_.result;
    }
  }

  using result_storage_type =
      typename expression_result_type<typename ExprTraits::expression_type, Op,
                                      Operands...>::type;
  using operand_storage_type =
      std::tuple<details::store_by_value_or_const_ref_t<Operands>...>;

  struct eager_expression_storage {
    constexpr explicit eager_expression_storage(Operands&&... operands)
        : operands(std::forward<Operands>(operands)...) {}

    result_storage_type result{};
    operand_storage_type operands;
  };
  struct lazy_expression_storage {
    constexpr explicit lazy_expression_storage(Operands&&... operands)
        : operands(std::forward<Operands>(operands)...) {}

    operand_storage_type operands;
  };

  using storage_type =
      std::conditional_t<ExprTraits::lazy_evaluation, lazy_expression_storage,
                         eager_expression_storage>;
  storage_type storage_;
};

template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<typename ExpressionTraits<
                                Op, std::decay_t<Operands>...>::expression_type,
                            Op, Operands...> {
  using Base = ExpressionImpl<
      typename ExpressionTraits<Op, std::decay_t<Operands>...>::expression_type,
      Op, Operands...>;

 public:
  using Base::Base;
};

}  // namespace optila