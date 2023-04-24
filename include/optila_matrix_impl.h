#pragma once

#include <cassert>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_type_traits.h"
#include "optila_evaluator_impl.h"
#include "optila_optimizer_impl.h"

namespace optila {

namespace details {
template <typename From, typename To>
constexpr static void assign_matrix_to_matrix(const From& from, To& to) {
  static_assert(std::is_same_v<common_value_type_t<typename From::value_type,
                                                   typename To::value_type>,
                               typename From::value_type>,
                "Cannot assign incompatible types to matrix");
  if constexpr (details::is_dynamic_expression_v<To>) {
    to.resize(from.num_rows(), from.num_cols());
  } else {
    static_assert(
        (From::num_rows_compile_time == To::num_rows_compile_time ||
         From::num_rows_compile_time == Dynamic) &&
            (From::num_cols_compile_time == To::num_cols_compile_time ||
             From::num_cols_compile_time == Dynamic),
        "Cannot assign to static matrix with different size");
    assert(from.num_rows() == to.num_rows() &&
           from.num_cols() == to.num_cols());
  }
  for (std::size_t i = 0; i < to.num_rows(); ++i) {
    for (std::size_t j = 0; j < to.num_cols(); ++j) {
      to(i, j) = from(i, j);
    }
  }
}

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename To>
constexpr static void assign_array_2d_to_matrix(
    const ValueType (&from)[NumRows][NumCols], To& to) {
  if constexpr (To::num_rows_compile_time == Dynamic ||
                To::num_cols_compile_time == Dynamic) {
    to.resize(NumRows, NumCols);
  } else {
    static_assert(NumRows == To::num_rows_compile_time &&
                      NumCols == To::num_cols_compile_time,
                  "Static matrix initialization must match matrix size");
  }

  std::size_t i = 0;
  for (const auto& row : from) {
    std::size_t j = 0;
    for (const auto& elem : row) {
      to(i, j++) = elem;
    }
    ++i;
  }
}

}  // namespace details

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename Policy = DefaultMatrixPolicy>
class Matrix : public details::matrix_tag {
 public:
  static_assert(Policy::NumRowsHint != Dynamic &&
                    Policy::NumColsHint != Dynamic,
                "Matrix policy must specify static hints for matrix size");

  using value_type = ValueType;

  constexpr Matrix() = default;

  // Constructor for nested-brace syntax
  template <
      std::size_t R = NumRows, std::size_t C = NumCols,
      typename std::enable_if_t<(R != Dynamic) && (C != Dynamic), int> = 0>
  constexpr Matrix(const ValueType (&from)[R][C]) {
    details::assign_array_2d_to_matrix(from, *this);
  }

  template <typename Expr,
            typename = std::enable_if_t<
                details::is_expression_literal_v<std::decay_t<Expr>> &&
                details::is_matrix_v<Expr>>>
  constexpr Matrix(Expr&& expr) {
    // Accept l-value and r-value expressions but do not std::forward<Expr> to
    // the evaluator. The evaluator does not accept r-value expressions and will
    // not manage the lifetime of the expression.
    Evaluator<std::decay_t<Expr>, optimize_expression_t<Expr>>(expr)
        .evaluate_into(*this);
  }

  template <typename OtherValueType, std::size_t OtherNumRows,
            std::size_t OtherNumCols, typename OtherPolicy>
  constexpr Matrix(const Matrix<OtherValueType, OtherNumRows, OtherNumCols,
                                OtherPolicy>& other) {
    details::assign_matrix_to_matrix(other, *this);
  }

  [[nodiscard]] constexpr value_type& operator()(std::size_t i, std::size_t j) {
    return storage_[linear_index(i, j)];
  }

  // Return by value or const reference depending on the size and other
  // characteristics of the value type
  using const_value_type = details::efficient_type_qualifiers_t<value_type>;
  [[nodiscard]] constexpr const_value_type operator()(std::size_t i,
                                                      std::size_t j) const {
    return storage_[linear_index(i, j)];
  }

  [[nodiscard]] constexpr value_type* data() noexcept {
    return storage_.data();
  }

  [[nodiscard]] constexpr const value_type* data() const noexcept {
    return storage_.data();
  }

  constexpr static auto num_rows_compile_time = NumRows;
  constexpr static auto num_cols_compile_time = NumCols;
  constexpr static auto num_rows_hint =
      NumRows != Dynamic ? NumRows : Policy::NumRowsHint;
  constexpr static auto num_cols_hint =
      NumCols != Dynamic ? NumCols : Policy::NumColsHint;

  [[nodiscard]] constexpr std::size_t num_rows() const {
    if constexpr (num_rows_compile_time != Dynamic)
      return num_rows_compile_time;
    else
      return storage_.num_rows();
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    if constexpr (num_cols_compile_time != Dynamic)
      return num_cols_compile_time;
    else
      return storage_.num_cols();
  }

  void resize(std::size_t new_num_rows, std::size_t new_num_cols) {
    static_assert(
        num_rows_compile_time == Dynamic || num_cols_compile_time == Dynamic,
        "Cannot resize a static-sized matrix");

    if constexpr (num_rows_compile_time != Dynamic) {
      assert(new_num_rows == num_rows_compile_time);
      new_num_rows = num_rows_compile_time;
    }
    if constexpr (num_cols_compile_time != Dynamic) {
      assert(new_num_cols == num_cols_compile_time);
      new_num_cols = num_cols_compile_time;
    }

    storage_.resize(new_num_rows, new_num_cols);
  }

 private:
  using storage_type = std::conditional_t<
      num_rows_compile_time != Dynamic && num_cols_compile_time != Dynamic,
      details::MatrixStaticStorage<ValueType, NumRows, NumCols>,
      details::MatrixDynamicStorage<ValueType>>;
  storage_type storage_;

  [[nodiscard]] constexpr std::size_t linear_index(std::size_t i,
                                                   std::size_t j) const {
    if constexpr (Policy::Order == StorageOrder::RowMajor) {
      return i * num_cols() + j;
    } else {  // StorageOrder::ColumnMajor
      return i + j * num_rows();
    }
  }
};

// Deduction guide for Matrix
template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
Matrix(const ValueType (&)[NumRows][NumCols])
    -> Matrix<ValueType, NumRows, NumCols>;

template <typename Expr,
          typename = std::enable_if_t<
              details::is_expression_literal_v<std::decay_t<Expr>> &&
              details::is_matrix_v<Expr>>>
Matrix(Expr&& expr) -> Matrix<typename std::decay_t<Expr>::value_type,
                              std::decay_t<Expr>::num_rows_compile_time,
                              std::decay_t<Expr>::num_cols_compile_time>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
constexpr decltype(auto) make_matrix(
    const ValueType (&from)[NumRows][NumCols]) {
  return Matrix<ValueType, NumRows, NumCols>(from);
}

}  // namespace optila