#pragma once

#include <cassert>

#include "details/optila_matrix.h"
#include "details/optila_type_traits.h"

namespace optila {

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          StorageOrder Order = StorageOrder::RowMajor>
class Matrix : public details::matrix_tag {
 public:
  using value_type = ValueType;

  constexpr Matrix() = default;

  // Constructor for nested-brace syntax
  template <
      std::size_t R = NumRows, std::size_t C = NumCols,
      typename std::enable_if_t<(R != Dynamic) && (C != Dynamic), int> = 0>
  constexpr Matrix(const ValueType (&init)[R][C]) {
    static_assert(R == NumRows && C == NumCols,
                  "Static matrix initialization must match matrix size");
    std::size_t i = 0;
    for (const auto& row : init) {
      std::size_t j = 0;
      for (const auto& elem : row) {
        (*this)(i, j++) = elem;
      }
      ++i;
    }
  }

  // Constructor for dynamic-sized matrices
  constexpr Matrix(
      std::initializer_list<std::initializer_list<value_type>> init) {
    // Constructor implementation for dynamic-sized matrices
    if constexpr (num_rows_static() == Dynamic ||
                  num_cols_static() == Dynamic) {
      const std::size_t num_rows = init.size();
      const std::size_t num_cols = init.begin()->size();
      resize(num_rows, num_cols);
    }

    std::size_t i = 0;
    for (const auto& row : init) {
      std::size_t j = 0;
      for (const auto& elem : row) {
        (*this)(i, j++) = elem;
      }
      ++i;
    }
  }

  [[nodiscard]] constexpr value_type& operator()(std::size_t i, std::size_t j) {
    return storage_[linear_index(i, j)];
  }

  // Return by value or const reference depending on the size and other
  // characteristics of the value type
  using const_value_type = details::return_by_value_or_const_ref_t<value_type>;
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

  constexpr static std::size_t num_rows_static() { return NumRows; }
  constexpr static std::size_t num_cols_static() { return NumCols; }
  [[nodiscard]] constexpr std::size_t num_rows() const {
    if constexpr (num_rows_static() != Dynamic)
      return num_rows_static();
    else
      return storage_.num_rows();
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    if constexpr (num_cols_static() != Dynamic)
      return num_cols_static();
    else
      return storage_.num_cols();
  }

  void resize(std::size_t new_num_rows, std::size_t new_num_cols) {
    static_assert(num_rows_static() == Dynamic || num_cols_static() == Dynamic,
                  "Cannot resize a static-sized matrix");

    if constexpr (num_rows_static() != Dynamic) {
      assert(new_num_rows == num_rows_static());
      new_num_rows = num_rows_static();
    }
    if constexpr (num_cols_static() != Dynamic) {
      assert(new_num_cols == num_cols_static());
      new_num_cols = num_cols_static();
    }

    storage_.resize(new_num_rows, new_num_cols);
  }

 private:
  using storage_type = std::conditional_t<
      num_rows_static() != Dynamic && num_cols_static() != Dynamic,
      details::MatrixStaticStorage<ValueType, NumRows, NumCols>,
      details::MatrixDynamicStorage<ValueType>>;
  storage_type storage_;

  [[nodiscard]] constexpr std::size_t linear_index(std::size_t i,
                                                   std::size_t j) const {
    if constexpr (Order == StorageOrder::RowMajor) {
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

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
constexpr decltype(auto) make_matrix(
    std::initializer_list<std::initializer_list<ValueType>> init) {
  return Matrix<ValueType, NumRows, NumCols>(init);
}

}  // namespace optila