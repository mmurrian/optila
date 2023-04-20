#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

namespace optila {

enum class StorageOrder {
  RowMajor,
  ColumnMajor,
};

enum : std::size_t { Dynamic = static_cast<std::size_t>(-1) };

struct DefaultMatrixPolicy {
  constexpr static auto NumRowsHint = 1000;
  constexpr static auto NumColsHint = 1000;
  constexpr static StorageOrder Order = StorageOrder::RowMajor;
};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename Policy>
class Matrix;

namespace details {

struct matrix_tag {};

// Type traits for detecting matrix, vector, and scalar types
template <typename T>
struct is_matrix : std::is_base_of<details::matrix_tag, std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_matrix_v = is_matrix<T>::value;

template <typename T>
struct is_static_row_vector
    : std::conjunction<
          is_matrix<T>,
          std::bool_constant<std::decay_t<T>::num_rows_compile_time == 1>> {};

template <typename T>
inline constexpr bool is_static_row_vector_v = is_static_row_vector<T>::value;

template <typename T>
struct is_static_column_vector
    : std::conjunction<
          is_matrix<T>,
          std::bool_constant<std::decay_t<T>::num_cols_compile_time == 1>> {};

template <typename T>
inline constexpr bool is_static_column_vector_v =
    is_static_column_vector<T>::value;

template <typename T>
struct is_static_vector
    : std::disjunction<is_static_row_vector<T>, is_static_column_vector<T>> {};

template <typename T>
inline constexpr bool is_static_vector_v = is_static_vector<T>::value;

template <typename T>
struct is_matrix_literal : std::false_type {};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename Policy>
struct is_matrix_literal<Matrix<ValueType, NumRows, NumCols, Policy>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_matrix_literal_v = is_matrix_literal<T>::value;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          bool OwnsData = true>
struct MatrixStaticStorage {
  using value_type = ValueType;

  constexpr value_type* data() { return data_.data(); }
  [[nodiscard]] constexpr const value_type* data() const {
    return data_.data();
  }

  constexpr value_type& operator[](std::size_t idx) { return data_[idx]; }
  constexpr const value_type& operator[](std::size_t idx) const {
    return data_[idx];
  }

  constexpr static auto num_rows() { return NumRows; }
  constexpr static auto num_cols() { return NumCols; }

 private:
  struct array_wrapper {
    constexpr value_type& operator[](std::size_t idx) { return data_[idx]; }
    constexpr const value_type& operator[](std::size_t idx) const {
      return data_[idx];
    }

    constexpr value_type* data() { return data_; }
    [[nodiscard]] constexpr const value_type* data() const { return data_; }

   private:
    value_type* data_{};
  };

  using storage_type =
      std::conditional_t<OwnsData, std::array<value_type, NumRows * NumCols>,
                         array_wrapper>;
  storage_type data_{};
};

template <typename ValueType, bool OwnsData = true>
struct MatrixDynamicStorage {
  using value_type = ValueType;

  MatrixDynamicStorage() = default;
  ~MatrixDynamicStorage() {
    if constexpr (OwnsData) delete[] data_;
  }
  MatrixDynamicStorage(const MatrixDynamicStorage& from) { *this = from; }
  MatrixDynamicStorage& operator=(const MatrixDynamicStorage& from) {
    if (this != &from) {
      num_rows_ = from.num_rows_;
      num_cols_ = from.num_cols_;
      if constexpr (OwnsData) {
        data_ = new value_type[num_rows_ * num_cols_];
        std::copy(from.data_, from.data_ + num_rows_ * num_cols_, data_);
      } else {
        data_ = from.data_;
      }
    }
    return *this;
  }
  MatrixDynamicStorage(MatrixDynamicStorage&& from) noexcept {
    *this = std::move(from);
  }
  MatrixDynamicStorage& operator=(MatrixDynamicStorage&& from) noexcept {
    std::swap(num_rows_, from.num_rows_);
    std::swap(num_cols_, from.num_cols_);
    std::swap(data_, from.data_);
    return *this;
  }

  constexpr value_type* data() { return data_; }
  [[nodiscard]] constexpr const value_type* data() const { return data_; }

  constexpr value_type& operator[](std::size_t idx) { return data_[idx]; }
  constexpr const value_type& operator[](std::size_t idx) const {
    return data_[idx];
  }

  [[nodiscard]] constexpr std::size_t num_rows() const { return num_rows_; }
  [[nodiscard]] constexpr std::size_t num_cols() const { return num_cols_; }

  [[nodiscard]] constexpr std::size_t size() const {
    return num_rows_ * num_cols_;
  }

  void resize(std::size_t num_rows, std::size_t num_cols) {
    num_rows_ = num_rows;
    num_cols_ = num_cols;
    if (OwnsData) {
      delete[] data_;
      data_ = new value_type[num_rows * num_cols];
    }
  }

 private:
  std::size_t num_rows_{};
  std::size_t num_cols_{};
  value_type* data_{};
};

// A non-owning, static storage matrix is just a pointer to the data
static_assert(sizeof(MatrixStaticStorage<double, 10, 10, false>) <=
              std::max(alignof(double*), sizeof(double*)));

// An owning, static storage matrix is just an array
static_assert(sizeof(MatrixStaticStorage<double, 10, 10, true>) <=
              std::max(alignof(double), sizeof(double)) * 10 * 10);

// Owning and non-owning dynamic storage matrices are the same size
static_assert(sizeof(MatrixDynamicStorage<double, false>) ==
              sizeof(MatrixDynamicStorage<double, true>));

}  // namespace details
}  // namespace optila