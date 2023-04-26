#pragma once

#include <algorithm>
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

template <std::size_t Row0, std::size_t Col0, std::size_t NumRows,
          std::size_t NumCols>
struct MatrixBounds {
  using Transpose_t = MatrixBounds<Col0, Row0, NumCols, NumRows>;

  constexpr MatrixBounds() = default;

  constexpr MatrixBounds(std::size_t row0, std::size_t col0,
                         std::size_t numRows, std::size_t numCols)
      : m_row0(row0), m_col0(col0), m_num_rows(numRows), m_num_cols(numCols) {}

  constexpr static std::size_t row0_compile_time = Row0;
  constexpr static std::size_t col0_compile_time = Col0;
  constexpr static std::size_t num_rows_compile_time = NumRows;
  constexpr static std::size_t num_cols_compile_time = NumCols;
  constexpr static std::size_t row_end_compile_time =
      Row0 != Dynamic && NumRows != Dynamic ? Row0 + NumRows : Dynamic;
  constexpr static std::size_t col_end_compile_time =
      Col0 != Dynamic && NumCols != Dynamic ? Col0 + NumCols : Dynamic;

  [[nodiscard]] constexpr static bool is_static() {
    return Row0 != Dynamic && Col0 != Dynamic && NumRows != Dynamic &&
           NumCols != Dynamic;
  }
  [[nodiscard]] constexpr static bool is_vector() {
    return NumRows == 1 || NumCols == 1;
  }
  [[nodiscard]] constexpr std::size_t vector_size() {
    static_assert(is_vector(), "Not a vector");
    return std::max(NumRows, NumCols);
  }

  [[nodiscard]] constexpr std::size_t row0() const {
    return row0_compile_time != Dynamic ? row0_compile_time : m_row0;
  }
  [[nodiscard]] constexpr std::size_t col0() const {
    return col0_compile_time != Dynamic ? col0_compile_time : m_col0;
  }
  [[nodiscard]] constexpr std::size_t row_end() const {
    return row_end_compile_time != Dynamic ? row_end_compile_time
                                           : m_row0 + m_num_rows;
  }
  [[nodiscard]] constexpr std::size_t col_end() const {
    return col_end_compile_time != Dynamic ? col_end_compile_time
                                           : m_col0 + m_num_cols;
  }
  [[nodiscard]] constexpr std::size_t num_rows() const {
    return num_rows_compile_time != Dynamic ? num_rows_compile_time
                                            : m_num_rows;
  }
  [[nodiscard]] constexpr std::size_t num_cols() const {
    return num_cols_compile_time != Dynamic ? num_cols_compile_time
                                            : m_num_cols;
  }

 private:
  std::size_t m_row0 = Row0;
  std::size_t m_col0 = Col0;
  std::size_t m_num_rows = NumRows;
  std::size_t m_num_cols = NumCols;
};

namespace details {

template <std::size_t T1, std::size_t T2>
inline constexpr std::size_t max_start_compile_time =
    T1 != Dynamic && T2 != Dynamic ? std::max(T1, T2) : Dynamic;

template <std::size_t T1, std::size_t T2>
inline constexpr std::size_t min_end_compile_time =
    T1 != Dynamic && T2 != Dynamic ? std::min(T1, T2) : Dynamic;

template <std::size_t T1, std::size_t T2>
constexpr std::size_t max_start(std::size_t t1, std::size_t t2) {
  if constexpr (T1 != Dynamic && T2 != Dynamic) {
    return max_start_compile_time<T1, T2>;
  } else {
    return std::max(t1, t2);
  }
}

template <std::size_t T1, std::size_t T2>
constexpr std::size_t min_end(std::size_t t1, std::size_t t2) {
  if constexpr (T1 != Dynamic && T2 != Dynamic) {
    return min_end_compile_time<T1, T2>;
  } else {
    return std::min(t1, t2);
  }
}

}  // namespace details

/*  This operation computes the overlapping region between two submatrices (or
 * bounds) where both operands and the result are referenced to the original
 * matrix. */
template <typename Bounds1, typename Bounds2>
constexpr decltype(auto) intersect(const Bounds1& b1, const Bounds2& b2) {
  constexpr std::size_t row0_compile_time =
      details::max_start_compile_time<Bounds1::row0_compile_time,
                                      Bounds2::row0_compile_time>;
  constexpr std::size_t col0_compile_time =
      details::max_start_compile_time<Bounds1::col0_compile_time,
                                      Bounds2::col0_compile_time>;
  constexpr std::size_t row_end_compile_time =
      details::min_end_compile_time<Bounds1::row_end_compile_time,
                                    Bounds2::row_end_compile_time>;
  constexpr std::size_t col_end_compile_time =
      details::min_end_compile_time<Bounds1::col_end_compile_time,
                                    Bounds2::col_end_compile_time>;

  const std::size_t row0 =
      details::max_start<Bounds1::row0_compile_time,
                         Bounds2::row0_compile_time>(b1.row0(), b2.row0());
  const std::size_t col0 =
      details::max_start<Bounds1::col0_compile_time,
                         Bounds2::col0_compile_time>(b1.col0(), b2.col0());
  const std::size_t numRows = details::min_end<Bounds1::row_end_compile_time,
                                               Bounds2::row_end_compile_time>(
                                  b1.row_end(), b2.row_end()) -
                              row0;
  const std::size_t numCols = details::min_end<Bounds1::col_end_compile_time,
                                               Bounds2::col_end_compile_time>(
                                  b1.col_end(), b2.col_end()) -
                              col0;

  return MatrixBounds<row0_compile_time, col0_compile_time,
                      row_end_compile_time - row0_compile_time,
                      col_end_compile_time - col0_compile_time>(
      row0, col0, numRows, numCols);
}

/* This operation computes the overlapping region between an outer submatrix (or
 * bounds) and an inner submatrix (or bounds) where the inner submatrix is
 * specified relative to the outer submatrix, and the result is referenced to
 * the original matrix. */
template <typename Bounds1, typename Bounds2>
constexpr decltype(auto) submatrix_intersect(const Bounds1& outer_bounds,
                                             const Bounds2& inner_bounds) {
  const MatrixBounds<Bounds2::row0_compile_time + Bounds1::row0_compile_time,
                     Bounds2::col0_compile_time + Bounds1::col0_compile_time,
                     Bounds2::num_rows_compile_time,
                     Bounds2::num_cols_compile_time>
      inner_bounds_relative_to_base(inner_bounds.row0() + outer_bounds.row0(),
                                    inner_bounds.col0() + outer_bounds.col0(),
                                    inner_bounds.num_rows(),
                                    inner_bounds.num_cols());
  return intersect(outer_bounds, inner_bounds_relative_to_base);
}

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