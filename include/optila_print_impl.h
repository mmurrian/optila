#pragma once

#include <iostream>
#include <type_traits>

#include "optila_expression_impl.h"
#include "optila_matrix_impl.h"
#include "optila_operation_impl.h"
#include "optila_scalar_impl.h"

namespace optila {

namespace Operation {
template <typename Op>
const char* operation_to_symbol();

template <>
inline const char* operation_to_symbol<ScalarAddition>() {
  return u8"\u002B";
}

template <>
inline const char* operation_to_symbol<Addition>() {
  return u8"\u002B";
}

template <>
inline const char* operation_to_symbol<ScalarSubtraction>() {
  return u8"\u2212";
}

template <>
inline const char* operation_to_symbol<Subtraction>() {
  return u8"\u2212";
}

template <>
inline const char* operation_to_symbol<Multiplication>() {
  return u8"\u00D7";
}

template <>
inline const char* operation_to_symbol<ScalarMultiplication>() {
  return u8"\u2219";
}

template <>
inline const char* operation_to_symbol<MatrixScalarDivision>() {
  return u8"\u00F7";
}

template <>
inline const char* operation_to_symbol<ScalarDivision>() {
  return u8"\u00F7";
}

template <>
inline const char* operation_to_symbol<DotProduct>() {
  return u8"\u22C5";
}

template <>
inline const char* operation_to_symbol<CrossProduct>() {
  return u8"\u2A2F";
}

template <>
inline const char* operation_to_symbol<Transpose>() {
  return u8"\u1D40";
}

template <typename Op>
int operation_precedence();

template <>
inline int operation_precedence<ScalarAddition>() {
  return 1;
}

template <>
inline int operation_precedence<Addition>() {
  return 1;
}

template <>
inline int operation_precedence<ScalarSubtraction>() {
  return 1;
}

template <>
inline int operation_precedence<Subtraction>() {
  return 1;
}

template <>
inline int operation_precedence<Multiplication>() {
  return 2;
}

template <>
inline int operation_precedence<ScalarMultiplication>() {
  return 2;
}

template <>
inline int operation_precedence<MatrixScalarDivision>() {
  return 2;
}

template <>
inline int operation_precedence<ScalarDivision>() {
  return 2;
}

template <>
inline int operation_precedence<DotProduct>() {
  return 2;
}

template <>
inline int operation_precedence<CrossProduct>() {
  return 2;
}

template <>
inline int operation_precedence<Transpose>() {
  return 3;
}

};  // namespace Operation

template <typename Op, typename... Operands>
void print_expression(const Expression<Op, Operands...>& expr, std::ostream& os,
                      int parent_precedence, std::index_sequence<0>) {
  int precendence = Operation::operation_precedence<Op>();
  bool needs_parentheses = precendence > 0 && parent_precedence >= precendence;

  if (needs_parentheses) {
    os << "(";
  }

  print_expression(std::get<0>(expr.operands()), os, precendence);

  if (needs_parentheses) {
    os << ")";
  }

  os << Operation::operation_to_symbol<Op>();
}

template <typename Op, typename... Operands, std::size_t... Is>
void print_expression(const Expression<Op, Operands...>& expr, std::ostream& os,
                      int parent_precedence, std::index_sequence<Is...>) {
  int precendence = Operation::operation_precedence<Op>();
  bool needs_parentheses = precendence > 0 && parent_precedence >= precendence;

  if (needs_parentheses) {
    os << "(";
  }

  ((os << (Is == 0 ? "" : Operation::operation_to_symbol<Op>()),
    print_expression(std::get<Is>(expr.operands()), os, precendence)),
   ...);

  if (needs_parentheses) {
    os << ")";
  }
}

template <typename Op, typename... Operands>
void print_expression(const Expression<Op, Operands...>& expr, std::ostream& os,
                      int parent_precedence = 0) {
  print_expression(expr, os, parent_precedence,
                   std::make_index_sequence<sizeof...(Operands)>());
}

template <typename ValueType>
void print_expression(const Scalar<ValueType>& expr, std::ostream& os,
                      int /*parent_precedence*/ = 0) {
  os << expr();
}

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
void print_expression(const Matrix<ValueType, NumRows, NumCols>& expr,
                      std::ostream& os, int /*parent_precedence*/ = 0) {
  os << "[";
  for (std::size_t i = 0; i < expr.num_rows(); ++i) {
    os << "[";
    for (std::size_t j = 0; j < expr.num_cols(); ++j) {
      os << expr(i, j) << ((j < expr.num_cols() - 1) ? ", " : "");
    }
    os << "]" << ((i < expr.num_rows() - 1) ? ", " : "");
  }
  os << "]";
}

template <typename Op, typename... Operands>
std::ostream& operator<<(std::ostream& os,
                         const Expression<Op, Operands...>& expr) {
  print_expression(expr, os);
  return os;
}

template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const Scalar<ValueType>& expr) {
  print_expression(expr, os);
  return os;
}

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
std::ostream& operator<<(std::ostream& os,
                         const Matrix<ValueType, NumRows, NumCols>& expr) {
  print_expression(expr, os);
  return os;
}

}  // namespace optila
