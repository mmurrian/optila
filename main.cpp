#include <cassert>

#include "optila.h"

int assignment_tests() {
  optila::Matrix<double, optila::Dynamic, 3> A =
      optila::Matrix<double, 3, 3>{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  optila::Matrix<double, 3, 3> B = A;

  optila::Matrix<double, 3, 3> C = A * B;

  return 0;
}

int other_tests() {
  // Dynamically allocated matrices
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> B{
      {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);
  assert(B.num_rows() == 3);
  assert(B.num_cols() == 3);

  auto C = optila::evaluate(A + B);
  auto D = optila::evaluate(A - B);
  auto E = optila::evaluate(A * B);
  auto scalar = optila::Scalar<double>{30.0};

  auto F = optila::evaluate(scalar * A);
  auto G = optila::evaluate(A * scalar);

  assert(C == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}}}));
  assert(D == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{-8, -6, -4}, {-2, 0, 2}, {4, 6, 8}}}));
  assert(E == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}}));
  assert(F == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}}));
  assert(G == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}}));

  // Statically-known matrices
  static constexpr auto static_A =
      optila::make_matrix<double, 3, 3>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  static constexpr auto static_B =
      optila::make_matrix<double, 3, 3>({{9, 8, 7}, {6, 5, 4}, {3, 2, 1}});

  static_assert(static_A.num_rows() == 3);
  static_assert(static_A.num_cols() == 3);
  static_assert(static_B.num_rows() == 3);
  static_assert(static_B.num_cols() == 3);

  static constexpr auto static_C = optila::evaluate(static_A + static_B);
  static constexpr auto static_D = optila::evaluate(static_A - static_B);
  static constexpr auto static_E = optila::evaluate(static_A * static_B);
  static constexpr auto static_scalar = optila::make_scalar<double>(30.0);

  static constexpr auto static_F = optila::evaluate(static_scalar * static_A);
  static constexpr auto static_G = optila::evaluate(static_A * static_scalar);

  static_assert(static_C ==
                decltype(static_C){{{10, 10, 10}, {10, 10, 10}, {10, 10, 10}}});
  static_assert(static_D ==
                decltype(static_D){{{-8, -6, -4}, {-2, 0, 2}, {4, 6, 8}}});
  static_assert(static_E == decltype(static_E){
                                {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}});

  static_assert(
      static_F ==
      decltype(static_F){{{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}});
  static_assert(
      static_G ==
      decltype(static_G){{{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}});

  return 0;
}

int dynamic_tests() {
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  optila::Matrix<double, optila::Dynamic, optila::Dynamic> b{{{1}, {2}, {3}}};
  assert(b.num_rows() == 3);
  assert(b.num_cols() == 1);

  auto C = A * b;

  assert(C.num_rows() == 3);
  assert(C.num_cols() == 1);

  auto D = optila::evaluate(C);
  assert(D(0, 0) == 14);
  assert(D(1, 0) == 32);
  assert(D(2, 0) == 50);

  auto E = optila::evaluate(optila::dot(b, b));
  assert(E == 14);

  auto F = optila::evaluate(C + b);
  assert(F(0, 0) == 15);

  auto G = optila::submatrix<1, 1, 2, 2>(A);
  assert(G.num_rows() == 2);
  assert(G.num_cols() == 2);

  auto H = optila::evaluate(G);
  assert(H(0, 0) == 5);
  assert(H(0, 1) == 6);
  assert(H(1, 0) == 8);
  assert(H(1, 1) == 9);

  return 0;
}

int constexpr_tests() {
  static constexpr optila::Matrix<double, 3, 3> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  // Tests deduction guide for static matrices. Note that this constructs a
  // matrix of ints.
  static constexpr optila::Matrix b({{1}, {2}, {3}});
  // Tests static_convert from int to double.
  static constexpr auto C = A * optila::static_convert<double>(b);

  static_assert(C.num_rows() == 3);
  static_assert(C.num_cols() == 1);

  static constexpr auto D = optila::evaluate(C);
  static_assert(D(0, 0) == 14);
  static_assert(D(1, 0) == 32);
  static_assert(D(2, 0) == 50);

  static constexpr auto E = optila::evaluate(optila::dot(b, b));
  static_assert(E == 14);

  static constexpr auto F =
      optila::evaluate(C + optila::static_convert<double>(b));
  static_assert(F(0, 0) == 15);

  static constexpr auto G = optila::submatrix<1, 1, 2, 2>(A);
  static_assert(G.num_rows() == 2);
  static_assert(G.num_cols() == 2);

  static constexpr auto H = optila::evaluate(G);
  static_assert(H(0, 0) == 5);
  static_assert(H(0, 1) == 6);
  static_assert(H(1, 0) == 8);
  static_assert(H(1, 1) == 9);

  return 0;
}

int main(int argc, char *argv[]) {
  int retval = 0;
  retval -= dynamic_tests();
  retval -= constexpr_tests();
  retval -= other_tests();
  retval -= assignment_tests();
  return retval;
}