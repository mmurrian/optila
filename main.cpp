#include <cassert>

#include "optila.h"

int dynamic_tests() {
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  optila::Matrix<double, optila::Dynamic, optila::Dynamic> b{{1}, {2}, {3}};
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
  static constexpr optila::Matrix<double, 3, 1> b{{1}, {2}, {3}};
  static constexpr auto C = A * b;

  static_assert(C.num_rows() == 3);
  static_assert(C.num_cols() == 1);

  static constexpr auto D = optila::evaluate(C);
  static_assert(D(0, 0) == 14);
  static_assert(D(1, 0) == 32);
  static_assert(D(2, 0) == 50);

  static constexpr auto E = optila::evaluate(optila::dot(b, b));
  static_assert(E == 14);

  static constexpr auto F = optila::evaluate(C + b);
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
  return retval;
}