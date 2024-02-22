#pragma once
#include "core/vector.cuh"

template <class LinearOp>
void ConjugateGradient(LinearOp A, VectorView<float> b, VectorView<float> x) {
  Vector<float> r(b.size);
  Vector<float> buffer_(b.size);
  Vector<float> Ap(b.size);
  A(x, buffer_);              // buffer_ = Ax
  Sub<float>(b, buffer_, r);  // r = b - Ax
  Vector<float> p = r;
  float r_sqr = r.View().NormSqr();

  int cnt = 0;

  while (true) {
    A(p, Ap);
    float alpha = r_sqr / Dot<float>(p, Ap, buffer_);
    Mul<float>(p, alpha, buffer_);  // buffer_ = alpha * p
    Add<float>(x, buffer_, x);      // x = x + alpha * p

    cnt++;

    Mul<float>(Ap, alpha, buffer_);  // buffer_ = alpha * Ap
    Sub<float>(r, buffer_, r);       // r = r - alpha * Ap

    float r_sqr_new = r.View().NormSqr();

    if (r_sqr_new < 1e-16f * float(b.size)) {
      printf("CG solved in %d steps\n", cnt);
      break;
    }

    float beta = r_sqr_new / r_sqr;
    r_sqr = r_sqr_new;
    Mul<float>(p, beta, buffer_);  // buffer_ = beta * p
    Add<float>(r, buffer_, p);     // p = r + beta * p
  }
}
