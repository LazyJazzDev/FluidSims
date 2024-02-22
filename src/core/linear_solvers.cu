#include "core/linear_solvers.cuh"

void ConjugateGradient(LinearOp &A, VectorView<float> b, VectorView<float> x) {
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

void Jacobi(JacobiOp &A,
            VectorView<float> b,
            VectorView<float> x,
            int iterations) {
  Vector<float> buffer_(b.size);
  for (int i = 0; i < iterations; i++) {
    A.LU(x, buffer_);
    Sub(b, buffer_.View(), buffer_.View());
    A.D_inv(buffer_, x);
  }
}

void MultiGridCore(VectorView<float> x,
                   VectorView<float> b,
                   const std::vector<MultiGridLevel> &levels,
                   int iterations,
                   int level_index = 0) {
  auto &level = levels[level_index];
  for (int i = 0; i < iterations; i++) {
    level.pre_smooth(b, x);
  }
  if (level_index < levels.size() - 1) {
    Vector<float> residual(b.size);
    level.linear_op(x, residual);
    Sub(b, residual.View(), residual.View());
    auto down_sampled_residual = level.down_sample(residual);
    Vector<float> error(down_sampled_residual.Size());
    MultiGridCore(error.View(), down_sampled_residual.View(), levels,
                  iterations, level_index + 1);
    auto up_sampled_error = level.up_sample(error.View());
    Add(x, up_sampled_error.View(), x);
  }
  for (int i = 0; i < iterations; i++) {
    level.post_smooth(b, x);
  }
}

void MultiGrid(MultiGridOp &A,
               VectorView<float> b,
               VectorView<float> x,
               int iterations) {
  auto levels = A.MultiGridLevels();
  MultiGridCore(x, b, levels, iterations);
}
