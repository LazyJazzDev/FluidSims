#pragma once
#include "core/vector.cuh"

struct LinearOp {
  virtual void operator()(VectorView<float> x, VectorView<float> y) = 0;
};

struct JacobiOp : public LinearOp {
  virtual void LU(VectorView<float> x, VectorView<float> y) = 0;
  virtual void D_inv(VectorView<float> x, VectorView<float> y) = 0;
};

void ConjugateGradient(LinearOp &A, VectorView<float> b, VectorView<float> x);

void Jacobi(JacobiOp &A,
            VectorView<float> b,
            VectorView<float> x,
            int iterations = 128);
