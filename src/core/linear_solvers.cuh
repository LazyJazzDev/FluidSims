#pragma once
#include "core/vector.cuh"

struct LinearOp {
  virtual void operator()(VectorView<float> x, VectorView<float> y) = 0;
};

struct JacobiOp {
  virtual void LU(VectorView<float> x, VectorView<float> y) = 0;
  virtual void D_inv(VectorView<float> x, VectorView<float> y) = 0;
};

struct MultiGridLevel {
  std::function<void(VectorView<float>, VectorView<float>)> pre_smooth;
  std::function<void(VectorView<float>, VectorView<float>)> post_smooth;
  std::function<void(VectorView<float>, VectorView<float>)> linear_op;
  std::function<Vector<float>(VectorView<float>)> down_sample;
  std::function<Vector<float>(VectorView<float>)> up_sample;
};

struct MultiGridOp {
  virtual std::vector<MultiGridLevel> MultiGridLevels() = 0;
};

void ConjugateGradient(LinearOp &A, VectorView<float> b, VectorView<float> x);

void Jacobi(JacobiOp &A,
            VectorView<float> b,
            VectorView<float> x,
            int iterations = 128);

void MultiGrid(MultiGridOp &A,
               VectorView<float> b,
               VectorView<float> x,
               int iterations = 10);
