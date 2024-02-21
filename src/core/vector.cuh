#pragma once

#include "core/grids.cuh"
#include "thrust/device_vector.h"

template <class Ty>
class Vector;

template <class Ty>
struct SqaureOp {
  __device__ Ty operator()(const Ty &a) {
    return a * a;
  }
};

template <class Ty>
struct VectorView {
  Ty *buffer;
  size_t size;
  VectorView(Vector<Ty> &vec);

  VectorView(Grid<Ty> &grid) {
    buffer = grid.Buffer().get();
    size = grid.Header().TotalCells();
  }

  friend std::ostream &operator<<(std::ostream &os, const VectorView<Ty> &vec) {
    thrust::host_vector<Ty> h_vec(vec.size);
    thrust::copy(thrust::device_pointer_cast(vec.buffer),
                 thrust::device_pointer_cast(vec.buffer + vec.size),
                 h_vec.begin());
    os << "[";
    for (auto &val : h_vec) {
      os << val << ", ";
    }
    os << "]";
    return os;
  }

  Ty NormSqr() const {
    return thrust::transform_reduce(thrust::device_pointer_cast(buffer),
                                    thrust::device_pointer_cast(buffer + size),
                                    SqaureOp<Ty>{}, Ty{}, thrust::plus<Ty>());
  }
};

template <class Ty>
class Vector {
 public:
  Vector() = default;
  Vector(size_t size, Ty value = Ty()) : buffer_(size, value) {
  }

  Vector(const VectorView<Ty> &view) {
    buffer_ = thrust::device_vector<Ty>(view.buffer, view.buffer + view.size);
  }

  thrust::device_ptr<Ty> Buffer() {
    return buffer_.data();
  }

  [[nodiscard]] size_t Size() const {
    return buffer_.size();
  }

  VectorView<Ty> View() {
    return VectorView<Ty>(*this);
  }

  friend std::ostream &operator<<(std::ostream &os, const Vector<Ty> &vec) {
    thrust::host_vector<Ty> h_vec(vec.buffer_);
    os << "[";
    for (auto &val : h_vec) {
      os << val << ", ";
    }
    os << "]";
    return os;
  }

 private:
  thrust::device_vector<Ty> buffer_;
};

template <class Ty>
VectorView<Ty>::VectorView(Vector<Ty> &vec)
    : buffer(thrust::raw_pointer_cast(vec.Buffer())), size(vec.Size()) {
}

template <class Ty>
Vector<Ty> ZerosLike(const VectorView<Ty> &vec) {
  return Vector<Ty>(vec.size, Ty{});
}

template <class Ty>
Vector<Ty> OnesLike(const VectorView<Ty> &vec) {
  return Vector<Ty>(vec.size, Ty{1});
}

template <class Ty>
void Add(VectorView<Ty> a, VectorView<Ty> b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(b.buffer),
                    thrust::device_pointer_cast(res.buffer),
                    thrust::plus<Ty>());
}

template <class Ty>
void Sub(VectorView<Ty> a, VectorView<Ty> b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(b.buffer),
                    thrust::device_pointer_cast(res.buffer),
                    thrust::minus<Ty>());
}

template <class Ty>
void Mul(VectorView<Ty> a, VectorView<Ty> b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(b.buffer),
                    thrust::device_pointer_cast(res.buffer),
                    thrust::multiplies<Ty>());
}

template <class Ty>
void Div(VectorView<Ty> a, VectorView<Ty> b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(b.buffer),
                    thrust::device_pointer_cast(res.buffer),
                    thrust::divides<Ty>());
}

template <class Ty, class Op>
struct ScalarOp {
  Op op;
  Ty scalar;

  ScalarOp(Ty scalar) : scalar(scalar) {
  }

  __device__ Ty operator()(const Ty &a) {
    return op(a, scalar);
  }
};

template <class Ty>
void Add(VectorView<Ty> a, Ty b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(res.buffer),
                    ScalarOp<Ty, thrust::plus<Ty>>{b});
}

template <class Ty>
void Sub(VectorView<Ty> a, Ty b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(res.buffer),
                    ScalarOp<Ty, thrust::minus<Ty>>{b});
}

template <class Ty>
void Mul(VectorView<Ty> a, Ty b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(res.buffer),
                    ScalarOp<Ty, thrust::multiplies<Ty>>{b});
}

template <class Ty>
void Div(VectorView<Ty> a, Ty b, VectorView<Ty> res) {
  thrust::transform(thrust::device_pointer_cast(a.buffer),
                    thrust::device_pointer_cast(a.buffer + a.size),
                    thrust::device_pointer_cast(res.buffer),
                    ScalarOp<Ty, thrust::divides<Ty>>{b});
}

template <class Ty>
Ty Dot(VectorView<Ty> a, VectorView<Ty> b) {
  Vector<Ty> res(a.size);
  Mul<Ty>(a, b, res);
  return thrust::reduce(thrust::device_pointer_cast(res.Buffer()),
                        thrust::device_pointer_cast(res.Buffer() + res.Size()),
                        Ty{}, thrust::plus<Ty>());
}
template <class Ty>
Ty Dot(VectorView<Ty> a, VectorView<Ty> b, VectorView<Ty> buffer_) {
  Mul<Ty>(a, b, buffer_);
  return thrust::reduce(
      thrust::device_pointer_cast(buffer_.buffer),
      thrust::device_pointer_cast(buffer_.buffer + buffer_.size), Ty{},
      thrust::plus<Ty>());
}
