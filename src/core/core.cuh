#pragma once

#include "Eigen/Eigen"
#include "core/grids.cuh"
#include "core/interface.h"
#include "core/linear_solvers.cuh"
#include "core/vector.cuh"

#define BLOCK_SIZE 256
#define CALL_SHAPE(x) ((x) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
  glm::mat3 C;
};

struct AdjacentInfo {
  float local{};
  float edge[3][2]{};

  friend std::ostream &operator<<(std::ostream &os, const AdjacentInfo &info) {
    os << info.local << ";";
    for (int dim = 0; dim < 3; dim++) {
      for (int edge = 0; edge < 2; edge++) {
        os << info.edge[dim][edge] << ";";
      }
    }
    return os;
  }
};

struct AdjacentOp : public LinearOp, public JacobiOp {
  AdjacentOp(Grid<AdjacentInfo> &grid) : adjacent_info(grid.View()) {
  }

  AdjacentOp(GridView<AdjacentInfo> view) : adjacent_info(view) {
  }

  void operator()(VectorView<float> x, VectorView<float> y) override;

  void LU(VectorView<float> x, VectorView<float> y) override;

  void D_inv(VectorView<float> x, VectorView<float> y) override;

  GridView<AdjacentInfo> adjacent_info;
};

struct FluidOperator : public JacobiOp, public MultiGridPCGOp {
  FluidOperator(GridHeader center_header = GridHeader{}) {
    adjacent_info = Grid<AdjacentInfo>(center_header);
    rigid_J = Grid<Eigen::Vector<float, 6>>(center_header);
  }

  Grid<AdjacentInfo> adjacent_info;

  std::vector<Grid<AdjacentInfo>> down_sampled_adjacent_info;

  Grid<Eigen::Vector<float, 6>> rigid_J;

  void operator()(VectorView<float> x, VectorView<float> y) override;

  void LU(VectorView<float> x, VectorView<float> y) override;

  void D_inv(VectorView<float> x, VectorView<float> y) override;

  std::vector<MultiGridLevel> MultiGridLevels(int iterations) override;
};

struct RigidInfo {
  float scale_{0.0f};
  float mass_{1.0f};
  glm::mat3 inertia_{1.0f};
  glm::mat3 rotation_{1.0f};
  glm::vec3 offset_{0.0f};
  glm::vec3 velocity_{0.0f};
  glm::vec3 angular_velocity_{0.0f};

  __device__ __host__ glm::mat4 GetAffineMatrix() const {
    auto R = rotation_ * scale_;
    return glm::mat4{R[0][0],   R[0][1],   R[0][2],   0.0f,    R[1][0], R[1][1],
                     R[1][2],   0.0f,      R[2][0],   R[2][1], R[2][2], 0.0f,
                     offset_.x, offset_.y, offset_.z, 1.0f};
  }

  __device__ __host__ glm::vec3 LinearVelocity(
      const glm::vec3 &position) const {
    return velocity_ + glm::cross(angular_velocity_, position - offset_);
  }
};

class FluidCore : public FluidInterface {
 public:
  explicit FluidCore(SimSettings sim_settings);

  void SetParticles(const std::vector<glm::vec3> &positions) override;

  [[nodiscard]] std::vector<glm::vec3> GetParticles() const override;

  void SetCube(const glm::vec3 &position,
               float size,
               const glm::mat3 &rotation,
               float mass,
               const glm::vec3 &velocity = glm::vec3{0.0f},
               const glm::vec3 &angular_velocity = glm::vec3{0.0f}) override;

  glm::mat4 GetCube() const override;

  void Update(float delta_time) override;

 private:
  void SubStep(float delta_time);

  SimSettings sim_settings_;
  glm::ivec3 grid_size_;
  GridHeader grid_center_header_;
  GridHeader grid_point_header_;
  GridHeader grid_cell_header_;

  Grid<float> pressure_;
  Grid<float> level_set_;

  Grid<float> b_;
  FluidOperator operator_;

  MACGrid<float> transfer_weight_;
  MACGrid<float> velocity_;
  MACGrid<float> velocity_bak_;
  MACGrid<float> fluid_mass_;
  MACGrid<float> rigid_volume_;
  MACGrid<bool> valid_sample_;
  MACGrid<bool> valid_sample_bak_;

  thrust::device_vector<Particle> particles_;

  RigidInfo rigid_info_;
  glm::vec3 gravity_{0.0f, -9.8f, 0.0f};
};
