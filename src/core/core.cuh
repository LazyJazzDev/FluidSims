#pragma once

#include "core/grids.cuh"
#include "core/interface.h"
#include "core/linear_solvers.cuh"
#include "core/vector.cuh"

#define BLOCK_SIZE 256
#define CALL_SHAPE(x) ((x) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
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
  }

  Grid<AdjacentInfo> adjacent_info;

  std::vector<Grid<AdjacentInfo>> down_sampled_adjacent_info;

  void operator()(VectorView<float> x, VectorView<float> y) override;

  void LU(VectorView<float> x, VectorView<float> y) override;

  void D_inv(VectorView<float> x, VectorView<float> y) override;

  std::vector<MultiGridLevel> MultiGridLevels(int iterations) override;
};

class FluidCore : public FluidInterface {
 public:
  explicit FluidCore(SimSettings sim_settings);

  void SetParticles(const std::vector<glm::vec3> &positions) override;

  [[nodiscard]] std::vector<glm::vec3> GetParticles() const override;

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
  MACGrid<float> transfer_weight_bak_;
  MACGrid<float> velocity_;
  MACGrid<float> velocity_bak_;
  MACGrid<float> mass_sample_;
  MACGrid<bool> valid_sample_;

  thrust::device_vector<Particle> particles_;
};
