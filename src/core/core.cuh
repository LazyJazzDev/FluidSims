#pragma once
#include "core/grids.cuh"
#include "core/interface.h"
#include "core/views.cuh"

#define BLOCK_SIZE 256
#define CALL_SHAPE(x) ((x) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
};

class FluidCore : public FluidInterface {
 public:
  explicit FluidCore(SimSettings sim_settings);
  void SetParticles(const std::vector<glm::vec3> &positions) override;
  [[nodiscard]] std::vector<glm::vec3> GetParticles() const override;
  void Update(float delta_time) override;

 private:
  SimSettings sim_settings_;
  glm::ivec3 grid_size_;
  GridHeader grid_center_header_;
  GridHeader grid_point_header_;
  GridHeader grid_cell_header_;

  Grid<float> pressure_;
  Grid<float> level_set_;
  MACGrid<float> velocity_;
  MACGrid<float> mass_sample_;

  thrust::device_vector<Particle> particles_;
};
