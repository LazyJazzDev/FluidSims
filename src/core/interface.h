#pragma once
#include "glm/glm.hpp"
#include "vector"

struct SimSettings {
  glm::ivec3 grid_size{128};
  float delta_x{1.0f / 128.0f};
  float delta_t{1.5e-3f};
  float rho{1.0f};
  bool alternative_time_step{false};
};

class FluidInterface {
 public:
  virtual void SetParticles(const std::vector<glm::vec3> &particles) = 0;
  [[nodiscard]] virtual std::vector<glm::vec3> GetParticles() const = 0;
  virtual void Update(float delta_time) = 0;
};

FluidInterface *CreateFluidLogicInstance(const SimSettings &sim_settings);
