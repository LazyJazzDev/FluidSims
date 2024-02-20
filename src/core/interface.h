#pragma once
#include "glm/glm.hpp"
#include "vector"

class FluidInterface {
 public:
  virtual void SetParticles(const std::vector<glm::vec3> &particles) = 0;
  [[nodiscard]] virtual std::vector<glm::vec3> GetParticles() const = 0;
  virtual void Update(float delta_time) = 0;
};

FluidInterface *CreateFluidLogicInstance(const glm::ivec3 &grid_size);
