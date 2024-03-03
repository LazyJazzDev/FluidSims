#pragma once
#include "glm/glm.hpp"
#include "vector"

struct SimSettings {
  glm::ivec3 grid_size{50};
  float delta_x{1.0f / 50.0f};
  float delta_t{0.003f};
  float rho{1.0f};
  bool alternative_time_step{true};
  bool apic{true};
  bool sorting_p2g{true};
};

class FluidInterface {
 public:
  virtual void SetParticles(const std::vector<glm::vec3> &particles) = 0;
  [[nodiscard]] virtual std::vector<glm::vec3> GetParticles() const = 0;

  virtual void SetCube(const glm::vec3 &position,
                       float size = 1.0f,
                       const glm::mat3 &rotation = glm::mat3{1.0f},
                       float mass = 1.0f,
                       const glm::vec3 &velocity = glm::vec3{0.0f},
                       const glm::vec3 &angular_velocity = glm::vec3{0.0f}) = 0;

  [[nodiscard]] virtual glm::mat4 GetCube() const = 0;

  virtual void Update(float delta_time) = 0;
};

FluidInterface *CreateFluidLogicInstance(const SimSettings &sim_settings);
