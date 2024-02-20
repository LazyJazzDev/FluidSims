#include <utility>

#include "core/core.cuh"
#include "core/device_clock.cuh"

FluidInterface *CreateFluidLogicInstance(const SimSettings &sim_settings) {
  return new FluidCore(sim_settings);
}

FluidCore::FluidCore(SimSettings sim_settings)
    : sim_settings_(std::move(sim_settings)) {
  grid_cell_header_.delta_x = sim_settings_.delta_x;
  grid_cell_header_.origin = glm::vec3{0.0f};
  grid_cell_header_.size = sim_settings_.grid_size;
  grid_center_header_.delta_x = sim_settings_.delta_x;
  grid_center_header_.size = sim_settings_.grid_size;
  grid_center_header_.origin = glm::vec3{0.5f * grid_center_header_.delta_x};
  grid_point_header_ = grid_cell_header_;
  grid_point_header_.size += 1;

  pressure_ = Grid<float>(grid_center_header_);
  level_set_ = Grid<float>(grid_center_header_);
  velocity_ = MACGrid<float>(grid_cell_header_);
  mass_sample_ = MACGrid<float>(grid_cell_header_);
}

__global__ void SetParticlesKernel(Particle *particles,
                                   const glm::vec3 *particle_positions,
                                   int num_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_particles) {
    particles[i].position = particle_positions[i];
  }
}

void FluidCore::SetParticles(const std::vector<glm::vec3> &positions) {
  thrust::device_vector<glm::vec3> dev_particle_pos = positions;
  particles_.resize(positions.size());
  SetParticlesKernel<<<CALL_SHAPE(positions.size())>>>(
      thrust::raw_pointer_cast(particles_.data()),
      thrust::raw_pointer_cast(dev_particle_pos.data()), positions.size());
}

struct ParticleToVec3 {
  __host__ __device__ glm::vec3 operator()(const Particle &p) {
    return p.position;
  }
};

std::vector<glm::vec3> FluidCore::GetParticles() const {
  thrust::device_vector<glm::vec3> dev_res(particles_.size());

  thrust::transform(particles_.begin(), particles_.end(), dev_res.begin(),
                    ParticleToVec3());
  std::vector<glm::vec3> res(particles_.size());
  thrust::copy(dev_res.begin(), dev_res.end(), res.begin());
  return res;
}

__global__ void AdvectionKernel(Particle *particles,
                                int num_particles,
                                float delta_time) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_particles) {
    Particle &p = particles[i];
    p.position += p.velocity * delta_time;
  }
}

__global__ void ApplyGravityKernel(Particle *particles,
                                   int num_particles,
                                   float delta_time,
                                   glm::vec3 gravity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_particles) {
    particles[i].velocity += gravity * delta_time;
  }
}

void FluidCore::Update(float delta_time) {
  DeviceClock device_clock;

  pressure_.Clear();
  level_set_.Clear(-1.0f);
  velocity_.Clear();
  mass_sample_.Clear();
  device_clock.Record("Clear");

  AdvectionKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), particles_.size(), delta_time);
  device_clock.Record("Advection");

  ApplyGravityKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), particles_.size(), delta_time,
      glm::vec3{0.0f, -9.8f, 0.0f});
  device_clock.Record("Apply Gravity");

  device_clock.Finish();
}
