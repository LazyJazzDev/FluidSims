#include <utility>

#include "core/core.cuh"
#include "core/device_clock.cuh"
#include "core/linear_solvers.cuh"

template <class VectorType>
__device__ float KernelFunction(const VectorType &v) {
  auto len = glm::length(v);
  if (len < 0.5f) {
    return 0.75f - len * len;
  } else if (len < 1.5f) {
    return 0.5f * (1.5f - len) * (1.5f - len);
  }
  return 0.0f;
}

template <class VectorType>
__device__ glm::vec3 KernelFunctionGradient(const VectorType &v) {
  auto len = glm::length(v);
  if (len < 1e-6f) {
    return glm::vec3{0.0f};
  } else if (len < 0.5f) {
    return -2.0f * len * glm::normalize(v);
  } else if (len < 1.5f) {
    return -(1.5f - len) * glm::normalize(v);
  }
  return glm::vec3{0.0f};
}

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
  b_ = Grid<float>(grid_center_header_);
  operator_ = FluidOperator(grid_center_header_);

  velocity_ = MACGrid<float>(grid_cell_header_);
  transfer_weight_ = MACGrid<float>(grid_cell_header_);
  velocity_bak_ = MACGrid<float>(grid_cell_header_);
  transfer_weight_bak_ = MACGrid<float>(grid_cell_header_);
  valid_sample_ = MACGrid<bool>(grid_cell_header_);

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

__global__ void AssignCellIndicesKernel(const Particle *particle,
                                        GridHeader cell_header,
                                        int32_t *cell_indices,
                                        int num_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_particles) {
    auto grid_pos = cell_header.World2Grid(particle[i].position);
    int32_t res_ind;
    if (grid_pos.x < 0.0f || grid_pos.x >= float(cell_header.size.x) ||
        grid_pos.y < 0.0f || grid_pos.y >= cell_header.size.y ||
        grid_pos.z < 0.0f || grid_pos.z >= cell_header.size.z) {
      res_ind = -1;
    } else {
      auto index3 = glm::min(glm::max(glm::ivec3(grid_pos), glm::ivec3{0}),
                             cell_header.size - 1);
      res_ind = cell_header.Index(grid_pos);
    }
    cell_indices[i] = res_ind;
  }
}

__global__ void BinarySearchBoundsKernel(int *cell_indices,
                                         int num_particles,
                                         int *lower_bound_indices,
                                         int num_cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= num_cells) {
    int L = 0, R = num_particles;
    while (L < R) {
      int M = (L + R) / 2;
      if (cell_indices[M] < i) {
        L = M + 1;
      } else {
        R = M;
      }
    }
    lower_bound_indices[i] = L;
  }
}

__global__ void ConstructLevelSetKernel(const Particle *particles,
                                        const int *lower_bound_indices,
                                        GridHeader center_header,
                                        float *level_set) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < center_header.TotalCells()) {
    auto index3 = center_header.Index(i);
    float accum_level = 0.0f;
    auto p_pos = glm::vec3{index3};
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          auto neighbor_index3 = index3 + glm::ivec3{x, y, z};
          if (neighbor_index3.x < 0 ||
              neighbor_index3.x >= center_header.size.x ||
              neighbor_index3.y < 0 ||
              neighbor_index3.y >= center_header.size.y ||
              neighbor_index3.z < 0 ||
              neighbor_index3.z >= center_header.size.z) {
            continue;
          }
          int neighbor_index = center_header.Index(neighbor_index3);
          int start = lower_bound_indices[neighbor_index];
          int end = lower_bound_indices[neighbor_index + 1];
          for (int j = start; j < end; j++) {
            auto p_index3 = center_header.World2Grid(particles[j].position);
            auto dist = glm::length(p_pos - p_index3);
            if (dist < 1.0f) {
              accum_level += 1.0f - dist;
            }
          }
        }
      }
    }
    level_set[i] = -1.0f + min(accum_level, 2.0);
  }
}

__global__ void Particle2GridTransferKernel(
    const Particle *particles,
    const int *lower_bound_indices,
    GridHeader center_header,
    GridHeader grid_point_header,
    MACGridView<float> vel_grids,
    MACGridView<float> transfer_weights) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float accum_weights[3]{};
  float accum_weighted_values[3]{};
  if (id < grid_point_header.TotalCells()) {
    auto index3 = grid_point_header.Index(id);
    auto grid_pos = glm::vec3{index3};
    for (int dx = -2; dx <= 1; dx++) {
      for (int dy = -2; dy <= 1; dy++) {
        for (int dz = -2; dz <= 1; dz++) {
          auto neighbor_index3 = index3 + glm::ivec3{dx, dy, dz};
          if (neighbor_index3.x < 0 ||
              neighbor_index3.x >= center_header.size.x ||
              neighbor_index3.y < 0 ||
              neighbor_index3.y >= center_header.size.y ||
              neighbor_index3.z < 0 ||
              neighbor_index3.z >= center_header.size.z) {
            continue;
          }
          int neighbor_index = center_header.Index(neighbor_index3);
          int start = lower_bound_indices[neighbor_index];
          int end = lower_bound_indices[neighbor_index + 1];
          for (int i = start; i < end; i++) {
            auto particle = particles[i];
            for (int j = 0; j < 3; j++) {
              auto diff =
                  vel_grids.grids[j].Header().World2Grid(particle.position) -
                  grid_pos;
              auto weight = KernelFunction(diff);
              if (weight > 0.0f) {
                accum_weights[j] += weight;
                accum_weighted_values[j] += weight * particle.velocity[j];
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < 3; i++) {
      auto vel = 0.0f;
      if (accum_weights[i] > 0.0f) {
        vel = accum_weighted_values[i] / accum_weights[i];
      }
      if (vel_grids.grids[i].LegalIndex(index3)) {
        vel_grids.grids[i](index3) = vel;
        transfer_weights.grids[i](index3) = accum_weights[i];
      }
    }
  }
}

__global__ void ExtrapolateKernel(GridHeader grid_point_header,
                                  MACGridView<float> vel_grids,
                                  MACGridView<float> transfer_weights,
                                  MACGridView<float> vel_grids_new,
                                  MACGridView<float> transfer_weights_new) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < grid_point_header.TotalCells()) {
    auto index3 = grid_point_header.Index(id);
    float neighbor_weights[3]{};
    float neighbor_vels[3]{};
    glm::ivec3 dvs[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                         {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (auto dv : dvs) {
      auto neighbor_index3 = index3 + dv;
      for (int i = 0; i < 3; i++) {
        if (vel_grids.grids[i].LegalIndex(neighbor_index3)) {
          auto neighbor_weight = transfer_weights.grids[i](neighbor_index3);
          auto neighbor_vel = vel_grids.grids[i](neighbor_index3);
          if (neighbor_weight > 0.0f) {
            neighbor_weights[i] += neighbor_weight;
            neighbor_vels[i] += neighbor_weight * neighbor_vel;
          }
        }
      }
    }

    for (int i = 0; i < 3; i++) {
      if (neighbor_weights[i] > 0.0f) {
        if (transfer_weights_new.grids[i].LegalIndex(index3)) {
          if (transfer_weights_new.grids[i](index3) == 0.0f) {
            vel_grids_new.grids[i](index3) =
                neighbor_vels[i] / neighbor_weights[i];
            transfer_weights_new.grids[i](index3) = neighbor_weights[i];
          }
        }
      }
    }
  }
}

__device__ __host__ bool InsideObstacle(const glm::vec3 &pos) {
  return !(pos.x > 0.05f && pos.x < 0.95f && pos.y > 0.05f && pos.y < 0.95f &&
           pos.z > 0.05f && pos.z < 0.95f);
}

__device__ __host__ void UpdateSurfaceInfo(const glm::vec3 &pos,
                                           const glm::vec4 &plane,
                                           float &dist,
                                           glm::vec3 &normal) {
  auto d = glm::dot(glm::vec3{plane}, pos) + plane.w;
  if (d < dist) {
    dist = d;
    normal = glm::vec3{plane};
  }
}

__device__ __host__ glm::vec3 NearestSurfaceNormal(const glm::vec3 &pos) {
  float dist = 1e10f;
  glm::vec3 normal{0, 1, 0};
  UpdateSurfaceInfo(pos, glm::vec4{0, 1, 0, -0.05f}, dist, normal);
  UpdateSurfaceInfo(pos, glm::vec4{0, -1, 0, 0.95f}, dist, normal);
  UpdateSurfaceInfo(pos, glm::vec4{1, 0, 0, -0.05f}, dist, normal);
  UpdateSurfaceInfo(pos, glm::vec4{-1, 0, 0, 0.95f}, dist, normal);
  UpdateSurfaceInfo(pos, glm::vec4{0, 0, 1, -0.05f}, dist, normal);
  UpdateSurfaceInfo(pos, glm::vec4{0, 0, -1, 0.95f}, dist, normal);
  return normal;
}

__global__ void CalculateMassSampleKernel(GridView<float> mass_sample,
                                          float rho,
                                          float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < mass_sample.Header().TotalCells()) {
    auto index3 = mass_sample.Header().Index(id);
    int precision = 10;
    float inv_precision = 1.0f / float(precision);

    int fluid_sample = 0;
    int total_sample = 0;

    for (int dx = 0; dx < precision; dx++) {
      for (int dy = 0; dy < precision; dy++) {
        for (int dz = 0; dz < precision; dz++) {
          glm::vec3 pos =
              glm::vec3{index3} +
              ((glm::vec3{dx, dy, dz} + 0.5f) * inv_precision - 0.5f);
          glm::vec3 world_pos = mass_sample.Header().Grid2World(pos);
          if (!InsideObstacle(world_pos)) {
            fluid_sample++;
          }
          total_sample++;
        }
      }
    }

    mass_sample[id] = rho * delta_x * delta_x * delta_x * float(fluid_sample) /
                      float(total_sample);
  }
}

__device__ void ConstructEdge(float local_level_set,
                              float neighbor_level_set,
                              float edge_mass,
                              float sig_vel,
                              float delta_t,
                              float rho,
                              float delta_x,
                              float &local_term,
                              float &neighbor_term,
                              float &b) {
  //    b += 1.0f;
  if (edge_mass > 0.0f && local_level_set > 0.0f) {
    auto left_coe = delta_t / (rho * rho * delta_x * delta_x);
    auto right_coe = 1.0f / (rho * delta_x);
    b += right_coe * edge_mass * sig_vel;

    local_term += left_coe * edge_mass;
    if (neighbor_level_set > 0.0f) {
      neighbor_term -= left_coe * edge_mass;
    } else {
      float theta =
          max(local_level_set / (local_level_set - neighbor_level_set), 1e-6f);
      local_term += right_coe * edge_mass * (1.0f - theta) / theta;
    }
  }
}

__global__ void PreparePoissonEquationKernel(
    MACGridView<float> mass,
    MACGridView<float> velocity,
    GridView<float> level_set,
    float delta_t,
    float rho,
    float delta_x,
    GridView<AdjacentInfo> adjacent_infos,
    GridView<float> bs) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < bs.Header().TotalCells()) {
    auto index3 = bs.Header().Index(id);
    float local_level_set = level_set[id];
    AdjacentInfo adjacent_info{};
    float b = 0.0f;
    for (int dim = 0; dim < 3; dim++) {
      for (int edge = 0; edge < 2; edge++) {
        glm::ivec3 offset{0};
        offset[dim] = edge;
        auto edge_mass = mass.grids[dim](index3 + offset);
        auto sig_vel = velocity.grids[dim](index3 + offset);
        if (edge) {
          sig_vel = -sig_vel;
        }
        offset[dim] = edge * 2 - 1;
        if (level_set.LegalIndex(index3 + offset)) {
          ConstructEdge(local_level_set, level_set(index3 + offset), edge_mass,
                        sig_vel, delta_t, rho, delta_x, adjacent_info.local,
                        adjacent_info.edge[dim][edge], b);
        }
      }
    }

    adjacent_infos[id] = adjacent_info;
    bs[id] = b;
  }
}

__device__ void CalculateAirPressure(float level_set_fluid,
                                     float level_set_air,
                                     float fluid_pressure,
                                     float &air_pressure) {
  float theta = max(level_set_fluid / (level_set_fluid - level_set_air), 1e-6f);
  air_pressure = fluid_pressure * (theta - 1.0f) / theta;
}

__global__ void UpdateVelocityFieldKernel(GridView<float> pressure,
                                          GridView<float> level_set,
                                          GridView<float> vel_field,
                                          GridView<bool> valid_sample,
                                          float delta_t,
                                          float rho,
                                          float delta_x,
                                          int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < vel_field.Header().TotalCells()) {
    auto index3 = vel_field.Header().Index(id);
    auto offset = glm::ivec3{0};
    offset[dim] = -1;
    float new_vel = vel_field(index3);
    bool valid = false;
    if (index3[dim] > 0 && index3[dim] < vel_field.Header().size[dim] - 1) {
      auto lower_pressure = pressure(index3 + offset);
      auto upper_pressure = pressure(index3);
      auto lower_level_set = level_set(index3 + offset);
      auto upper_level_set = level_set(index3);
      if (lower_level_set > 0.0f || upper_level_set > 0.0f) {
        valid = true;
        if (upper_level_set <= 0.0f) {
          CalculateAirPressure(lower_level_set, upper_level_set, lower_pressure,
                               upper_pressure);
        } else if (lower_level_set <= 0.0f) {
          CalculateAirPressure(upper_level_set, lower_level_set, upper_pressure,
                               lower_pressure);
        }
        new_vel -=
            (upper_pressure - lower_pressure) * delta_t / (delta_x * rho);
      }
    } else {
      new_vel = 0.0f;
    }
    valid_sample(index3) = valid;
    vel_field(index3) = new_vel;
  }
}

__global__ void MixVelocityFieldKernel(GridView<float> vel_field,
                                       GridView<float> mass,
                                       float weight) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < vel_field.Header().TotalCells()) {
    vel_field[id] *= mass[id] / weight;
  }
}

__global__ void Grid2ParticleTransferKernel(Particle *particles,
                                            MACGridView<float> vel_field,
                                            MACGridView<float> mass,
                                            MACGridView<bool> valid_sample,
                                            float unit_mass,
                                            int num_particle) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_particle) {
    Particle particle = particles[id];

    glm::vec3 normal = NearestSurfaceNormal(particle.position);

    glm::vec3 resulting_vel{0.0f};

    for (int dim = 0; dim < 3; dim++) {
      auto nearest_point =
          vel_field.grids[dim].Header().NearestGridPoint(particle.position);
      auto grid_pos =
          vel_field.grids[dim].Header().World2Grid(particle.position);
      glm::vec3 accum_weighted_vel_normal{0.0f};
      float accum_weight_normal = 0.0f;
      glm::vec3 accum_weighted_vel_tangential{0.0f};
      float accum_weight_tangential = 0.0f;

      glm::vec3 axis{0.0f};
      axis[dim] = 1.0f;

      glm::vec3 normal_component = glm::dot(normal, axis) * normal;
      glm::vec3 tangential_component = axis - normal_component;

      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          for (int dz = -1; dz <= 1; dz++) {
            auto index3 = nearest_point + glm::ivec3{dx, dy, dz};
            if (vel_field.grids[dim].LegalIndex(index3)) {
              if (valid_sample.grids[dim](index3)) {
                auto weight = KernelFunction(glm::vec3{index3} - grid_pos) *
                              mass.grids[dim](index3);
                if (weight > 0.0f) {
                  auto vel = vel_field.grids[dim](index3);
                  auto m = mass.grids[dim](index3);
                  accum_weighted_vel_tangential +=
                      m * weight * vel * tangential_component;
                  accum_weight_tangential += mass.grids[dim](index3) * weight;
                  accum_weighted_vel_normal +=
                      weight * vel * normal_component * m / unit_mass;
                  accum_weight_normal += weight;
                }
              }
            }
          }
        }
      }
      if (accum_weight_normal > 0.0f) {
        resulting_vel += accum_weighted_vel_normal / accum_weight_normal;
      }
      if (accum_weight_tangential > 0.0f) {
        resulting_vel +=
            accum_weighted_vel_tangential / accum_weight_tangential;
      }
    }

    particle.velocity = resulting_vel;

    particles[id] = particle;
  }
}

void FluidCore::Update(float delta_time) {
  DeviceClock device_clock;

  //    pressure_.Clear();
  level_set_.Clear(-1.0f);
  velocity_.Clear();
  transfer_weight_.Clear();
  mass_sample_.Clear();
  device_clock.Record("Clear");

  AdvectionKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), particles_.size(), delta_time);
  device_clock.Record("Advection");

  ApplyGravityKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), particles_.size(), delta_time,
      glm::vec3{0.0f, -9.8f, 0.0f});
  device_clock.Record("Apply Gravity");

  thrust::device_vector<int32_t> cell_indices(particles_.size());
  AssignCellIndicesKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), grid_cell_header_, cell_indices.data().get(),
      particles_.size());

  device_clock.Record("Assign Cell Index");

  thrust::sort_by_key(cell_indices.begin(), cell_indices.end(),
                      particles_.begin());
  device_clock.Record("Sort Particles");

  thrust::device_vector<int> lower_bound_indices(
      grid_cell_header_.TotalCells() + 1);

  BinarySearchBoundsKernel<<<CALL_SHAPE(grid_cell_header_.TotalCells() + 1)>>>(
      cell_indices.data().get(), particles_.size(),
      lower_bound_indices.data().get(), grid_cell_header_.TotalCells());

  device_clock.Record("Binary Search Bounds");

  ConstructLevelSetKernel<<<CALL_SHAPE(grid_cell_header_.TotalCells())>>>(
      particles_.data().get(), lower_bound_indices.data().get(),
      grid_center_header_, level_set_.Buffer().get());
  device_clock.Record("Construct Level Set");

  Particle2GridTransferKernel<<<CALL_SHAPE(grid_point_header_.TotalCells())>>>(
      particles_.data().get(), lower_bound_indices.data().get(),
      grid_center_header_, grid_point_header_, velocity_.View(),
      transfer_weight_.View());
  device_clock.Record("Particle 2 Grid Transfer");

  velocity_bak_ = velocity_;
  transfer_weight_bak_ = transfer_weight_;

  for (int i = 0; i < 5; i++) {
    ExtrapolateKernel<<<CALL_SHAPE(grid_point_header_.TotalCells())>>>(
        grid_point_header_, velocity_.View(), transfer_weight_.View(),
        velocity_bak_.View(), transfer_weight_bak_.View());
    velocity_ = velocity_bak_;
    transfer_weight_ = transfer_weight_bak_;
  }

  device_clock.Record("Velocity Extrapolate");

  CalculateMassSampleKernel<<<CALL_SHAPE(
      mass_sample_.UGrid().Header().TotalCells())>>>(
      mass_sample_.UView(), sim_settings_.rho, sim_settings_.delta_x);
  CalculateMassSampleKernel<<<CALL_SHAPE(
      mass_sample_.VGrid().Header().TotalCells())>>>(
      mass_sample_.VView(), sim_settings_.rho, sim_settings_.delta_x);
  CalculateMassSampleKernel<<<CALL_SHAPE(
      mass_sample_.WGrid().Header().TotalCells())>>>(
      mass_sample_.WView(), sim_settings_.rho, sim_settings_.delta_x);

  device_clock.Record("Calculate Mass Sample");

  PreparePoissonEquationKernel<<<CALL_SHAPE(grid_cell_header_.TotalCells())>>>(
      mass_sample_.View(), velocity_.View(), level_set_.View(),
      sim_settings_.delta_t, sim_settings_.rho, sim_settings_.delta_x,
      operator_.adjacent_info.View(), b_.View());

  device_clock.Record("Prepare Poisson Equation");

  //    ConjugateGradient(operator_, b_, pressure_);

  //  Jacobi(operator_, b_, pressure_, 3000);

  //  MultiGrid(operator_, b_, pressure_, 10);

  MultiGridPCG(operator_, b_, pressure_);

  device_clock.Record("Solve Poisson Equation");

  UpdateVelocityFieldKernel<<<CALL_SHAPE(
      velocity_.UGrid().Header().TotalCells())>>>(
      pressure_.View(), level_set_.View(), velocity_.UView(),
      valid_sample_.UView(), sim_settings_.delta_t, sim_settings_.rho,
      sim_settings_.delta_x, 0);
  UpdateVelocityFieldKernel<<<CALL_SHAPE(
      velocity_.VGrid().Header().TotalCells())>>>(
      pressure_.View(), level_set_.View(), velocity_.VView(),
      valid_sample_.VView(), sim_settings_.delta_t, sim_settings_.rho,
      sim_settings_.delta_x, 1);
  UpdateVelocityFieldKernel<<<CALL_SHAPE(
      velocity_.WGrid().Header().TotalCells())>>>(
      pressure_.View(), level_set_.View(), velocity_.WView(),
      valid_sample_.WView(), sim_settings_.delta_t, sim_settings_.rho,
      sim_settings_.delta_x, 2);

  device_clock.Record("Update Velocity Field");

  Grid2ParticleTransferKernel<<<CALL_SHAPE(particles_.size())>>>(
      particles_.data().get(), velocity_.View(), mass_sample_.View(),
      valid_sample_.View(),
      sim_settings_.rho * sim_settings_.delta_x * sim_settings_.delta_x *
          sim_settings_.delta_x,
      particles_.size());

  device_clock.Record("Grid 2 Particle Transfer");

  device_clock.Finish();

  //    level_set_.StoreAsSheet("level_set.csv");
  //
  //    velocity_.UGrid().StoreAsSheet("velocity_u.csv");
  //    velocity_.VGrid().StoreAsSheet("velocity_v.csv");
  //    velocity_.WGrid().StoreAsSheet("velocity_w.csv");
  //
  //    mass_sample_.UGrid().StoreAsSheet("mass_sample_u.csv");
  //    mass_sample_.VGrid().StoreAsSheet("mass_sample_v.csv");
  //    mass_sample_.WGrid().StoreAsSheet("mass_sample_w.csv");
  //
  //    b_.StoreAsSheet("divergence.csv");
  //    operator_.adjacent_info.StoreAsSheet("adjacent_info.csv");
  //
  //    pressure_.StoreAsSheet("pressure.csv");
  //    std::system("pause");
}

__global__ void PoissonOperatorKernel(GridView<AdjacentInfo> adjacent_infos,
                                      GridView<float> pressure,
                                      GridView<float> result) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < result.Header().TotalCells()) {
    auto index3 = result.Header().Index(id);
    auto adjacent_info = adjacent_infos[id];
    float res = adjacent_info.local * pressure[id];
    for (int dim = 0; dim < 3; dim++) {
      for (int edge = 0; edge < 2; edge++) {
        glm::ivec3 offset{0};
        offset[dim] = edge * 2 - 1;
        auto neighbor_index3 = index3 + offset;
        if (pressure.LegalIndex(neighbor_index3)) {
          res += adjacent_info.edge[dim][edge] * pressure(neighbor_index3);
        }
      }
    }
    result[id] = res;
  }
}

__global__ void PoissonLUOperatorKernel(GridView<AdjacentInfo> adjacent_infos,
                                        GridView<float> pressure,
                                        GridView<float> result) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < result.Header().TotalCells()) {
    auto index3 = result.Header().Index(id);
    auto adjacent_info = adjacent_infos[id];
    float res = 0.0f;
    for (int dim = 0; dim < 3; dim++) {
      for (int edge = 0; edge < 2; edge++) {
        glm::ivec3 offset{0};
        offset[dim] = edge * 2 - 1;
        auto neighbor_index3 = index3 + offset;
        if (pressure.LegalIndex(neighbor_index3)) {
          res += adjacent_info.edge[dim][edge] * pressure(neighbor_index3);
        }
      }
    }
    result[id] = res;
  }
}

__global__ void PoissonDInvOperatorKernel(GridView<AdjacentInfo> adjacent_infos,
                                          GridView<float> pressure,
                                          GridView<float> result) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < result.Header().TotalCells()) {
    float local = adjacent_infos[id].local;
    float res;
    if (local != 0.0f) {
      res = pressure[id] / local;
    } else {
      res = 0.0f;
    }
    result[id] = res;
  }
}

void AdjacentOp::operator()(VectorView<float> x, VectorView<float> y) {
  PoissonOperatorKernel<<<CALL_SHAPE(adjacent_info.Header().TotalCells())>>>(
      adjacent_info, GridView<float>(adjacent_info.Header(), x.buffer),
      GridView<float>(adjacent_info.Header(), y.buffer));
}

void AdjacentOp::LU(VectorView<float> x, VectorView<float> y) {
  PoissonLUOperatorKernel<<<CALL_SHAPE(adjacent_info.Header().TotalCells())>>>(
      adjacent_info, GridView<float>(adjacent_info.Header(), x.buffer),
      GridView<float>(adjacent_info.Header(), y.buffer));
}

void AdjacentOp::D_inv(VectorView<float> x, VectorView<float> y) {
  PoissonDInvOperatorKernel<<<CALL_SHAPE(
      adjacent_info.Header().TotalCells())>>>(
      adjacent_info, GridView<float>(adjacent_info.Header(), x.buffer),
      GridView<float>(adjacent_info.Header(), y.buffer));
}

void FluidOperator::operator()(VectorView<float> x, VectorView<float> y) {
  AdjacentOp{adjacent_info}(x, y);
}

void FluidOperator::LU(VectorView<float> x, VectorView<float> y) {
  AdjacentOp{adjacent_info}.LU(x, y);
}

void FluidOperator::D_inv(VectorView<float> x, VectorView<float> y) {
  AdjacentOp{adjacent_info}.D_inv(x, y);
}

__global__ void DownSampleResidualKernel(GridView<float> residual,
                                         GridView<float> residual_coarse) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < residual_coarse.Header().TotalCells()) {
    auto index3 = residual_coarse.Header().Index(id);
    auto index3_fine = index3 * 2;
    float res = 0.0f;
    for (int dx = 0; dx < 2; dx++) {
      for (int dy = 0; dy < 2; dy++) {
        for (int dz = 0; dz < 2; dz++) {
          auto index3_fine_offset = index3_fine + glm::ivec3{dx, dy, dz};
          if (residual.LegalIndex(index3_fine_offset)) {
            res += residual(index3_fine_offset);
          }
        }
      }
    }
    residual_coarse[id] = res / 8.0f;
  }
}

__global__ void UpSampleCorrectionKernel(GridView<float> correction,
                                         GridView<float> correction_fine) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < correction_fine.Header().TotalCells()) {
    auto index3 = correction_fine.Header().Index(id);
    auto index3_coarse = index3 / 2;
    correction_fine[id] = correction(index3_coarse);
  }
}

__global__ void DownSampleAdjacentInfoKernel(
    GridView<AdjacentInfo> adjacent_info,
    GridView<AdjacentInfo> adjacent_info_coarse) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < adjacent_info_coarse.Header().TotalCells()) {
    auto index3 = adjacent_info_coarse.Header().Index(id);
    auto index3_fine = index3 * 2;
    AdjacentInfo res{};
    for (int dx = 0; dx < 2; dx++) {
      for (int dy = 0; dy < 2; dy++) {
        for (int dz = 0; dz < 2; dz++) {
          auto offset = glm::ivec3{dx, dy, dz};
          auto index3_fine_offset = index3_fine + offset;
          if (adjacent_info.LegalIndex(index3_fine_offset)) {
            auto fine_adj = adjacent_info(index3_fine_offset);
            res.local += fine_adj.local;
            for (int dim = 0; dim < 3; dim++) {
              for (int edge = 0; edge < 2; edge++) {
                if (edge == offset[dim]) {
                  res.edge[dim][edge] += fine_adj.edge[dim][edge];
                } else {
                  res.local += fine_adj.edge[dim][edge];
                }
              }
            }
          }
        }
      }
    }
    adjacent_info_coarse[id] = res;
  }
}

std::vector<MultiGridLevel> FluidOperator::MultiGridLevels() {
  std::vector<MultiGridLevel> levels;

  GridView<AdjacentInfo> current_adjacent_info = adjacent_info.View();

  down_sampled_adjacent_info.clear();

  while (true) {
    levels.emplace_back();
    auto &level = levels.back();
    level.linear_op = [current_adjacent_info](VectorView<float> x,
                                              VectorView<float> y) {
      AdjacentOp{current_adjacent_info}(x, y);
    };
    level.pre_smooth = [current_adjacent_info](VectorView<float> x,
                                               VectorView<float> y) {
      auto adj_op = AdjacentOp{current_adjacent_info};
      Jacobi(adj_op, x, y, 1);
    };
    level.post_smooth = [current_adjacent_info](VectorView<float> x,
                                                VectorView<float> y) {
      auto adj_op = AdjacentOp{current_adjacent_info};
      Jacobi(adj_op, x, y, 1);
    };
    auto header = current_adjacent_info.Header();

    //        printf("%d %d %d\n", header.size.x, header.size.y, header.size.z);

    if (header.size.x > 5 && header.size.y > 5 && header.size.z > 5) {
      auto coarser_header = header;
      coarser_header.size = (coarser_header.size + 1) / 2;
      coarser_header.delta_x *= 2;
      level.down_sample = [header, coarser_header](VectorView<float> x) {
        Vector<float> y(coarser_header.TotalCells());
        DownSampleResidualKernel<<<CALL_SHAPE(coarser_header.TotalCells())>>>(
            GridView<float>(header, x.buffer),
            GridView<float>(coarser_header, y.Buffer().get()));
        return y;
      };
      level.up_sample = [header, coarser_header](VectorView<float> x) {
        Vector<float> y(header.TotalCells());
        UpSampleCorrectionKernel<<<CALL_SHAPE(header.TotalCells())>>>(
            GridView<float>(coarser_header, x.buffer),
            GridView<float>(header, y.Buffer().get()));
        return y;
      };
      down_sampled_adjacent_info.emplace_back(coarser_header);
      current_adjacent_info = down_sampled_adjacent_info.back().View();
    } else {
      break;
    }
  }

  return levels;
}
