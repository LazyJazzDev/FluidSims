#pragma once
#include "glm/glm.hpp"
#include "thrust/device_vector.h"

struct GridHeader;

template <class Ty>
class GridView;

template <class Ty>
class Grid;

struct GridHeader {
  glm::ivec3 size{0, 0, 0};
  glm::vec3 origin{0.0f, 0.0f, 0.0f};
  glm::vec3 spacing{1.0f, 1.0f, 1.0f};

  __device__ __host__ int TotalCells() const {
    return size.x * size.y * size.z;
  }

  __device__ __host__ int Index(glm::ivec3 i) const {
    return i.x + size.x * (i.y + size.y * i.z);
  }

  __device__ __host__ int Index(int x, int y, int z) const {
    return x + size.x * (y + size.y * z);
  }

  __device__ __host__ glm::ivec3 Index(int i) const {
    glm::ivec3 res;
    res.x = i % size.x;
    res.y = (i / size.x) % size.y;
    res.z = i / (size.x * size.y);
    return res;
  }

  __device__ __host__ glm::vec3 World2Grid(glm::vec3 world_pos) const {
    return (world_pos - origin) / spacing;
  }

  __device__ __host__ glm::vec3 Grid2World(glm::vec3 grid_pos) const {
    return grid_pos * spacing + origin;
  }

  __device__ __host__ glm::ivec3 NearestGridPoint(glm::vec3 world_pos) const {
    return glm::ivec3(World2Grid(world_pos) + 0.5f);
  }

  __device__ __host__ glm::ivec3 GridCell(glm::vec3 world_pos) const {
    return glm::ivec3(World2Grid(world_pos));
  }
};

template <class Ty>
class GridView {
 public:
  GridView(GridHeader header, Ty *buffer) : header_(header), buffer_(buffer) {
  }

  __device__ Ty &operator[](int i) {
    return buffer_[i];
  }

  __device__ const Ty &operator[](int i) const {
    return buffer_[i];
  }

  __device__ Ty &operator()(const glm::ivec3 &index) {
    return buffer_[header_.Index(index)];
  }

  __device__ const Ty &operator()(const glm::ivec3 &index) const {
    return buffer_[header_.Index(index)];
  }

  __device__ Ty &operator()(int x, int y, int z) {
    return buffer_[header_.Index(x, y, z)];
  }

  __device__ const Ty &operator()(int x, int y, int z) const {
    return buffer_[header_.Index(x, y, z)];
  }

  GridHeader Header() const {
    return header_;
  }

 private:
  GridHeader header_;
  Ty *buffer_;
};

template <class Ty>
class Grid {
 public:
  Grid(GridHeader header = GridHeader()) : header_(header) {
    buffer_.resize(header.TotalCells());
  }

  GridHeader Header() const {
    return header_;
  }

  thrust::device_ptr<Ty> Buffer() {
    return buffer_.data();
  }

  GridView<Ty> View() {
    return GridView<Ty>(header_, buffer_.data().get());
  }

 private:
  GridHeader header_;
  thrust::device_vector<Ty> buffer_;
};
