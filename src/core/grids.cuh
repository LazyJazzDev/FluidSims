#pragma once

#include "fstream"
#include "glm/glm.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"

struct GridHeader;

template <class Ty>
class GridView;

template <class Ty>
class Grid;

struct GridHeader {
  glm::ivec3 size{0, 0, 0};
  glm::vec3 origin{0.0f, 0.0f, 0.0f};
  float delta_x{1.0f};

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
    return (world_pos - origin) / delta_x;
  }

  __device__ __host__ glm::vec3 Grid2World(glm::vec3 grid_pos) const {
    return grid_pos * delta_x + origin;
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
  GridView(GridHeader header = GridHeader(), Ty *buffer = nullptr)
      : header_(header), buffer_(buffer) {
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

  __device__ __host__ bool LegalIndex(const glm::ivec3 &index) const {
    return index.x >= 0 && index.x < header_.size.x && index.y >= 0 &&
           index.y < header_.size.y && index.z >= 0 && index.z < header_.size.z;
  }

  __device__ __host__ bool InRangeWorldPos(const glm::vec3 &world_pos) const {
    auto grid_pos = header_.World2Grid(world_pos);
    return InRangeGridPos(glm::ivec3(grid_pos));
  }

  __device__ __host__ bool InRangeGridPos(const glm::ivec3 &grid_pos) const {
    return grid_pos.x >= 0 && grid_pos.x < header_.size.x && grid_pos.y >= 0 &&
           grid_pos.y < header_.size.y && grid_pos.z >= 0 &&
           grid_pos.z < header_.size.z;
  }

  __device__ __host__ GridHeader Header() const {
    return header_;
  }

  void Clear(Ty content = Ty{}) {
    thrust::fill(thrust::device_pointer_cast(buffer_),
                 thrust::device_pointer_cast(buffer_ + header_.TotalCells()),
                 content);
  }

  void StoreAsSheet(const std::string &file_name,
                    bool open_after_store = false) const {
    std::vector<Ty> host_buffer(header_.TotalCells());
    thrust::copy(thrust::device_pointer_cast(buffer_),
                 thrust::device_pointer_cast(buffer_ + header_.TotalCells()),
                 host_buffer.begin());
    std::ofstream file_out(file_name);
    for (int z = 0; z < header_.size.z; z++) {
      file_out << "z=" << z << ",,,,,," << std::endl;
      for (int y = 0; y < header_.size.y; y++) {
        for (int x = 0; x < header_.size.x; x++) {
          file_out << host_buffer[header_.Index(x, y, z)] << ", ";
        }
        file_out << std::endl;
      }
    }
    file_out.close();
    if (open_after_store) {
      std::system(("start /wait " + file_name).c_str());
    }
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

  void Clear(Ty content = Ty{}) {
    thrust::fill(buffer_.begin(), buffer_.end(), content);
  }

  void StoreAsSheet(const std::string &file_name,
                    bool open_after_store = false) {
    View().StoreAsSheet(file_name, open_after_store);
  }

 private:
  GridHeader header_;
  thrust::device_vector<Ty> buffer_;
};

template <class Ty>
struct MACGridView {
  GridView<Ty> grids[3];
};

template <class Ty>
class MACGrid {
 public:
  MACGrid(GridHeader cell_header = GridHeader())
      : cell_header_(std::move(cell_header)) {
    auto u_grid_header_ = cell_header_;
    u_grid_header_.size.x += 1;
    u_grid_header_.origin.y += 0.5f * u_grid_header_.delta_x;
    u_grid_header_.origin.z += 0.5f * u_grid_header_.delta_x;
    grids_[0] = Grid<Ty>(u_grid_header_);
    auto v_grid_header_ = cell_header_;
    v_grid_header_.size.y += 1;
    v_grid_header_.origin.x += 0.5f * v_grid_header_.delta_x;
    v_grid_header_.origin.z += 0.5f * v_grid_header_.delta_x;
    grids_[1] = Grid<Ty>(v_grid_header_);
    auto w_grid_header_ = cell_header_;
    w_grid_header_.size.z += 1;
    w_grid_header_.origin.x += 0.5f * w_grid_header_.delta_x;
    w_grid_header_.origin.y += 0.5f * w_grid_header_.delta_x;
    grids_[2] = Grid<Ty>(w_grid_header_);
  }

  Grid<Ty> &GetGrid(int dim) {
    return grids_[dim];
  }

  const Grid<Ty> &GetGrid(int dim) const {
    return grids_[dim];
  }

  Grid<Ty> &UGrid() {
    return grids_[0];
  }

  const Grid<Ty> &UGrid() const {
    return grids_[0];
  }

  Grid<Ty> &VGrid() {
    return grids_[1];
  }

  const Grid<Ty> &VGrid() const {
    return grids_[1];
  }

  Grid<Ty> &WGrid() {
    return grids_[2];
  }

  const Grid<Ty> &WGrid() const {
    return grids_[2];
  }

  GridView<Ty> UView() {
    return UGrid().View();
  }

  GridView<Ty> VView() {
    return VGrid().View();
  }

  GridView<Ty> WView() {
    return WGrid().View();
  }

  GridHeader CellHeader() const {
    return cell_header_;
  }

  void Clear(Ty content = Ty{}) {
    UGrid().Clear(content);
    VGrid().Clear(content);
    WGrid().Clear(content);
  }

  MACGridView<Ty> View() {
    return {UGrid().View(), VGrid().View(), WGrid().View()};
  }

 private:
  GridHeader cell_header_;

  Grid<Ty> grids_[3];
};
