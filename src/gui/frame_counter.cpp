#include "gui/frame_counter.h"

#include "string"

void FrameCounter::RecordFrame() {
  frames_.push_back(std::chrono::high_resolution_clock::now());
  while (frames_.size() > 100) {
    frames_.pop_front();
  }
}

float FrameCounter::GetFPS() const {
  if (frames_.size() < 2) {
    return 0.0f;
  }

  auto first = frames_.front();
  auto last = frames_.back();
  return static_cast<float>(frames_.size() - 1) /
         std::chrono::duration<float>(last - first).count();
}

std::string FrameCounter::GetFPSString() const {
  if (frames_.size() < 2) {
    return "N/A";
  }
  return std::to_string(GetFPS());
}
