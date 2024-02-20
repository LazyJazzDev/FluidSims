#pragma once
#include "chrono"
#include "queue"
#include "string"

class FrameCounter {
 public:
  void RecordFrame();
  float GetFPS() const;
  std::string GetFPSString() const;

 private:
  std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>>
      frames_;
};
