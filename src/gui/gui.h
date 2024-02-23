#pragma once

#include "GameX/GameX.h"
#include "condition_variable"
#include "core/interface.h"
#include "gui/camera_third_person.h"
#include "gui/frame_counter.h"
#include "thread"

struct GUISettings {
  float particle_radius{0.005f};
  std::function<bool(const glm::vec3 &)> initial_particle_range =
      [](const glm::vec3 &p) {
        //      return p.x > 0.05f && p.x < 0.95f && p.y > 0.05f && p.y < 0.95f
        //      &&
        //             p.z > 0.05f && p.z < 0.95f;
        return glm::length(p - glm::vec3{0.3, 0.8f, 0.3f}) < 0.1f ||
               (p.x > 0.05f && p.x < 0.95f && p.y > 0.05f && p.y < 0.4f &&
                p.z > 0.05f && p.z < 0.95f);
      };
  bool multithreaded{false};
  float render_delta_t{0.01f};
  int initial_type = 1;  // 0 aligned, 1 random
};

class GUIApp : public GameX::Base::Application {
 public:
  typedef GameX::Base::ApplicationSettings AppSettings;

  GUIApp(const AppSettings &settings,
         const GUISettings &gui_settings = {},
         const SimSettings &sim_settings = {});

  ~GUIApp();

 private:
  void OnInit() override;

  void OnUpdate() override;

  void OnRender() override;

  void OnCleanup() override;

  void CursorPosCallback(double xpos, double ypos) override;

  void ScrollCallback(double xoffset, double yoffset) override;

  void LogicThread();

  GUISettings gui_settings_{};
  SimSettings sim_settings_{};

  GameX::Graphics::UScene scene_;
  GameX::Graphics::UImage envmap_image_;
  GameX::Graphics::UStaticModel particle_model_;
  GameX::Graphics::UStaticModel container_model_;
  GameX::Graphics::UAmbientLight ambient_light_;
  GameX::Graphics::UDirectionalLight directional_light_;
  GameX::Graphics::UCamera camera_;
  GameX::Graphics::UFilm film_;
  GameX::Graphics::UEntity entity_;
  std::unique_ptr<GameX::Graphics::ColorParticleGroup> color_particle_group_;

  float camera_distance_{3.0f};

  std::unique_ptr<CameraControllerThirdPerson> camera_controller_;

  bool ignore_next_mouse_move_{true};

  FluidInterface *instance_{nullptr};

  std::thread logic_thread_;
  std::condition_variable update_resource_cv_;
  std::mutex render_resource_mutex_;
  bool particle_updated_{false};
  bool thread_exit_{false};
};
