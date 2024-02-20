#pragma once

#include "GameX/GameX.h"
#include "core/interface.h"
#include "gui/camera_third_person.h"
#include "gui/frame_counter.h"

struct SimSettings {
  float delta_t{1e-3f};
  float particle_radius{0.01f};
  int grid_size{32};
  std::function<bool(const glm::vec3 &)> initial_particle_range =
      [](const glm::vec3 &p) {
        return glm::length(p - glm::vec3{0.3, 0.5f, 0.3f}) < 0.2f;
      };
};

class GUIApp : public GameX::Base::Application {
 public:
  typedef GameX::Base::ApplicationSettings AppSettings;
  GUIApp(const AppSettings &settings, const SimSettings &sim_settings = {});
  ~GUIApp();

 private:
  void OnInit() override;
  void OnUpdate() override;
  void OnRender() override;
  void OnCleanup() override;

  void CursorPosCallback(double xpos, double ypos) override;

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

  std::unique_ptr<CameraControllerThirdPerson> camera_controller_;

  bool ignore_next_mouse_move_{true};

  FluidInterface *instance_{nullptr};
};
