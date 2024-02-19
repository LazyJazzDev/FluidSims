#pragma once

#include "GameX/GameX.h"
#include "gui/camera_third_person.h"

class GUIApp : public GameX::Base::Application {
 public:
  typedef GameX::Base::ApplicationSettings AppSettings;
  GUIApp(const AppSettings &settings);
  ~GUIApp();

 private:
  void OnInit() override;
  void OnUpdate() override;
  void OnRender() override;
  void OnCleanup() override;

  void CursorPosCallback(double xpos, double ypos) override;

  GameX::Graphics::UScene scene_;
  GameX::Graphics::UImage envmap_image_;
  GameX::Graphics::UStaticModel particle_model_;
  GameX::Graphics::UAmbientLight ambient_light_;
  GameX::Graphics::UDirectionalLight directional_light_;
  GameX::Graphics::UCamera camera_;
  GameX::Graphics::UFilm film_;
  GameX::Graphics::UEntity entity_;
  std::unique_ptr<GameX::Graphics::ColorParticleGroup> color_particle_group_;

  std::unique_ptr<CameraControllerThirdPerson> camera_controller_;

  bool ignore_next_mouse_move_{true};
};
