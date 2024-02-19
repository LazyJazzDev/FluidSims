#include "gui/gui.h"

GUIApp::GUIApp(const GUIApp::AppSettings &settings) : Application(settings) {
  scene_ = Renderer()->CreateScene();
  auto extent = VkCore()->SwapChain()->Extent();
  film_ = Renderer()->CreateFilm(extent.width, extent.height);
  particle_model_ = Renderer()->CreateStaticModel("models/sphere.obj");
  envmap_image_ = Renderer()->CreateImage("textures/envmap.hdr");

  glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void GUIApp::OnInit() {
  Application::OnInit();
  scene_->SetEnvmapImage(envmap_image_.get());
  ambient_light_ =
      scene_->CreateLight<GameX::Graphics::AmbientLight>(glm::vec3{1.0f}, 0.3f);
  directional_light_ = scene_->CreateLight<GameX::Graphics::DirectionalLight>(
      glm::vec3{1.0f}, glm::vec3{3.0f, 2.0f, 1.0f}, 0.7f);
  camera_ = scene_->CreateCamera(glm::vec3{0.0f, 10.0f, 10.0f},
                                 glm::vec3{0.0f, 0.0f, 0.0f}, 45.0f, 1.0f, 0.1f,
                                 100.0f);

  //    entity_ = scene_->CreateEntity(particle_model_.get());
  color_particle_group_ =
      scene_->CreateParticleGroup<GameX::Graphics::ColorParticleGroup>(
          particle_model_.get(), 1048576);

  color_particle_group_->SetGlobalSettings(
      GameX::Graphics::ColorParticleGroup::GlobalSettings{0.1f});
  color_particle_group_->SetParticleInfo(
      {{glm::vec3{-1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 0.5f, 0.5f}},
       {glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.5f, 0.5f, 0.5f}}});

  auto extent = VkCore()->SwapChain()->Extent();
  float aspect =
      static_cast<float>(extent.width) / static_cast<float>(extent.height);
  camera_controller_ =
      std::make_unique<CameraControllerThirdPerson>(camera_.get(), aspect);
  camera_controller_->SetCenter(glm::vec3{0.0f, 0.0f, 0.0f});
}

void GUIApp::OnUpdate() {
  static auto last_time = std::chrono::steady_clock::now();
  auto current_time = std::chrono::steady_clock::now();
  float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(
                         current_time - last_time)
                         .count();
  last_time = current_time;
  camera_controller_->Update(delta_time);
}

void GUIApp::OnRender() {
  auto cmd_buffer = VkCore()->CommandBuffer();
  Renderer()->RenderPipeline()->Render(cmd_buffer->Handle(), *scene_, *camera_,
                                       *film_);

  OutputImage(cmd_buffer->Handle(), film_->output_image.get());
}

void GUIApp::OnCleanup() {
}

void GUIApp::CursorPosCallback(double xpos, double ypos) {
  static double last_xpos = xpos;
  static double last_ypos = ypos;
  double dx = xpos - last_xpos;
  double dy = ypos - last_ypos;

  last_xpos = xpos;
  last_ypos = ypos;

  if (!ignore_next_mouse_move_) {
    camera_controller_->CursorMove(dx, dy);
  }

  ignore_next_mouse_move_ = false;
}

GUIApp::~GUIApp() {
  // Release all assets in reverse order
}
