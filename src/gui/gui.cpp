#include "gui/gui.h"

#include "random"

GUIApp::GUIApp(const AppSettings &settings,
               const GUISettings &gui_settings,
               const SimSettings &sim_settings)
    : Application(settings),
      gui_settings_(gui_settings),
      sim_settings_(sim_settings) {
  scene_ = Renderer()->CreateScene();
  auto extent = VkCore()->SwapChain()->Extent();
  film_ = Renderer()->CreateFilm(extent.width, extent.height);

  GameX::Base::AssetProbe::PublicInstance()->AddSearchPath(FLUID_ASSETS_DIR);

  container_model_ = Renderer()->CreateStaticModel("model/container.obj");

  particle_model_ = Renderer()->CreateStaticModel("models/sphere.obj");
  envmap_image_ = Renderer()->CreateImage("textures/envmap.hdr");

  //  glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

GUIApp::~GUIApp() {
  // Release all assets in reverse order
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
          particle_model_.get(), 1048576 * 36);

  color_particle_group_->SetGlobalSettings(
      GameX::Graphics::ColorParticleGroup::GlobalSettings{
          gui_settings_.particle_radius});

  entity_ = scene_->CreateEntity(container_model_.get());
  entity_->SetAffineMatrix(glm::translate(glm::mat4(1.0f), glm::vec3{0.05f}) *
                           glm::scale(glm::mat4(1.0f), glm::vec3{0.9f}));

  auto extent = VkCore()->SwapChain()->Extent();
  float aspect =
      static_cast<float>(extent.width) / static_cast<float>(extent.height);
  camera_controller_ =
      std::make_unique<CameraControllerThirdPerson>(camera_.get(), aspect);
  camera_controller_->SetCenter(glm::vec3{0.5f, 0.5f, 0.5f});
  camera_controller_->SetDistance(camera_distance_);

  instance_ = CreateFluidLogicInstance(sim_settings_);

  std::vector<glm::vec3> positions;

  for (int x = 0; x < sim_settings_.grid_size.x * 2; x++) {
    for (int y = 0; y < sim_settings_.grid_size.y * 2; y++) {
      for (int z = 0; z < sim_settings_.grid_size.z * 2; z++) {
        glm::vec3 p =
            (glm::vec3{x, y, z} + 0.5f) * sim_settings_.delta_x * 0.5f;
        if (gui_settings_.initial_particle_range(p)) {
          positions.push_back(p);
        }
      }
    }
  }

  instance_->SetParticles(positions);
  if (gui_settings_.multithreaded) {
    logic_thread_ = std::thread(&GUIApp::LogicThread, this);
  }
}

void GUIApp::OnUpdate() {
  static auto last_time = std::chrono::steady_clock::now();
  auto current_time = std::chrono::steady_clock::now();
  float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(
                         current_time - last_time)
                         .count();
  last_time = current_time;
  camera_controller_->Update(delta_time);

  if (!gui_settings_.multithreaded) {
    instance_->Update(gui_settings_.render_delta_t);
    particle_updated_ = true;
  }

  {
    std::unique_lock<std::mutex> lock(render_resource_mutex_);
    if (particle_updated_) {
      auto particles = instance_->GetParticles();
      std::vector<GameX::Graphics::ColorParticleGroup::ParticleInfo>
          particle_infos(particles.size());

      std::transform(particles.begin(), particles.end(), particle_infos.begin(),
                     [](const glm::vec3 &p) {
                       return GameX::Graphics::ColorParticleGroup::ParticleInfo{
                           p, glm::vec3{0.6f, 0.7f, 0.8f}};
                     });

      color_particle_group_->SetParticleInfo(particle_infos);
      particle_updated_ = false;
      update_resource_cv_.notify_one();
    }
  }

  static FrameCounter frame_counter;
  frame_counter.RecordFrame();
  auto fps = frame_counter.GetFPSString();
  glfwSetWindowTitle(window_, ("FPS: " + fps).c_str());
}

void GUIApp::OnRender() {
  auto cmd_buffer = VkCore()->CommandBuffer();
  Renderer()->RenderPipeline()->Render(cmd_buffer->Handle(), *scene_, *camera_,
                                       *film_);

  OutputImage(cmd_buffer->Handle(), film_->output_image.get());
}

void GUIApp::OnCleanup() {
  if (gui_settings_.multithreaded) {
    thread_exit_ = true;
    {
      std::unique_lock<std::mutex> lock(render_resource_mutex_);
      particle_updated_ = false;
      update_resource_cv_.notify_one();
    }
    logic_thread_.join();
  }
}

void GUIApp::CursorPosCallback(double xpos, double ypos) {
  static double last_xpos = xpos;
  static double last_ypos = ypos;
  double dx = xpos - last_xpos;
  double dy = ypos - last_ypos;

  last_xpos = xpos;
  last_ypos = ypos;

  if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
    ignore_next_mouse_move_ = true;
  }

  if (!ignore_next_mouse_move_) {
    camera_controller_->CursorMove(dx, dy);
  }

  ignore_next_mouse_move_ = false;
}

void GUIApp::ScrollCallback(double xoffset, double yoffset) {
  camera_distance_ *= std::pow(0.9f, yoffset);
  camera_distance_ = std::max(1.0f, std::min(camera_distance_, 10.0f));
  camera_controller_->StoreCurrentState();
  camera_controller_->SetInterpolationFactor();
  camera_controller_->SetDistance(camera_distance_);
}

void GUIApp::LogicThread() {
  while (!thread_exit_) {
    instance_->Update(gui_settings_.render_delta_t);
    {
      std::unique_lock<std::mutex> lock(render_resource_mutex_);
      if (thread_exit_) {
        break;
      }
      particle_updated_ = true;
      while (particle_updated_) {
        update_resource_cv_.wait(lock);
      }
    }
  }
}
