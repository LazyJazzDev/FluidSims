#include "gui/gui.h"
#include "iostream"

int main() {
  GUIApp::AppSettings settings;
  settings.width = 1920;
  settings.height = 1080;
  GUIApp app(settings, GUISettings{}, SimSettings{});
  app.Run();
}
