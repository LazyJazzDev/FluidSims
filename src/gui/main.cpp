#include "gui/gui.h"
#include "iostream"

int main() {
  GUIApp::AppSettings settings;
  settings.fullscreen = true;
  GUIApp app(settings, GUISettings{}, SimSettings{});
  app.Run();
}
