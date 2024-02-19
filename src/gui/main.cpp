#include "gui/gui.h"
#include "iostream"

int main() {
  GUIApp::AppSettings settings;
  GUIApp app(settings);
  app.Run();
}
