#include "Doom/psx_main.h"
#include "PsyDoom/Launcher/Launcher.h"
#include <clocale>

#ifdef ANDROID
#include <SDL_main.h>
#endif

#ifdef ANDROID
int SDL_main(int argc, char **argv) {
#else
int main(const int argc, const char* const* const argv) {
#endif
    // Interpret 'char*' strings as UTF-8 for 'fopen' etc.
    setlocale(LC_ALL, "en_US.UTF-8");

    // Run the game!
    #if PSYDOOM_LAUNCHER
        return Launcher::launcherMain(argc, argv);
    #else
        return psx_main(argc, argv);
    #endif
}
