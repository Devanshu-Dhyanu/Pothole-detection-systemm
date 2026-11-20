# RoadGuard Android Mobile APK Application
### Course: CSE471 | Deep Learning for Computer Vision | LPU

This directory contains the Android Studio configuration and native Java bridge for deploying RoadGuard AI on mobile edge devices and vehicle dashcam smartphones.

## Key Capabilities
1. **CameraX Hardware Access**: Direct access to high-resolution rear vehicle camera with auto-focus (`capture="environment"`).
2. **GPS Geotagging**: Injects live WGS84 GPS latitude and longitude into each inference packet.
3. **PWA 1-Tap Offline Sync**: Can also be installed directly via Google Chrome on Android by selecting **"Add to Home screen"**.

## Build Instructions (Android Studio)
1. Open Android Studio and select **Open an Existing Project** pointing to `Code/android_app/`.
2. Connect an Android smartphone (Android 8.0+ / API 24+) or launch the Android Virtual Device (AVD).
3. Ensure the RoadGuard backend is running locally (`python Code/app/main.py`).
4. Click **Run > Run 'app'** or execute `./gradlew assembleDebug` to build the `.apk`.

