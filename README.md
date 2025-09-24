Project 1: Camera Calibration and Fiducial Marker-Based Localization
Requirements
Hardware
  A USB webcam.
  A printed checkerboard pattern.
  Several printed AprilTags from the tag36h11 family.
Software
  Python 3.8+
  pip (Python package installer)

Setup
  In Powershell
    mkdir C:\Project_Name
    cd C:\Project_Name
    # Create Virtual Enviroment
    python -m venv myenv
    # Activate Virtual Enviroment
    .\venv\Scripts\Activate.ps1
  Install dependencies
    pip install opencv-contrib-python numpy pupil-apriltags
Configure Script
  CHECKERBOARD_SQUARES: Set this to the number of internal corners. (Columns-1, Rows-1). By default this is 7x6.
  TAG_SIZE_METERS: Measure sid length of black square on AprilTag in meters. Be as exact as possible.
  WORLD_FRAME_POSES: Starting from a fixed point (0,0,0), measure to the center (X,Y,Z) of the AprilTags and update the                      script
Calibration Phase
  Remove any calibration files (to avoid conflicts).
  Run the script in powershell: python project1.py
  Once the Calibration Capture window appears, hold the checkerboard in fron of the camera
  Press w to capture an image, get many different angles, positions, and distances.
  After 15 captures, press q to save and quit.
  After eveything is automatically process, it will save to calibration_results.npy
  Finally a windowwill appear showing the distorted and undistorted image.

Main Menu
  Camera window shows live X,Y,Z position accoridng to your frame. After first time calibration, this is the default      mode.
  'v' - Validation Mode: Toggles the ability to measure distances
    Make sure your AprilTags are visible and being scanned.
    Left-Click two points on that surface, the calculate 3D surface will be displayed.
    Press 'x' to clear measurements
  'a' - Accuracy Evalutaion Mode: Testing poisital accuracy
    Place your camera at a physically measured location (ground truth). 
    Enter that locations X,Y,Z coordinates.
    The text on-screen wiill display the Root Mean Square Error (RMSE) between the ground truth and calculated           position.
  'c' - Re-Calibrate: Restarts the calibration phase.
  'q' - Quit: Shuts down the program.
