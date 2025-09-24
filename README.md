Project 1: Camera Calibration and Fiducial Marker-Based Localization<br/>
Requirements<br/>
Hardware<br/>
  A USB webcam.<br/>
  A printed checkerboard pattern.<br/>
  Several printed AprilTags from the tag36h11 family.<br/>
Software<br/>
  Python 3.8+<br/>
  pip (Python package installer)<br/>

Setup<br/>
  In Powershell<br/>
    mkdir C:\Project_Name<br/>
    cd C:\Project_Name<br/>
    # Create Virtual Enviroment<br/>
    python -m venv myenv<br/>
    # Activate Virtual Enviroment<br/>
    .\venv\Scripts\Activate.ps1<br/>
  Install dependencies<br/>
    pip install opencv-contrib-python numpy pupil-apriltags<br/>
Configure Script<br/>
  CHECKERBOARD_SQUARES: Set this to the number of internal corners. (Columns-1, Rows-1). By default this is 7x6.<br/>
  TAG_SIZE_METERS: Measure sid length of black square on AprilTag in meters. Be as exact as possible.<br/>
  WORLD_FRAME_POSES: Starting from a fixed point (0,0,0), measure to the center (X,Y,Z) of the AprilTags and update the script<br/>
Calibration Phase<br/>
  Remove any calibration files (to avoid conflicts).<br/>
  Run the script in powershell: python project1.py<br/>
  Once the Calibration Capture window appears, hold the checkerboard in fron of the camera<br/>
  Press w to capture an image, get many different angles, positions, and distances.<br/>
  After 15 captures, press q to save and quit.<br/>
  After eveything is automatically process, it will save to calibration_results.npy<br/>
  Finally a windowwill appear showing the distorted and undistorted image.<br/>

Main Menu<br/>
  Camera window shows live X,Y,Z position accoridng to your frame. After first time calibration, this is the default      mode.<br/>
  'v' - Validation Mode: Toggles the ability to measure distances<br/>
    Make sure your AprilTags are visible and being scanned.<br/>
    Left-Click two points on that surface, the calculate 3D surface will be displayed.<br/>
    Press 'x' to clear measurements<br/>
  'a' - Accuracy Evalutaion Mode: Testing poisital accuracy<br/>
    Place your camera at a physically measured location (ground truth). <br/>
    Enter that locations X,Y,Z coordinates.<br/>
    The text on-screen wiill display the Root Mean Square Error (RMSE) between the ground truth and calculated           position.
  'c' - Re-Calibrate: Restarts the calibration phase.
  'q' - Quit: Shuts down the program.
