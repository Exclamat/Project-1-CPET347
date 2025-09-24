import cv2
import numpy as np
import glob
from pupil_apriltags import Detector
import os

CHECKERBOARD_SQUARES = (7, 6)      # Number of internal corners w*h
TAG_SIZE_METERS = 0.1651        # Physical size of the AprilTag's black square.
CALIBRATION_FILE = 'calibration_results.npy' # File with calibration data.

WORLD_FRAME_POSES = { # World coordinates of known AprilTags based on user-defined (0(x),0(y),0(z))
    13: {"tvec": np.array([0.97, 0.223, 0.08], dtype=np.float32)},
    7:  {"tvec": np.array([1.17, 0.1778, 0.2794], dtype=np.float32)},
    17: {"tvec": np.array([1.24, 0.0127, 0.1524], dtype=np.float32)},
    3:  {"tvec": np.array([1.524, 0.508, 0.163], dtype=np.float32)},
}

def load_results(filepath=CALIBRATION_FILE): # Load calibration data, if there is any
    if not os.path.exists(filepath):
        return None, None
    try: # Try to load calibration data, if none, throw error
        data = np.load(filepath, allow_pickle=True).item() # Load the saved dictionary.
        mtx = data.get('mtx')
        dist = data.get('dist')
        print("Successfully loaded calibration data.")
        return mtx, dist
    
    except Exception as e: # File could not be loaded/was not found
        print(f"Error loading calibration file: {e}")
        return None, None

def save_results(mtx, dist, error, filepath=CALIBRATION_FILE): # Saves calibration data to .npy file
    data = {'mtx': mtx, 'dist': dist, 'error': error}
    np.save(filepath, data)
    print(f"Saved to '{filepath}'")

def capture_images(cap): #Used to capture calibration data
    print("Show the checkerboard to the camera from many different angles and distances.")
    print("Press 'w' to capture an image (15 are needed).")
    print("Press 'q' to save and quit.")
    
    capture_count = 0
    max_captures = 15 # Needed number of captures
    
    while capture_count < max_captures:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        text = f"Press 'w' to capture ({capture_count}/{max_captures}) | 'q' to save and quit"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Calibration Capture', frame)

        key = cv2.waitKey(1) # Wait 1 ms for key to pressed
        
        if key == ord('q'): # Save and quit
            print("Finished capturing images.")
            break
        
        elif key == ord('w'): # Take picture
            filename = f"calibration_{capture_count + 1}.png"
            cv2.imwrite(filename, frame)
            capture_count += 1
            print(f"Saved {filename}")
            
    cv2.destroyWindow('Calibration Capture')
    return capture_count

def perform_calibration(num_images): # Calculates camera intrinsics from images
    if num_images < 5:
        print("Not enough images for a reliable calibration. At least 5 are recommended. Skipping.")
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement.
    
    objp = np.zeros((CHECKERBOARD_SQUARES[0] * CHECKERBOARD_SQUARES[1], 3), np.float32) # 3d Object Points for checkerboard
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SQUARES[0], 0:CHECKERBOARD_SQUARES[1]].T.reshape(-1, 2)
    
    objpoints, imgpoints = [], [] # 3D for real world, 2D for image.
    
    images = glob.glob('calibration_*.png')
    first_image_for_display = None # Store the first valid image for comparison

    for fname in images:
        img = cv2.imread(fname)
        if first_image_for_display is None:
            first_image_for_display = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SQUARES, None) # Find checkerboard corners.
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Refine corners to sub-pixel accuracy.
            imgpoints.append(corners2)
    
    if not objpoints:
        print("ERROR: Could not find the checkerboard pattern.")
        print("Please try again.")
        return None, None
        
    # Solve for camera matrix (mtx), distortion (dist), rotation (rvecs), and translation (tvecs).
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error to quantify calibration accuracy.
    mean_error = sum([cv2.norm(imgpoints[i], cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)[0], cv2.NORM_L2) / len(imgpoints[i]) for i in range(len(objpoints))])
    final_error = mean_error / len(objpoints)
    
    print(f"Camera Matrix (K):\n{mtx}")
    print(f"\nDistortion Coefficients:\n{dist}")
    print(f"\nMean Reprojection Error: {final_error:.4f} pixels")
    
    save_results(mtx, dist, final_error)

    if first_image_for_display is not None: # Show undistorted image
        print("Press any key in the image window to continue...")
        
        undistorted_img = cv2.undistort(first_image_for_display, mtx, dist, None, None)
        
        comparison_image = np.concatenate((first_image_for_display, undistorted_img), axis=1) # Create a side-by-side comparison image
        
        cv2.imshow('Original vs. Undistorted', comparison_image)
        cv2.waitKey(0) # Wait for user to press a key
        cv2.destroyWindow('Original vs. Undistorted')

    return mtx, dist

def pose_estimation(cap, mtx, dist): # Main loop for AprilTag detection and interaction
    print("Press 'v' for Validation | 'a' for Accuracy Eval | 'c' to re-calibrate | 'q' to exit.")

    camera_params = (mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]) # Extract (fx, fy, cx, cy) for the detector.
    detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, refine_edges=1) # Look for tag family 36h11

    # State variables for UI modes.
    is_validation_mode = False
    validation_points = []
    measured_distance = 0
    is_accuracy_mode = False
    ground_truth_pose = None
    
    
    frame_count = 0 # Used to take auto images for RGB, HSV, and Grayscale conversion
    save_frames = { 
        120: 'png',
        180: 'jpg',
        240: 'bmp'
    }
    
    window_name = 'World Frame Pose Estimation'
    cv2.namedWindow(window_name)

    def validation_mode(event, x, y, flags, param): # Used for selecting points in vallidation mode
        nonlocal validation_points, measured_distance
        if event == cv2.EVENT_LBUTTONDOWN and is_validation_mode:
            if len(validation_points) < 2: # Check for 2 points
                validation_points.append((x, y))
                measured_distance = 0 # Reset on new point selection.

    cv2.setMouseCallback(window_name, validation_mode)

    while True:
        success, frame = cap.read()
        if not success: break
        
        original_frame = cv2.flip(frame, 1) # Flip frame horizontally.

        if frame_count in save_frames: # Saves specified frames from save_frame
            extension = save_frames[frame_count]
            print(f"\n--- Saving frame {frame_count} ---")
            
            bgr_filename = f"frame_{frame_count}_bgr.{extension}" # Save original
            cv2.imwrite(bgr_filename, original_frame)
            print(f"Saved {bgr_filename}")

            rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) # Save RGB
            rgb_filename = f"frame_{frame_count}_rgb.{extension}"
            cv2.imwrite(rgb_filename, rgb_frame)
            print(f"Saved {rgb_filename}")

            gray_frame_save = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY) # Save Grayscale
            gray_filename = f"frame_{frame_count}_gray.{extension}"
            cv2.imwrite(gray_filename, gray_frame_save)
            print(f"Saved {gray_filename}")

            hsv_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV) # Save HSV
            hsv_filename = f"frame_{frame_count}_hsv.{extension}"
            cv2.imwrite(hsv_filename, hsv_frame)
            print(f"Saved {hsv_filename}")

        # Usinf flipped frame.
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE_METERS)
        
        world_pose_estimates = []
        for tag in detections:
            for i in range(4): # Draw a green bounding box on detected tag.
                start_point = tuple(tag.corners[i-1].astype(int))
                end_point = tuple(tag.corners[i].astype(int))
                cv2.line(original_frame, start_point, end_point, (0, 255, 0), 2)
            
            if tag.tag_id in WORLD_FRAME_POSES: # Check if tag is one known world tags.
                R_tag_cam, t_tag_cam = tag.pose_R, tag.pose_t # Pose of tag relative to camera.
                t_world_tag = WORLD_FRAME_POSES[tag.tag_id]["tvec"].reshape(3, 1) # Pose of tag in the world.
                
                R_cam_tag = R_tag_cam.T # Invert the tag-camera transform to get camera-tag, then compose with world-tag to get world-camera.
                t_cam_tag = -R_tag_cam.T @ t_tag_cam
                R_world_cam = np.identity(3) @ R_cam_tag
                t_world_cam = np.identity(3) @ t_cam_tag + t_world_tag
                world_pose_estimates.append(t_world_cam)
        
        avg_pose = None
        if world_pose_estimates:
            avg_pose = np.mean(np.array(world_pose_estimates), axis=0) # Average multiple tag estimates for stability.

        # Draw UI
        if is_accuracy_mode:
            cv2.putText(original_frame, "Accuracy Mode: Check console to enter Ground Truth", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 165, 255), 2)
            if ground_truth_pose is not None and avg_pose is not None:
                rmse = np.sqrt(np.mean(np.square(ground_truth_pose - avg_pose))) # Calculate positional RMSE.
                rmse_text = f"Positional RMSE: {rmse * 100:.2f} cm"
                cv2.putText(original_frame, rmse_text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 165, 255), 2)

        elif is_validation_mode:
            reference_tag = next((tag for tag in detections if tag.tag_id in WORLD_FRAME_POSES), None)
            if reference_tag is None:
                cv2.putText(original_frame, "Validation: Show a known AprilTag to measure", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(original_frame, "Validation: Click two points to measure", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                if len(validation_points) == 2:
                    points_3d = [] # Project 2D pixel clicks into 3D points on the reference tag's plane.
                    for pt in validation_points:
                        u, v = pt; fx, fy, cx, cy = camera_params
                        x, y = (u - cx) / fx, (v - cy) / fy # Normalize pixel coordinates.
                        ray_cam = np.array([x, y, 1.0]).reshape(3, 1) # Create a 3D ray from camera center through the pixel.
                        
                        R_tag_cam, t_tag_cam = reference_tag.pose_R, reference_tag.pose_t
                        plane_normal_cam = R_tag_cam[:, 2].reshape(3, 1) # The tag's Z-axis is the normal of its plane.
                        
                        d = -np.dot(plane_normal_cam.T, t_tag_cam) # Find intersection of the ray with the tag's plane.
                        t = -d / np.dot(plane_normal_cam.T, ray_cam)
                        points_3d.append(t * ray_cam)
                    measured_distance = np.linalg.norm(points_3d[0] - points_3d[1]) # Euclidean distance.
                    
                if measured_distance > 0:
                    cv2.putText(original_frame, f"Measured Distance: {measured_distance * 100:.2f} cm", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
                for pt in validation_points: cv2.circle(original_frame, pt, 5, (255, 0, 255), -1)

        else: # Default Pose Estimation Mode
            if avg_pose is not None:
                pos_text = f"Cam World Pos (X,Y,Z): {avg_pose[0][0]:.2f}, {avg_pose[1][0]:.2f}, {avg_pose[2][0]:.2f} m"
                cv2.putText(original_frame, pos_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, original_frame)
        
        frame_count += 1

        # User Input
        key = cv2.waitKey(1)
        if key == ord('q'): return False
        if key == ord('c'): return True
        
        if key == ord('v'): # Toggle Validation mode.
            is_validation_mode, is_accuracy_mode = not is_validation_mode, False
            validation_points, measured_distance = [], 0
            print(f"\n--- Validation Mode {'ON' if is_validation_mode else 'OFF'} ---")
        
        if key == ord('a'): # Toggle Accuracy Evaluation mode.
            is_accuracy_mode, is_validation_mode = not is_accuracy_mode, False
            print(f"\n--- Accuracy Evaluation Mode {'ON' if is_accuracy_mode else 'OFF'} ---")
            if is_accuracy_mode:
                try:
                    print("Please enter the physically measured ground truth position of your camera.")
                    x = float(input("Enter GROUND TRUTH X (in meters, use '.' for decimals): "))
                    y = float(input("Enter GROUND TRUTH Y (in meters, use '.' for decimals): "))
                    z = float(input("Enter GROUND TRUTH Z (in meters, use '.' for decimals): "))
                    ground_truth_pose = np.array([[x], [y], [z]])
                    print(f"Ground truth set to: {ground_truth_pose.T}")
                except ValueError:
                    print("Invalid input. Please enter numbers only. Exiting accuracy mode.")
                    is_accuracy_mode = False
        
        if key == ord('x') and is_validation_mode: # Clear validation points.
             validation_points, measured_distance = [], 0
             print("Validation points cleared.")

    return False

def main(): # Main function used for c++ style workflow
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Attempt to disable autofocus.

    force_recalibration = False
    while True:
        mtx, dist = load_results()
       
        if mtx is None or dist is None or force_recalibration: # Run the calibration phase, when requested.
            if force_recalibration: print("Re-calibration requested by user.")
            else: print("No calibration data found. Starting the calibration process.")
            
            num_captures = capture_images(cap)
            mtx, dist = perform_calibration(num_captures)
            
            if mtx is None:
                print("Calibration failed. The application cannot proceed.")
                break
        
        force_recalibration = pose_estimation(cap, mtx, dist)
        
        if not force_recalibration: # Exit main loop if user quits.
            break

    # Cleanup.
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")

if __name__ == '__main__':
    main()

