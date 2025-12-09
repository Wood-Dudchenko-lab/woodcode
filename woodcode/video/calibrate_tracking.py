import numpy as np
from pathlib import Path
import time

"""
Calibrate pixel-to-centimeter conversion using a single video frame.

This script allows you to calibrate a video by selecting two points on a frame 
that correspond to a known real-world distance. The output is a conversion factor 
(cm per pixel).

Usage from the command line:

1. Run the script directly:
    $ calibrate-tracking
   - A file dialog will appear asking you to select a video (.avi) file.
   - A pop-up dialog will ask for the real-world length of the reference line in cm.
   - A pop-up window will display the first frame of the video.
   - Click exactly two points corresponding to the reference distance.
   - The script will print:
        Pixel distance between the points
        Calibration factor (1 px = X cm)

2. Notes:
   - The reference length must be in centimeters (float or integer).
   - The script requires OpenCV (`opencv-python`) and Tkinter.
       Install via:
           pip install opencv-python
       Tkinter is usually included with Python.
   - To cancel at any point, press ESC while the video frame is open or cancel dialogs.

Returns:
    cm_per_pixel (float) -- conversion factor: real-world centimeters per pixel.

Example workflow:
    $ calibrate-tracking
    [Select video file in dialog]
    [Enter reference length in dialog, e.g., 20]
    [Click two points on the video frame]
    Pixel distance: 467.39 px
    Calibration complete:
      1 px = 0.0428 cm
"""

def calibrate(video_path: Path, real_length_cm: float):
    """Calibrate px → cm by clicking two points on a video frame."""

    # Import cv2 only when needed
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for calibration.\n\n"
            "Install it using:\n"
            "  pip install opencv-python\n"
            "or (recommended for conda users):\n"
            "  conda install -c conda-forge opencv"
        )

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read video frame")

    points = []

    # Mouse callback only records points and draws them
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(f"Point selected: {x}, {y}")
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Frame", frame)

    # Display frame and register callback
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    # Wait loop until 2 points are selected
    while len(points) < 2:
        if cv2.waitKey(50) & 0xFF == 27:  # ESC to cancel
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return None
        time.sleep(0.05)

    # Destroy window safely in main thread
    cv2.destroyAllWindows()

    # Compute pixel distance
    p1, p2 = points
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    print(f"Pixel distance: {pixel_distance:.2f} px")

    # Compute calibration factor
    cm_per_pixel = real_length_cm / pixel_distance
    print(f"\nCalibration complete:\n  1 px = {cm_per_pixel:.4f} cm")
    print()

    return cm_per_pixel


def main():
    # Import Tkinter for GUI prompts
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
    except ImportError:
        raise ImportError("tkinter is required for interactive file selection.")

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask user to select video
    file_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
    )
    if not file_path:
        print("No video selected. Exiting.")
        return
    video_path = Path(file_path)

    # Ask user for reference length in cm
    length_cm = simpledialog.askfloat(
        "Reference Length",
        "Enter real length of reference line in cm:",
        minvalue=0.01
    )
    if length_cm is None:
        print("No length entered. Exiting.")
        return

    # Call calibration
    calibrate(video_path, length_cm)



if __name__ == "__main__":
    main()
