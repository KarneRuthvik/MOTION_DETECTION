import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to detect motion and process the video
def detect_motion_and_duplicate(video_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the video's width, height, and frames per second (fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return
    
    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_gray, gray)
        
        # Threshold the difference to get a binary image
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of the moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop through the contours and draw bounding boxes around moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if there is enough space to the right of the object to paste the duplicate
                if x + 2 * w <= width:  # Ensure the duplicate fits within the frame
                    # Duplicate the moving object by copying the region
                    duplicate = frame[y:y+h, x:x+w].copy()
                    
                    # Paste the duplicate next to the original object
                    frame[y:y+h, x+w:x+2*w] = duplicate
                    
                    # Add text to the duplicated object
                    cv2.putText(frame, "Duplicated", (x+w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # If there is not enough space, paste the duplicate to the left of the object
                    if x - w >= 0:  # Ensure the duplicate fits within the frame
                        # Duplicate the moving object by copying the region
                        duplicate = frame[y:y+h, x:x+w].copy()
                        
                        # Paste the duplicate to the left of the original object
                        frame[y:y+h, x-w:x] = duplicate
                        
                        # Add text to the duplicated object
                        cv2.putText(frame, "Detected", (x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame (optional)
        cv2.imshow("Motion Detection", frame)
        
        # Update the previous frame
        prev_gray = gray
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and writer objects
    cap.release()
    out.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
def select_mp4_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an MP4 file",
        filetypes=[("MP4 files", "*.mp4")]
    )
    return file_path

# Main function
if __name__ == "__main__":
    video_path = select_mp4_file()  # Replace with your video file path
    output_path = "output_video.mp4"  # Replace with your desired output file path
    
    detect_motion_and_duplicate(video_path, output_path)
