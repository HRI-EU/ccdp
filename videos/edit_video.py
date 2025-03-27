import cv2
import numpy as np

# ----- Parameters -----
video_path = 'Success_4obj_3x.mp4'  # Path to your video file
output_path = 'output_video.mp4'  # Output file
fps = 30  # Adjust if needed or read from video properties

# Transition settings (in seconds)
t1 = 2      # First pause time in seconds
t2 = 11     # Second pause time in seconds
transition_duration = 1  # Duration (in seconds) of the transition (fade-in/out)

# Text settings
display_texts = ["""The first Box failed, now robot attempts to not put any apple there""", "The second box failed too,\n now the robot only has one option for the remaining apples"]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
thickness = 2
text_color = (255, 255, 255)  # Green; change this tuple to any BGR color, e.g., (255, 0, 0) for blue
text_position = (200, 200)  # Position where text will be placed

# Darkening factor (0 means completely black, 1 means no darkening)
darkening_factor = 0.5

# Blurring kernel size (must be odd and positive)
blur_kernel = (21, 21)

# ----- Helper Functions -----
def apply_transition_effect(frame, alpha):
    """
    Applies a gradual blur and darken effect on the frame.
    alpha: a value between 0 and 1 indicating the intensity of the effect.
           0 means no effect, 1 means full blur and darkening.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(frame, blur_kernel, 0)

    # Darken the image: blend the image with a dark version
    darkened = cv2.addWeighted(blurred, darkening_factor, np.zeros_like(frame), 1 - darkening_factor, 0)

    # Blend original and processed frame based on alpha
    # As alpha goes from 0 to 1, effect increases gradually.
    processed = cv2.addWeighted(frame, 1 - alpha, darkened, alpha, 0)
    return processed

def overlay_text(frame, text, color):
    """
    Overlays the given text onto the frame.
    """
    return cv2.putText(frame.copy(), text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

# ----- Main Processing -----
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use VideoWriter to write output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

# Convert times to frame numbers
t1_frame = int(t1 * video_fps)
t2_frame = int(t2 * video_fps)
pause_times = [t1_frame, t2_frame]
transition_frames = int(transition_duration * video_fps)
paused_length = 2
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = frame.copy()
    # Check if current frame is within t1 or t2 pause segments
    freeze_zone = False
    paused = 0
    for i in range(len(pause_times)):
        pause_frame = pause_times[i]
        display_text = display_texts[i]
        # Create a freeze window: freeze for the transition duration (for example)
        if pause_frame <= frame_idx < pause_frame + transition_frames:
            freeze_zone = True
#            print("First one")
            # Calculate alpha for smooth transition (linear ramp from 0 to 1)
            alpha = (frame_idx - pause_frame) / transition_frames
            print(alpha)
            current_frame = apply_transition_effect(current_frame, alpha)
            # Optionally overlay text when in transition
            if i == 0:
                   current_frame = cv2.putText(current_frame.copy(), "The first box failed", (250, 250), font, font_scale, text_color, thickness, cv2.LINE_AA)
                   current_frame = cv2.putText(current_frame.copy(), "Now the robot attempts to avoid ", (100, 300), font, font_scale, text_color, thickness, cv2.LINE_AA)
                   current_frame = cv2.putText(current_frame.copy(), "putting apples in the failed box", (110, 350), font, font_scale, text_color, thickness, cv2.LINE_AA)
            elif i == 1:
            	current_frame = cv2.putText(current_frame.copy(), "The second box failed too", (250, 250), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "Now the robot attempts to locate ", (100, 300), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "the remaining apples in the only ", (120, 350), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "option that has not failed yet.", (140, 400), font, font_scale, text_color, thickness, cv2.LINE_AA)
            if frame_idx == pause_frame + transition_frames - 1:
            	paused = 1
            break
            
        if pause_frame + transition_frames <= frame_idx < pause_frame + 2*transition_frames:
            freeze_zone = True
 #           print("second one")
            # Calculate alpha for smooth transition (linear ramp from 0 to 1)
            alpha = 1 - (frame_idx - pause_frame - transition_frames) / transition_frames
            current_frame = apply_transition_effect(current_frame, alpha)
            # Optionally overlay text when in transition
            if i == 0:
                   current_frame = cv2.putText(current_frame.copy(), "The first box failed", (250, 250), font, font_scale, text_color, thickness, cv2.LINE_AA)
                   current_frame = cv2.putText(current_frame.copy(), "Now the robot attempts to avoid ", (100, 300), font, font_scale, text_color, thickness, cv2.LINE_AA)
                   current_frame = cv2.putText(current_frame.copy(), "putting apples in the failed box", (110, 350), font, font_scale, text_color, thickness, cv2.LINE_AA)
            elif i == 1:
            	current_frame = cv2.putText(current_frame.copy(), "The second box failed too", (250, 250), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "Now the robot attempts to locate ", (100, 300), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "the remaining apples in the only ", (120, 350), font, font_scale, text_color, thickness, cv2.LINE_AA)
            	current_frame = cv2.putText(current_frame.copy(), "option that has not failed yet.", (140, 400), font, font_scale, text_color, thickness, cv2.LINE_AA)
            break


    if paused == 1:
    	for _ in range(int(paused_length * video_fps)):
	    	print("paused frames")
	    	out.write(current_frame)
    # Write the frame to output
    out.write(current_frame)

    # (Optional) Display the frame
    cv2.imshow('Frame', current_frame)
    if cv2.waitKey(int(1000 / video_fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
