import cv2
import numpy as np
import imageio


video_name = 'video2.mp4'
cap = cv2.VideoCapture(video_name)

_, first_frame = cap.read()

# Initialize the fist background map
bg_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 1)
bg_gray = np.array(bg_gray, dtype=np.int)

bg_list = [bg_gray]
mask_frames = []
fg_frames = []

while True:
    # Read next frame from video
    ret, frame = cap.read()
    if not ret:
        print('Done!')
        break
    
    # Convert frame to gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for denoising
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_gray = np.array(frame_gray, dtype=np.int)
    
    # Update the background - using moving average
    concat_list = np.concatenate([bg_list, [frame_gray]], 0) 
    bg_gray = np.mean(concat_list, axis=0, dtype=np.int)
    bg_list.append(bg_gray)
    
    # Calculate the difference to get the mask
    mask = bg_gray - np.array(frame_gray, dtype=np.int)
    mask = mask.__abs__()
    threshold = 50
    mask = np.array(mask > threshold, dtype=np.uint8)
    
    # Add mask frame to gif
    mask_frames.append(mask * 255)

    # Get forground from mask
    fg = np.multiply(np.reshape(np.repeat(mask, 3), (*mask.shape, -1)), frame)
    
    # Add forground frame to gif 
    fg_frames.append(fg)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27 or k == ord('q'):
        break

# Save the gif
print('Saving GIF...')
imageio.mimsave('results/' + video_name[:-4] + '_baseline_mask.gif',
                mask_frames, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_baseline_fg.gif',
                fg_frames, 'GIF', duration=0.1)
     
cap.release()
cv2.destroyAllWindows()
