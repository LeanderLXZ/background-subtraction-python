import numpy as np
import cv2
import imageio


video_name = 'video2.mp4'
cap = cv2.VideoCapture(video_name)

# MOG
# KaewTraKulPong, Pakorn, and Richard Bowden. "An improved adaptive background
# mixture model for real-time tracking with shadow detection." Video-based
# surveillance systems. Springer, Boston, MA, 2002. 135-144.
bs_mog1 = cv2.bgsegm.createBackgroundSubtractorMOG()

# MOG2
# Zivkovic, Zoran. "Improved adaptive Gaussian mixture model for background
# subtraction." Proceedings of the 17th International Conference on Pattern
# Recognition, 2004. ICPR 2004.. Vol. 2. IEEE, 2004.
bs_mog2 = cv2.createBackgroundSubtractorMOG2()

# GMG
# Godbehere, Andrew B., Akihiro Matsukawa, and Ken Goldberg. "Visual tracking
# of human visitors under variable-lighting conditions for a responsive audio
#  art installation." 2012 American Control Conference (ACC). IEEE, 2012.
bs_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

mask_frames_mog1 = []
mask_frames_mog2 = []
mask_frames_gmg = []
fg_frames_mog1 = []
fg_frames_mog2 = []
fg_frames_gmg = []

while True:
    # Read next frame from video
    ret, frame = cap.read()
    if not ret:
        print('Done!')
        break

    # Calculate the difference to get the mask
    mask_mog1 = bs_mog1.apply(frame)
    mask_mog2 = bs_mog2.apply(frame)
    mask_gmg = bs_gmg.apply(frame)
    mask_gmg = cv2.morphologyEx(mask_gmg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get forground from mask
    fg_mog1 = np.multiply(
        np.reshape(np.repeat(mask_mog1, 3), (*mask_mog1.shape, -1)), frame)
    fg_mog2 = np.multiply(
        np.reshape(np.repeat(mask_mog2, 3), (*mask_mog2.shape, -1)), frame)
    fg_gmg = np.multiply(
        np.reshape(np.repeat(mask_gmg, 3), (*mask_gmg.shape, -1)), frame)

    # Add mask frame to gif
    mask_frames_mog1.append(mask_mog1)
    mask_frames_mog2.append(mask_mog2)
    mask_frames_gmg.append(mask_gmg)

    # Add forground frame to gif
    fg_frames_mog1.append(fg_mog1 * 255)
    fg_frames_mog2.append(fg_mog2 * 255)
    fg_frames_gmg.append(fg_gmg * 255)

    k = cv2.waitKey(30) & 0xff
    if k == 27 or k == ord('q'):
        break

# Save the gif
print('Saving GIF...')
imageio.mimsave('results/' + video_name[:-4] + '_mog1_mask.gif',
                 mask_frames_mog1, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_mog1_fg.gif',
                fg_frames_mog1, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_mog2_mask.gif',
                mask_frames_mog2, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_mog2_fg.gif',
                fg_frames_mog2, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_gmg_mask.gif',
                mask_frames_gmg, 'GIF', duration=0.1)
imageio.mimsave('results/' + video_name[:-4] + '_gmg_fg.gif',
                fg_frames_gmg, 'GIF', duration=0.1)

cap.release()
cv2.destroyAllWindows()
