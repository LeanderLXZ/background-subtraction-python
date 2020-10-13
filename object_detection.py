import cv2
import numpy as np


video_name = 'video2.mp4'
cap = cv2.VideoCapture(video_name)

# MOG
# KaewTraKulPong, Pakorn, and Richard Bowden. "An improved adaptive background
# mixture model for real-time tracking with shadow detection." Video-based
# surveillance systems. Springer, Boston, MA, 2002. 135-144.
bs = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
color_m = (255, 0, 0)

out_fps = 12.0
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv2.VideoWriter('results/' + video_name[:-4] + '_od.avi',
                       fourcc, out_fps, (1280, 720))

while True:
    # Read next frame from video
    ret, frame = cap.read()
    if not ret:
        print('Done!')
        break
    frame_motion = frame.copy()

     # Calculate the difference to get the mask
    mask = bs.apply(frame_motion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    draw1 = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    draw1 = cv2.dilate(draw1, kernel, iterations=1)

    # Find the contor of the object
    contours_m, hierarchy_m = cv2.findContours(
        draw1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_m:
        if cv2.contourArea(c) < 300:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_motion, (x, y), (x + w, y + h), color_m, 2)

    k = cv2.waitKey(30) & 0xff
    if k == 27 or k == ord('q'):
        break
    
    out.write(frame_motion)

out.release()
cap.release()
cv2.destroyAllWindows()
