import cv2
import numpy as np
img = cv2.imread('colors.png')
img = cv2.resize(img, (400, 400))
kernel = np.ones((7, 7), np.uint8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_bound = np.array([-255, 100, 100])
upper_bound = np.array([255, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

segmented_img = cv2.bitwise_and(img, img, mask=mask)
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

for i in contours:
	M = cv2.moments(i)
	if M['m00'] != 0:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv2.drawContours(img, [i], -1, (0, 255, 0), 2)
		cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
		cv2.putText(img, "center", (cx - 20, cy - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	print(f"x: {cx} y: {cy}")
cv2.imshow("Output", img)
cv2.waitKey(0)

