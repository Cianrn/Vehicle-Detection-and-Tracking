from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import glob
import pickle 
from window_tracker import window_tracker

dist_p = pickle.load(open('./calibration_pickle.p', 'rb'))
mtx = dist_p["mtx"]
dist = dist_p["dist"]

def abs_sobel_thresh(img, orient = 'x', sobel_kernel=3, thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	if orient == 'y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Find absolute gradient
	abs_sobel = np.absolute(sobel)
	# Scale to 8 bit
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Apply threshold
	grad_binary = np.zeros_like(scaled_sobel)
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh = (0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Caculate gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# rescale to 8 bit
	gradmag = ((gradmag*255)/np.max(gradmag)).astype(np.uint8)
	# Apply threshold
	mag_binary = np.zeros_like(gradmag)
	mag_binary[(gradmag>= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Find absolute gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	# Find gradient direction
	direction = np.arctan2(abs_sobely, abs_sobelx)
	# Apply thresholds
	dir_binary = np.zeros_like(direction)
	dir_binary[(direction>= thresh[0]) & (direction <= thresh[1])] = 1
	
	return dir_binary

def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255), 
		rthresh=(0, 255), lthresh=(0,255), bthresh=(0,255)):
	# Convert to HLS and extract S channel
	HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	s_channel = HLS[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1 	

	RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	r_channel = HLS[:,:,1]
	r_binary = np.zeros_like(r_channel)
	r_binary[(r_channel > rthresh[0]) & (r_channel <= rthresh[1])] = 1

	LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	b_channel = LAB[:,:,2]
	b_binary = np.zeros_like(b_channel)
	b_binary[(b_channel > bthresh[0]) & (b_channel <= bthresh[1])] = 1
	l_channel = LAB[:,:,0]
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel > lthresh[0]) & (l_channel <= lthresh[1])] = 1
	
	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	v_channel = HSV[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

	col_binary = np.zeros_like(s_channel)
	# col_binary[(s_binary == 1) & (v_binary == 1)] = 1
	col_binary[(r_binary == 1) & (s_binary == 1) & (v_binary == 1)] = 1
	
	return col_binary

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

def process_image_lanes(img):
 	
 	# Undistort each image
 	img = cv2.undistort(img, mtx, dist, None, mtx)
 	# Process image and generate binaries
 	processedImage = np.zeros_like(img[:,:,0])
 	ksize = 3
 	mag = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))
 	gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(8, 255)) # 10,255
 	grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(50, 255)) #20,255
 	col_binary = color_threshold(img, sthresh = (65,255), 
 		vthresh = (65,255), rthresh=(65,255), lthresh=(50,255), 
 		bthresh=(50,255)) 
 	#processedImage[mag == 1 | col_binary == 1] = 255
 	processedImage[((gradx == 1) & (grady == 1)) | col_binary == 1] = 255

 	# Perspective Transform
 	img_size = (img.shape[1], img.shape[0])
 	#Perspective Transform
 	bot_width = .75 #.75
 	mid_width = .08 
 	height_pct = .625 
 	bottom_trim = .935 

 	src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],
 		[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],
 		[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],
 		[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
 	offset = img.shape[1]*.26 #.258 .275
 	dst = np.float32([[offset, 0],
 	 [img.shape[1]-offset, 0],
 	 [img.shape[1]-offset,img.shape[0]],
 	 [offset,img.shape[0]]])

 	M = cv2.getPerspectiveTransform(src,dst)
 	Minv = cv2.getPerspectiveTransform(dst, src)
 	warped = cv2.warpPerspective(processedImage,M,img_size,flags=cv2.INTER_LINEAR)

 	window_width = 25
 	window_height = 80
 	margin = 25 
 	smooth = 35

 	curve_points = window_tracker(window_width = window_width, window_height = window_height, margin = margin, smooth = smooth)
 	window_centroids = curve_points.find_window_centroids(warped)

 	# Points used to draw all the left and right windows
 	l_points = np.zeros_like(warped)
 	r_points = np.zeros_like(warped)

 	leftx = []
 	rightx = []
 	# Go through each level and draw the windows 
 	for level in range(0,len(window_centroids)):
 		leftx.append(window_centroids[level][0])
 		rightx.append(window_centroids[level][1])
		# Window_mask is a function to draw window areas
 		l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
 		r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		# Add graphic points from window mask here to total pixels found 
 		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
 		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

 	# Fit lane boundaries to left, right and center positions
 	yvals = range(0, warped.shape[0])
 	res_yvals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height)

 	left_fit = np.polyfit(res_yvals, leftx, 2)
 	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
 	left_fitx = np.array(left_fitx, np.int32)

 	right_fit = np.polyfit(res_yvals, rightx, 2)
 	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
 	right_fitx = np.array(right_fitx, np.int32)

 	warp_zero = np.zeros_like(warped).astype(np.uint8)
 	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
 	yplot = np.linspace(0, 719, num = 720)
 	pts_left = np.array([np.transpose(np.vstack([left_fitx, yplot]))])
 	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yplot])))])
 	pts = np.hstack((pts_left, pts_right))
 	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

 	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
 	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0) 

 	# Define conversions in x and y from pixels space to meters
 	ym_per_pix = 30/720 # meters per pixel in y dimension
 	xm_per_pix = 3.7/700 # meters per pixel in x dimension
 	# Fit new polynomials to x,y in world space

 	# Distance from road center
 	camera_center = (left_fitx[-1] + right_fitx[-1])/2
 	center_diff = (camera_center -warped.shape[1]/2)*xm_per_pix
 	side = 'left'
 	if center_diff <= 0:
 		side = 'right'

 	left_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)\
 		*xm_per_pix, 2)
 	right_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)\
 		*xm_per_pix, 2)
 	# Calculate the new radii of curvature
 	left_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) /\
 	 np.absolute(2*left_fit_cr[0])
 	right_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) /\
 	 np.absolute(2*right_fit_cr[0])

 	# cv2.putText(newwarp, 'Left Curve radius = ' +str(np.round(left_curverad,3)) + 'm', (50,50),\
 	#  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 	# cv2.putText(result, 'Right Curve Radius = ' +str(np.round(right_curverad,3)) + 'm', (50,100),\
 	#  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 	# cv2.putText(newwarp, str(np.round(center_diff,3)) + ' meters ' + str(side) + ' of center', (50,100),\
 	#  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

 	return newwarp, center_diff #result

# output_video = 'output2_tracked.mp4'
# #input_video = 'project_video.mp4'
# input_video = 'challenge_video.mp4'

# Clip1 = VideoFileClip(input_video)
# video = Clip1.fl_image(process_image)#.subclip(35,45)
# video.write_videofile(output_video, audio=False)

