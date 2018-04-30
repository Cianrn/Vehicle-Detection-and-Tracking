from feature_functions import *
from vehicle_tracker import vehicle_tracker
from video_gen import *
import time
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import collections
import pickle
from sklearn.preprocessing import StandardScaler
import cv2

###################################################################
## Importing data

cars = []
non_cars = []

basedir_cars = os.listdir('./vehicles')
basedir_noncars = os.listdir('./non-vehicles')

for imtype in basedir_cars:
    cars.extend(glob.glob('./vehicles/' + imtype + '/*'))
print('No. of vehicles:', len(cars))

for imtype in basedir_noncars:
    non_cars.extend(glob.glob('./non-vehicles/' + imtype + '/*'))
print('No. of non vehicles:', len(non_cars))

##################################################################
## Visualizing features ##

car_ex = np.random.randint(0, len(cars))
noncar_ex = np.random.randint(0, len(non_cars))

# Read in images
car_image = cv2.imread(cars[car_ex])
car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
noncar_image = cv2.imread(non_cars[noncar_ex])
noncar_image = cv2.cvtColor(noncar_image, cv2.COLOR_BGR2RGB)

# Paramaters to vary
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

# Extract car and non car features 
car_features, car_hog_image = single_img_features(car_image, 
						color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, vis = True)
notcar_features, noncar_hog_image = single_img_features(noncar_image, 
						color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

#Visualize HOG features
# images = [car_image, car_hog_image, noncar_image, noncar_hog_image]
# titles = ['car', 'car HOG', 'non car', 'non car HOG']
# fig = plt.figure(figsize = (12,3))
# visualize(fig, 1, 4, images, titles)

####################################################################
## Load svm classifier ##
## Training our SVM classifier
nsamples = 100
random_idx = np.random.randint(0, len(cars), nsamples)
train_cars = cars #np.array(cars)[random_idx]
train_noncars = non_cars #np.array(non_cars)[random_idx]

# Paramaters to vary
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [370, 656] # Min and max in y to search in slide_window()

t = time.time()
car_features = extract_features(train_cars, color_space=color_space, 
						spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat,
                        hog_feat=hog_feat)

noncar_features = extract_features(train_noncars, color_space=color_space, 
						spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat,
                        hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, noncar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block', hist_bins,
     'histogram bins and', spatial_size, 'spatial sampling' )
print('Feature vector length:', len(X_train[0]))
t2 = time.time()
print(round(t2-t, 2), 'Seconds to create features...')
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

#########################################################################
## Test Images ##

test_path = '../CarND-Vehicle-Detection/test_images/*'
images = glob.glob(test_path)
y_start = 370
y_stop = 656
scale = [1, 1.5]
thresh = 2

for idx, img in enumerate(images):
	test_image = mpimg.imread(img)
	draw_image = np.copy(test_image) 
	test_image = test_image.astype(np.float32)/255
	print(np.min(test_image), np.max(test_image))

	tracker = vehicle_tracker()
	heatmap1 = tracker.find_cars(test_image, 
		ystart=y_start, 
		ystop=y_stop, scale=scale[0], svc=svc, 
		X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
		cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins=hist_bins, threshold=thresh)

	heatmap2 = tracker.find_cars(test_image, 
		ystart=y_start, 
		ystop=y_stop, scale=scale[1], svc=svc, 
		X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
		cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins=hist_bins, threshold=thresh)

	heat = heatmap1 + heatmap2
	heat_thresh = np.copy(heat)
	heat_thresh[heat_thresh <= thresh] = 0
	labels = label(heat_thresh)
	out_img = draw_labeled_bboxes(np.copy(test_image), labels)

	heat_images = [test_image, heat, heat_thresh, out_img]
	titles = ['original', 'heatmap', 'heat_thresh', 'result']
	fig = plt.figure(figsize=(50,25))
	visualize(fig, 1, 4, heat_images, titles) 

	
######################################################################
# Our final process pipeline used for real-time detection and tracking
test_draw = []
test_draw_heat = []
heatmaps = collections.deque(maxlen=10)
def process_image(image):

	y_start = 390
	y_stop = 656
	scale = [1, 1.5, 2] #[1.5, 2.5]
	thresh= 13
	smooth = 10

	tracker = vehicle_tracker()
		
	heatmaps1 = tracker.find_cars(image, 
		ystart=y_start, 
		ystop=y_stop, scale=scale[0], svc=svc, 
		X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
		cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins=hist_bins, threshold=thresh)
	
	heatmaps2 = tracker.find_cars(image, 
		ystart=y_start, 
		ystop=y_stop, scale=scale[1], svc=svc, 
		X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
		cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins=hist_bins, threshold=thresh)

	heatmaps3 = tracker.find_cars(image, 
		ystart=y_start, 
		ystop=y_stop, scale=scale[2], svc=svc, 
		X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, 
		cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins=hist_bins, threshold=thresh)


	comb_heat = heatmaps1 + heatmaps2 + heatmaps3
	heatmaps.append(comb_heat)
	sum_heatmaps = sum(heatmaps)
	sum_heatmaps[sum_heatmaps <= thresh] = 0
	sum_heat = np.asarray(sum_heatmaps, dtype=np.float32)
	labels = label(sum_heat)
	out_img = draw_labeled_bboxes(np.copy(image), labels)
	lane_lines, center_diff = process_image_lanes(image)
	result = cv2.addWeighted(out_img, 1, lane_lines, 0.3, 0)

	
	cv2.putText(result, str(np.round(center_diff,3)) + ' meters from center',
		(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.putText(result, str(labels[1]) + ' car(s) detected ', (50,100),
 	 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	return result


#########################################################################
## Apply pipeline to video 
output_video = 'output2_detect.mp4'
input_video = 'project_video.mp4'
t = time.time()
Clip1 = VideoFileClip(input_video)
video = Clip1.fl_image(process_image)#.subclip(20,48)
video.write_videofile(output_video, audio=False)
t2 = time.time()
print((round(t2-t, 2))/60, 'minutes to make video...')