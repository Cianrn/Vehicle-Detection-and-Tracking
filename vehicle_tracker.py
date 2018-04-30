from feature_functions import *
import collections
from sklearn.preprocessing import StandardScaler

class vehicle_tracker():

	def __init__(self):

		self.detected = []
		self.no_detections = []
		self.heatmaps = None

	def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, 
			orient, pix_per_cell, cell_per_block, spatial_size, 
			hist_bins, threshold):
    
	    draw_img = np.copy(img)
	    img = img.astype(np.float32)/255
	    current_heatmap = np.zeros_like(img[:,:,0])
	    
	    img_tosearch = img[ystart:ystop,:,:]
	    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	    if scale != 1:
	        imshape = ctrans_tosearch.shape
	        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	        
	    ch1 = ctrans_tosearch[:,:,0]
	    ch2 = ctrans_tosearch[:,:,1]
	    ch3 = ctrans_tosearch[:,:,2]

	    # Define blocks and steps as above
	    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	    # How many features per block
	    nfeat_per_block = orient*cell_per_block**2
	    
	    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	    window = 64
	    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	    cells_per_step = 2  # Instead of overlap, define how many cells to step
	    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
	    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
	    
	    # Compute individual channel HOG features for the entire image
	    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    
	    count = 0

	    for xb in range(nxsteps):
	        for yb in range(nysteps):
	        	count += 1
	        	ypos = yb*cells_per_step
	        	xpos = xb*cells_per_step
	        	# Extract HOG for this patch
	        	hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	        	hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	        	hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	        	hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
	        	xleft = xpos*pix_per_cell
	        	ytop = ypos*pix_per_cell
	        	# Extract the image patch
	        	subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
	        	# Get color features
	        	spatial_features = bin_spatial(subimg, size=spatial_size)
	        	hist_features = color_hist(subimg, nbins=hist_bins)
	        	# Scale features and make a prediction
	        	test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
	        	#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
	        	test_prediction = svc.predict(test_features)
	        	if test_prediction == 1:
	        		xbox_left = np.int(xleft*scale)
	        		ytop_draw = np.int(ytop*scale)
	        		win_draw = np.int(window*scale)
	        		cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
	        		current_heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
	    return current_heatmap