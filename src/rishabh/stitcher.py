import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self, ransac_trials: int = 1500, ransac_eps: float = 0.001):
        self.ransac_trials = ransac_trials
        self.ransac_eps = ransac_eps

    def get_matched_key_points(self,query_img: np.array, train_img: np.array) -> tuple:
        # Function to detect and match keypoints in two images.

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
 
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(query_img,None)
        kp2, des2 = sift.detectAndCompute(train_img,None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good
        key_points_1 = np.hstack([np.array([kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1], 1]).reshape(-1,1) for i in matches])
        key_points_2 = np.hstack([np.array([kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1], 1]).reshape(-1,1) for i in matches])
        return (key_points_1, key_points_2)
    
    def estimate_candidate_homography(self, kps_i: np.array, kpd_i: np.array) -> np.array:
        """To compute homography

        Parameters
        ----------
        kp1_i : np.array
            keypoints of image 1
        kp2_i : np.array
            keypoints of image 2

        Returns
        -------
        np.array
            Homography matrix to transform image1 to image 2
        """
        A = []
        for i in range(kps_i.shape[1]):
            A_i = np.block([
                [ kps_i[:,i].reshape(1,-1),       np.zeros((1,3)),                     -1*kpd_i[0,i]*kps_i[:,i].reshape(1,-1)],
                [ np.zeros((1,3)),                kps_i[:,i].reshape(1,-1),            -1*kpd_i[1,i]*kps_i[:,i].reshape(1,-1) ],
            ])
            A.append(A_i)
        A = np.vstack(A)
        ## Get Homography using SVD
        H_i = np.linalg.svd(A).Vh[-1,:].reshape(3,3)
        H_i = H_i / H_i[2,2]
        return H_i

    
    def get_homography_with_ransac(self,img1: np.array, img2: np.array) -> np.array:
        """Estimate best homography using ransac

        Parameters
        ----------
        img1 : np.array
            Source Image
        img2 : np.array
            Destination Image

        Returns
        -------
        np.array
            Homography matrix
        """
        M1, N1, _ = img1.shape
        M2, N2, _ = img2.shape
        # Perform feature matching
        kp1, kp2 = self.get_matched_key_points(img1, img2)
        # Normalize kps
        t_1 = np.array([[2/N1, 0, -N1/2],[0, 2/M1, -M1/2],[0,0,1]])
        t_2 = np.array([[2/N2, 0, -N2/2],[0, 2/M2, -M2/2],[0,0,1]])
        kp1 = t_1 @ kp1
        kp2 = t_2 @ kp2
        # Perform RANSAC and compute homography
        max_inliers = 0
        best_homography = None
        for i in range(self.ransac_trials):
            ## Sample 4 key points pair
            idx = np.random.randint(0, kp1.shape[1], 4)
            kp1_j = kp1[:, idx]
            kp2_j = kp2[:, idx]
            ## Estimate candidate homography
            H_i = self.estimate_candidate_homography(kp1_j, kp2_j)
            ## error?
            kp2_hat = H_i @ kp1
            kp2_hat = kp2_hat / kp2_hat[-1,:].reshape(1,-1)
            error = np.sum((kp2_hat - kp2)**2, axis=0) ** (1/2)
            threshold = (5.99 ** (1/2)) * np.std(error) if self.ransac_eps is None else self.ransac_eps
            inliers_idx = (error <= threshold)
            num_inliers = inliers_idx.sum()
            if  num_inliers > max_inliers:
                max_inliers = num_inliers
                # best_homography = self.estimate_candidate_homography(kp1[:, inliers_idx], kp2[:, inliers_idx])
                best_homography = H_i        
        # Unnormalize homography matrix
        best_homography = np.linalg.inv(t_2) @ best_homography @ t_1
        best_homography = best_homography / best_homography[2,2]
        
        return best_homography
    
    def stitch_image(self,image1: np.array, image2: np.array, homography: np.array) -> np.array:
        """Stitch image by performing warping and blending

        Parameters
        ----------
        image1 : np.array
            Image to warp
        image2 : np.array
            Image on whose plane image1 would be warped
        homography : np.array
            Homography from image1 to image2

        Returns
        -------
        np.array
            Stitched image
        """

        image1 = image1.copy()
        image2 = image2.copy()
        source_bounds = np.array([
            #top_left   top_right           bottom_right        bottom_left
            [0,         image1.shape[1]-1,  image1.shape[1]-1,  0],                 #j
            [0,         0,                  image1.shape[0]-1,  image1.shape[0]-1], #i
            [1,         1,                  1,                  1]
        ])

        ## Find bounds of source on destination's plane
        s_bounds_d = homography @ source_bounds
        s_bounds_d = (s_bounds_d / s_bounds_d[-1,:].reshape(1,-1)).astype(np.int32)
        
        ## Calculate the final bounds for the two images in (j,i) format
        min_x = min(0, s_bounds_d[0,0], s_bounds_d[0,3])
        max_x = max(image2.shape[1]-1, s_bounds_d[0, 1], s_bounds_d[0, 2])
        min_y = min(0, s_bounds_d[1,0], s_bounds_d[1,1])
        max_y = max(image2.shape[0]-1, s_bounds_d[1,3], s_bounds_d[1,2])

        ## Translation matrix to move everything to (0,0)
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        ## Final image dimension
        final_shape = (max_y - min_y+1, max_x - min_x+1) # (HxW)
        final_image = np.zeros(final_shape + (3,), dtype=np.uint8)

        ## Get inverse transformation from location of stitched_image to image1
        H_dash = translation @ homography
        H_inv = np.linalg.inv(H_dash)

        ## Start warping 
        warped_image = np.zeros(final_shape + (3,), dtype=np.uint8)

        ## get coordinates of image1 from locations of final_image via inverse transformation
        y_s, x_s = np.meshgrid(np.arange(final_shape[0]), np.arange(final_shape[1]), indexing='ij')
        y_s, x_s = y_s.flatten(), x_s.flatten()
        backward_points = np.vstack([x_s, y_s, np.ones(len(x_s), dtype=np.int64)])
        backward_points = H_inv @ backward_points
        backward_points = backward_points / backward_points[2]
        y_s = backward_points[1,:].reshape(final_shape[0], final_shape[1])
        x_s = backward_points[0,:].reshape(final_shape[0], final_shape[1])
        del backward_points
        
        ## Get image1 intensities via nearest neighbour
        for y_d in range(final_shape[0]):
            for x_d in range(final_shape[1]):
                ## Round coordinates to nearest integer to get nearest coordinates 
                x_img1 = int(np.rint(x_s[y_d, x_d]))
                y_img1 = int(np.rint(y_s[y_d, x_d]))
                if 0 <= x_img1 < image1.shape[1] and 0 <= y_img1 < image1.shape[0]:
                    warped_image[y_d, x_d] = image1[y_img1, x_img1]
        del y_s, x_s

        # Place img2 in the final_image with offset
        final_image[-min_y:image2.shape[0]-min_y, -min_x:image2.shape[1]-min_x] = image2

        # Perform blending
        mask_warped_image = np.where(np.sum(warped_image, axis=2) == 0, 0, 1).astype(np.uint8)
        mask_dest_image   = np.where(np.sum(final_image, axis=2) == 0, 0, 1).astype(np.uint8)
        w1 = cv2.distanceTransform(mask_warped_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        w1 = cv2.normalize(w1, w1, 0, 1.0, cv2.NORM_MINMAX)
        w2 = cv2.distanceTransform(mask_dest_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        w2 = cv2.normalize(w2, w2, 0, 1.0, cv2.NORM_MINMAX)
        w1 = w1 / (w1 + w2 + 1e-12)
        w2 = w2 / (w1 + w2 + 1e-12)
        w1 = np.stack([w1] * 3, axis=2)
        w2 = np.stack([w2] * 3, axis=2)

        blended_image = cv2.add(w1 * warped_image,  w2 * final_image, dtype=cv2.CV_8U)#.astype(np.uint8)
        return blended_image
    
    def get_focal_in_pixels(self, img_path: str) -> float:


        from PIL import Image
        ## sensor_width collected from camera specs
        ## model and focal length retreived from EXIF data
        sensor_width_map = {
            'Canon DIGITAL IXUS 860 IS': 5.75,
            'DSC-W170': 6.16,
            'NEX-5N': 23.4,
            'NIKON D40': 23.7
        }

        image = Image.open(img_path)
        try:
            focal_mm = image._getexif()[37386]
            model = image._getexif()[272]
            sensor_width_mm = sensor_width_map[model]
            image_width_pixels= image.size[0]
            del image

            focal_pixels = (focal_mm * image_width_pixels) / sensor_width_mm
            return focal_pixels
        
        except KeyError:
            print("could not find sensor width/ focal length (mm) in metata data to estimate focal length. Assuming FOV of 55 degrees")
            FOV = 55 #assuming FOV= 55 degree
            return (image.size[0] * 0.5) / np.tan(FOV * 0.5 * np.pi/180)

    
    def cylindrical_warp(self, img: np.array, f: float) -> np.array:
        """This function returns the cylindrical warp for a given image and focal length.
        This step is just a pre-processing step. If not done, the middle part of the 
        stitched image would be smaller and the parts near the left and right edges would
        be very large resulting into very large final_image.
        Parameters
        ----------
        img : np.array
            Image
        f : float
            focal length in pixels

        Returns
        -------
        np.array
            Cylindrically warped image based on provided focal length
        """
        h_,w_ = img.shape[:2]
        K = np.array([[f,0,w_/2],[0,f,h_/2],[0,0,1]])
        # pixel coordinates
        y_i, x_i = np.indices((h_,w_))
        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3)
        # Remove the camera properties from the image by multiplying with K^{-1}.
        Kinv = np.linalg.inv(K) 
        X = (Kinv @ X.T).T # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        B = (K @ A.T).T # project back to image-pixels plane  by apllying camera intrinsic matrix 
        # back from homog coords
        B = B[:,:-1] / B[:,[-1]]
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)
        # map img to new coordinates. Used remap 
        return cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)

    
    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        homography_matrix_list =[]

        ## Estimate focal length for cylindrical warping.
        focal_lengths = [self.get_focal_in_pixels(i) for i in all_images]

        ## Preprocessing step: Cylindrical warping. If not done, the middle part of the 
        ## stitched image would be smaller and the parts near the left and right edges would
        ## be very large resulting into very large final_image. Check function definition for 
        ## more info.
        all_images = [self.cylindrical_warp(cv2.imread(i), f) for i,f  in zip(all_images, focal_lengths)]

        # for i in all_images:
        #     cv2.imshow('test2', i)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print('Found {} Images for stitching'.format(len(all_images)))

        ## Stitch images from middle towards edges
        num_images = len(all_images)
        mid = num_images // 2
        left, right = all_images[:mid], all_images[mid:]

        img1 = left[-1]
        left = left[:-1]
        img2 = right[0]
        right = right[1:]

        homography = self.get_homography_with_ransac(img1, img2)
        stitched_image = self.stitch_image(img1, img2, homography)
        homography_matrix_list.append(homography)

        while True:
            if len(left) == 0 or len(right) == 0:
                break
            img_left = left[-1]
            left = left[:-1]
            img_right = right[0]
            right = right[1:]
            homography = self.get_homography_with_ransac(img_left, stitched_image)
            stitched_image = self.stitch_image(img_left, stitched_image, homography)
            homography_matrix_list.append(homography)

            homography = self.get_homography_with_ransac(img_right, stitched_image)
            stitched_image = self.stitch_image(img_right, stitched_image, homography)
            homography_matrix_list.append(homography)
        
        if len(right) > 0:
            homography = self.get_homography_with_ransac(right[0], stitched_image)
            stitched_image = self.stitch_image(right[0], stitched_image, homography)
            homography_matrix_list.append(homography)
        
        # Collect all homographies calculated for pair of images and return
        # Return Final panaroma
        stitched_image = stitched_image
        #####
        
        return stitched_image, homography_matrix_list 