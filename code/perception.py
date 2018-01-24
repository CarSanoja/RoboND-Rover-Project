import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
###################################################################################
###################################################################################

# HERE IS MY FIRST IDEA TO MAKE THE PERCEPTION STEP:
# USING AN SUPERPIXEL SEGMENTATION ALGORITHMS (SLIC) TO MAKE THE SEGMENTATION OF IMAGES 
# AUTONOMOUS, MY APPROACH TO THIS PROBLEM SUPPOSED TO BE GOOD BECAUSE USE PIXEL VALUES
# QUANTIFYCATION TO SEGMENT THE IMAGES.

def rock_detection(img):
    yellow_mask = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > 130) \
                & (img[:,:,1] > 130) \
                & (img[:,:,2] < 50)
    # If we find a rock, we return a True value
    if above_thresh.any():
        return True
    else:
        return False

# Define a function to segment an image with rock detection
def rock_segmentation(img):
    above_thresh = (img[:,:,0] > 110) \
                & (img[:,:,1] > 110) \
                & (img[:,:,2] < 60)
    # We can create a MASK to store the values (of pixels) where we find the rock
    rock_mask=np.zeros_like(img[:,:,0]) 
    rock_mask[above_thresh]=255
    #We return a binary image with the rock
    return rock_mask

# Define a function to make de scene segmentation
# The funtion receives the image
# The function outputed: image_thresh (the ground, free space, original image), mask
# (Free space in binary format), rock_flag (Returns True if a rock is detected)
def color_segmentation(image):
    numSegments=2
    segments = slic(image, n_segments = numSegments, sigma = 2)
    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        image_thresh=cv2.bitwise_and(image, image, mask = mask)
        cv2.imshow("name",image_thresh)
        if segVal==1:
            if rock_detection==True:
                rock_flag=True
                return image_thresh,mask,rock_flag
            else:
                rock_flag=False
                return image_thresh,mask,rock_flag

# SUMMARY:
# THIS APPROACH HAD SOME ISSUES:
# IT IS TO WEEK FIGTHING WITH SHADOWS.
# IT MAKES A LOT OF ITERATIONS WHEN THE ROVER GET TRAP FOR OBSTACLES, THE SESION CODE DIES.
# ITS SLOW
# I COULD GET OVER 60% OF MAPPED AND 40-50% OF FIDELITY

#####################################################################################
#####################################################################################

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
 
    
    
    return Rover