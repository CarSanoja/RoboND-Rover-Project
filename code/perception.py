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
    color_select[above_thresh] = 255
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

# THIS SECOND APPROACH SUPPOSED TO BE MORE EFFICIENT HANDELING SHADOWS THROUGH THE
# CLAHE ALGORITHM WHO TAKES THE IMAGE AND IMPROVE THE CONTRAST GETTING WELL DEFINING
# CONTOURS IF THE PARAMETERS FUNCTION WERE RIGTH. IN ADDITION TO, MY IDEA IS TO COMBINE
# TWO METHODS, COMPARE THE CLAHE RESULTS WITH THE RESULTS OF MAKING AN OTSU THRESHOLD 
# METHOD THROUGH AN BINARY BITWISE. IN OTHER HAND, THIS TIME I AM GONNA WORK WITH THE
# HLS COLOR SPACE WHERE I WILL BE MANIPULATING THE LUMINANCE AND SATURATION CHANNEL MAKING
# OBSERVATION TO SEE WHERE WE GET BETTER RESULTS.

# THIS ALGORITHM WAS THINKING TO IMPROVE THE COLOR_THRES FUNCTION THAT WE GOT AS INTRODUCTION

def Segmentation(image):
    # OTSU METHOD FOLLOWING THE OPENCV DOCUMENTATION
    img = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    h,l,s= cv2.split(img)
    ret, l2 = cv2.threshold(l,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, s2 = cv2.threshold(s,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  
    # CLAHE ALGORTIHM FOLLOWING THE OPENCV DOCUMENTATION AND PUTTING STANDARDS PARAMETERS
    # FOR CLIP LIMIT AND GRID SIZE
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    h,l,s= cv2.split(image)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    imgCLAHE= clahe.apply(l)
    ret, l = cv2.threshold(imgCLAHE,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Comparing the two methods for optimal chooises
    l3 = cv2.bitwise_and(l,l,mask =l2)
    # Get Back to my space channel
    img_segmented=cv2.merge((h,l3,s2)) 
    #fig = plt.figure(figsize=(12,3))
    #plt.subplot(121)
    #plt.imshow(img22)
    img_segmented = cv2.cvtColor(img_segmented,cv2.COLOR_HLS2RGB)
    #img_segmented = cv2.cvtColor(img_segmented,cv2.COLOR_RGB2GRAY)
    #plt.subplot(122)
    #plt.imshow(img22, cmap='gray')
    return img_segmented

# SUMMARY:
# THE FUNCTION ALONE WORKS HORRIBLE, THE ROVER DOES NOT FOLLOW THE DIRECTION OF
# THE MEAN ANGLES FOR THE WARPED IMAGE

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
    imx = np.ones_like(img[:,:,0])
    mask = cv2.warpPerspective(imx , M, (img.shape[1], img.shape[0]))
    return warped,mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform

    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    image=Rover.img
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])


    # 3) Apply perspective transform
    warped,mask = perspect_transform(Rover.img, source, destination)

    # 2) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map=np.absolute(np.float32(threshed)-255)*mask
    rock_map = rock_segmentation(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,2]=threshed  #Navigable terrain
    Rover.vision_image[:,:,0]=obs_map   #Obstacle terrain
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image


    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    obsxpix, obsypix = rover_coords(obs_map)    #Obstacle map

    # 6) Convert rover-centric pixel values to world coordinates


    world_size=Rover.worldmap.shape[0]
    scale=2*dst_size
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, Rover.pos[0], Rover.pos[1],
        Rover.yaw, world_size, scale)



    # 7) Update Rover worldmap (to be displayed on right side of screen)

    if Rover.roll < 1 or Rover.roll > 300 - 1:
        if Rover.pitch < 1 or Rover.pitch > 300 - 1:
            Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
            Rover.worldmap[y_world, x_world, 2] = 255
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1



    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_angles = angles
    Rover.nav_dists = dist
    mean_dir = np.mean(angles)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    if rock_detection(Rover.img)==True:
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0],
                                                 Rover.pos[1], Rover.yaw, world_size, scale)
        print ("PIIIIEEEDRRRAAAAA")
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx=np.argmin(rock_dist)
        Rover.rocks_dists = rock_dist
        Rover.rocks_angles = rock_ang
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
            if Rover.roll < 1 or Rover.roll > 300 - 1:
                if Rover.pitch < 1 or Rover.pitch > 300 - 1:
                    Rover.worldmap[rock_ycen, rock_xcen, 0] = 255
        Rover.vision_image[:,:,1]=rock_map*255
    else:
        Rover.vision_image[:,:,1]=0
    
    
    return Rover