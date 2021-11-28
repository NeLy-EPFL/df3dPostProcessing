import os
import numpy as np
import cv2 as cv

def get_raw_images(video_path, num_imgs=np.inf, start=0):
    raw_imgs=[]
    cap = cv.VideoCapture(video_path)
    
    if cap.isOpened() == False:
        raise Exception('Video file cannot be read! Please check in_path to ensure it is correctly pointing to the video file')
    cont=0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cont += 1
        if ret == True and len(raw_imgs)<num_imgs:
            if cont > start:
                raw_imgs.append(frame)
        else:
            break
    
    return cap, raw_imgs

def get_foreground(imgs):
    #backSub = cv.createBackgroundSubtractorMOG2()
    sum_img = np.zeros_like(imgs[0])[:,:,0]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    for i, frame in enumerate(imgs):
        #fgMask = backSub.apply(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #open_mask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
        #if i > 10:
        sum_img = cv.bitwise_or(sum_img,gray)
    diff_img = sum_img - cv.cvtColor(imgs[0], cv.COLOR_BGR2GRAY)
    fg_img = cv.morphologyEx(diff_img, cv.MORPH_OPEN, kernel)
    #cv.imshow("mask", open_mask)
    #cv.imshow("fg_img", fg_img)
    #cv.waitKey(0)
            
    return fg_img, sum_img

def fill_holes(im_th):
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (int(w/2),int(h/2)), 255);
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    return im_out

def clean_img(img):
    ret,th = cv.threshold(img,100,255,cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(31,31))
    close_img = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)   

    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(225,225))
    #open_img = cv.morphologyEx(close_img, cv.MORPH_OPEN, kernel)
    #cv.imshow("close", close_img)
    #cv.waitKey(0)
    
    return close_img

def get_ball_params(mask, raw_img, pos_fly, show_detection, minDist=100, param1=100, param2=10, minRad=250, maxRad=300, rad_correct=0):
    if show_detection:
        out_img = raw_img.copy()
    in_img = mask.copy()
    #in_img = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(in_img, cv.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRad, maxRadius=maxRad)
    x_px = []
    y_px = []
    r_px = []
    if circles is not None:
        circles_round = np.round(circles[0, :]).astype("int")
        circles = circles[0, :]
        for (x, y, r) in circles:
            #print(x,y,r)
            x_px.append(x)
            y_px.append(y)
            r_px.append(r-rad_correct)
            #print("r=",r)
        if show_detection:
            for (x_int, y_int, r_int) in circles_round:
                cv.circle(out_img, (x_int, y_int), r_int-rad_correct, (0, 255, 0), 4)
                cv.rectangle(out_img, (x_int - 5, y_int - 5), (x_int + 5, y_int + 5), (0, 128, 255), -1)
    if show_detection:
        fly = np.round(pos_fly).astype("int")
        cv.rectangle(out_img, (fly[0]-5, fly[1]-5), (fly[0]+5, fly[1]+5), (0, 128, 255), -1)
        #cv.imshow("mask", mask)
        cv.imshow("output", out_img)
        cv.imwrite("ball_detection.jpg",out_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return [np.mean(x_px), np.mean(y_px)], np.mean(r_px)

def find_closest_to_roi(fly_mask, original):
    output = cv.connectedComponentsWithStats(fly_mask, 4, cv.CV_32S)
    stats = np.transpose(output[2])
    sizes = stats[4]
    centroids = output[3]
    roi = cv.selectROI(original)
    centroid_roi = [int(roi[0]+(roi[2]/2)),int(roi[1]+(roi[3]/2))]
    min_dist=np.inf
    for i, p in enumerate(centroids):
        dist = np.linalg.norm(p-centroid_roi)
        if dist < min_dist:
            min_dist = dist
            ind = i
    closest = np.round(centroids[ind]).astype("int")

    cv.destroyAllWindows()

    return closest

def get_fly_pos(mask, original):
    in_img = mask.copy()
    fly_mask = cv.bitwise_not(in_img)

    pos_fly = find_closest_to_roi(fly_mask, original)
    
    return pos_fly

def get_fly_pos_side(mask, original):
    in_img = mask.copy()
    ret,th = cv.threshold(in_img,180,255,cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
    fly_mask = cv.erode(th,kernel,iterations = 1)#cv.morphologyEx(th, cv.MORPH_CLOSE, kernel) 
    
    pos_fly = find_closest_to_roi(fly_mask, original) - np.array([0, 10])
    
    return pos_fly
    
def ball_size_and_pos(data_path, show_detection, front_camera=3, side_camera=7, px_size=18.4e-6):

    videos_folder = data_path[:data_path.find('/df3d')]
    front_video_path = os.path.join(videos_folder,f'camera_{front_camera}.mp4')
    side_video_path = os.path.join(videos_folder,f'camera_{side_camera}.mp4')

    try:
        _,front_imgs = get_raw_images(front_video_path, num_imgs=500, start=0)
        front = True
    except Exception as e:
        print('Front ' + str(e))
        front = False

    try:
        _,side_imgs = get_raw_images(side_video_path, num_imgs=500)
        side = True
    except Exception as e:
        print('Side ' + str(e))
        side = False
    
    #print(videos_folder)
    rad_all = []
    if front:
        fg_front, _ = get_foreground(front_imgs)
        #fg_front = fill_holes(fg_front)
        mask_front = clean_img(fg_front)
        pos_fly_front = get_fly_pos(mask_front, front_imgs[0])
        pos_ball_front, rad = get_ball_params(mask_front, front_imgs[0], pos_fly_front, show_detection)
        #print(pos_ball_front, pos_fly_front, rad)
        pos_front = pos_fly_front - pos_ball_front
        rad_all.append(rad)
    else:
        pos_front = np.array([0,0])
        
    if side:
        fg_side, sum_side = get_foreground(side_imgs)
        mask_side = clean_img(fg_side)
        pos_fly_side = get_fly_pos_side(sum_side, side_imgs[0])
        pos_ball_side, rad = get_ball_params(mask_side, side_imgs[0], pos_fly_side, show_detection)
        #print(pos_ball_side, pos_fly_side, rad)
        pos_side = pos_fly_side - pos_ball_side
        rad_all.append(rad)
    else:
        pos_side = np.array([0,0])
        
    radius_px = np.mean(rad_all)
    x_pos = pos_side[0]
    y_pos = -pos_front[0]
    z_pos = np.mean([pos_side[1],pos_front[1]]) if side else pos_front[1]
    position_px = np.array([x_pos, y_pos, z_pos])
    #print(pos_front)
    #print(pos_side)

    radius = radius_px * px_size
    position = position_px * px_size + np.array([0.0, 0.0, 0.9e-3])
    
    return radius, position
