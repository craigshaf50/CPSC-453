import cv2 as cv
import numpy as np
import imutils
import pandas as pd

color = (255, 255, 255)
r_click = False #indicate if right clicked, activates color segmentation
l_click = False #used to indicate if left clicked, activates color identification
click_pos = (0, 0) #position of mouse on left click
csv = pd.read_csv('colors.csv')

#lower and upper hsv boundaries for each color
colors = {'Red': [np.array([159, 50, 70]), np.array([180, 255, 255])],
          'Red2': [np.array([0, 50, 70]), np.array([9, 255, 255])],
          'Orange': [np.array([10, 50, 70]), np.array([19, 255, 255])],
          'Yellow': [np.array([20, 50, 70]), np.array([35, 255, 255])],
          'Green': [np.array([36, 50, 70]), np.array([89, 255, 255])],
          'Blue': [np.array([90, 50, 70]), np.array([128, 255, 255])],
          'Purple': [np.array([129, 50, 70]), np.array([158, 255, 255])],
          'Black': [np.array([0, 0, 0]), np.array([180, 255, 35])],
          'White': [np.array([0, 0, 231]), np.array([180, 18, 255])],
          'Gray': [np.array([0, 0, 40]), np.array([180, 18, 230])]}

#get contours of large colored objects
def color_cont(frame, points):
    mask = cv.inRange(frame, points[0], points[1])  #create mask with boundaries
    cnts = cv.findContours(mask, cv.RETR_TREE,
                           cv.CHAIN_APPROX_SIMPLE)  #find contours from mask
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv.contourArea(c)  #find how big contour is
        if area > 6000:  #only if contour is big enough, then
            M = cv.moments(c)
            cx = int(M['m10'] / M['m00'])  #calculate X position
            cy = int(M['m01'] / M['m00'])  #calculate Y position
            return c, cx, cy

#event listener for mouse events
def on_mouse(event, x, y, flags, params):
    global r_click, l_click, click_pos
    if event == cv.EVENT_LBUTTONDOWN:
        if l_click == True:
            l_click = False
            click_pos = (0, 0)
        elif l_click == False:
            click_pos = (x, y)
            l_click = True
    if event == cv.EVENT_RBUTTONDOWN:
        if r_click == True:
            r_click = False
        elif r_click == False:
            r_click = True

#find closest color based on color CSV, return the name
def get_match_name(r, g, b):
    mini = 766
    name = ''
    for i in range(len(csv)):
        current_color = csv.loc[i]  #Get current row
        #Find the absolute difference between color given and current color in image
        diff = abs(r - int(current_color['hex'][:2], 16)) + abs(g - int(current_color['hex'][2:4], 16)) + abs(b - int(current_color['hex'][4:], 16))
        if diff <= mini:
            mini = diff
            name = current_color['name']
    return name

#find closest color based on color CSV, return the rgb values of closest match
def get_match_rgb(r, g, b):
    mini = 766
    name = ''
    for i in range(len(csv)):
        current_color = csv.loc[i]  #Get current row
        #find the absolute difference between color given and current color in image
        diff = abs(r - int(current_color['hex'][:2], 16)) + abs(g - int(current_color['hex'][2:4], 16)) + abs(b - int(current_color['hex'][4:], 16))
        if diff <= mini:
            mini = diff
            hex_num = current_color['hex']
    mr,mg,mb = int(hex_num[:2], 16),int(hex_num[2:4], 16),int(hex_num[4:], 16)
    return mr, mg, mb

#returns estimated color group of color
def get_color_group(r, g, b):
    col = np.uint8([[[b,g,r]]])
    colHSV = cv.cvtColor(col, cv.COLOR_BGR2HSV)
    mask=False
    for name, clr in colors.items():  #for each color in colors
        mask = cv.inRange(colHSV, clr[0], clr[1])
        if name == 'Red2':  #group the two dict entries for red under the same name for displaying
            name = 'Red'
        if mask != False:
            return name



#starts video feed for tool
cap = cv.VideoCapture(0)

#main loop
while cap.isOpened():  
    _, frame = cap.read()
    cv.namedWindow('Frame: ')
    cv.setMouseCallback('Frame: ', on_mouse)

    #displaying instructions for user
    if (r_click == False) and (r_click == False):
        cv.rectangle(frame, (0, 0), (240, 120), (255, 255, 255), -1)
        cv.rectangle(frame, (0, 0), (240, 120), (0, 0, 0), 1)
        cv.putText(frame, 'Instructions:', (10, 20), 0, .5, (0, 0, 0), 1)
        cv.putText(frame, 'Left Click: Color Identification', (15, 40), 0, .4, (0, 0, 0), 1)
        cv.putText(frame, 'Right Click: Color Segmentation', (15, 60), 0, .4, (0, 0, 0), 1)
        cv.putText(frame, 'Space Bar: Pause/Resume Video', (15, 80), 0, .4, (0, 0, 0), 1)
        cv.putText(frame, 'Escape Btn: Exit Program', (15, 100), 0, .4, (0, 0, 0), 1)

    #color segmentation
    if r_click == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  #convert frame to HSV
        for name, clr in list(colors.items())[:7]:  #for each color in colors dict (exclude white, black, gray)
            if color_cont(hsv, clr):
                c, cx, cy = color_cont(hsv, clr)
                cv.drawContours(frame, [c], -1, (230, 230, 230), 2)  #draw contours
                cv.circle(frame, (cx-5, cy), 5, color, -1)  #plot circle for color label
                if name == 'Red2':  #group the two dict entries for red under the same name for displaying
                    name = 'Red'
                cv.putText(frame, name, (cx, cy+10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)  #put color label

    #color identification
    if l_click == True:
        cv.rectangle(frame, (0, 0), (240, 120), (255, 255, 255), -1)
        cv.rectangle(frame, (0, 0), (240, 120), (0, 0, 0), 1)
        cv.circle(frame, (click_pos[0], click_pos[1]), 5, (25, 25, 25), 2)
        circle_center_bgr = frame[click_pos[1], click_pos[0]]  #switch x and y coords due to axis change
        b, g, r = int(circle_center_bgr[0]), int(circle_center_bgr[1]), int(circle_center_bgr[2])
        est_color = get_color_group(r,g,b)
        #RGB of color
        cv.putText(frame, f'R: {r} G: {g} B: {b}', (10, 20), 0, .5, (0, 0, 0), 2)
        cv.putText(frame, f'Estimated Color Group: {est_color}', (10, 40), 0, .4, (0, 0, 0), 1)
        #circle displaying color in circle
        cv.circle(frame, (210, 15), 10, (25, 25, 25), 2)
        cv.circle(frame, (210, 15), 9, (b, g, r), -1)
        #closest color match
        cv.putText(frame, 'Closest Match:', (10, 70), 0, .4, (0, 0, 0), 1)
        cv.putText(frame, get_match_name(r, g, b), (10, 90), 0, .4, (0, 0, 0), 1)
        #closest match RGB values
        mr, mg, mb = get_match_rgb(r, g, b)
        cv.putText(frame, f'Match RGB: R: {mr} G: {mg} B: {mb}', (10, 110), 0, .4, (0, 0, 0), 1)
        #circle displaying color match in circle
        cv.circle(frame, (210, 75), 10, (25, 25, 25), 2)
        cv.circle(frame, (210, 75), 9, (mb, mg, mr), -1)

    cv.imshow("Frame: ", frame)  #show image, 

    key = cv.waitKey(1)
    #if escape key is pressed, application stops running
    if key == 27:
        break
    #if input is the space bar, the video feed pauses, resumes with any other key press
    elif key == 32:
        cv.waitKey()

cap.release()
cv.destroyAllWindows()  #closes all windows opened by opencv