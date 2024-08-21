import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time


def check_found(qr):
    if(len(qr) != 0):
        print(qr)
        return True
    else:
        return False


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Assume the largest contour is the QR code
    largest_contour = contours[0]
    
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the QR code area
    qr_code_area = image[y-20:y+h+20, x-20:x+w+20]
    
    # Show the original and unwrapped image
    result = cv2.cvtColor(qr_code_area, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, (400, 400))
    
    blur = cv2.GaussianBlur(result, (5, 5), 0)
    ret, bw_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    canny = cv2.Canny(result, 80, 120)

    src_points, dst_points = corners(canny)

    return src_points, dst_points, bw_im

def corners(canny):
    q1 = []
    for j in range(0, 40):
        for i in range(0, 40):
            if canny[j, i] == 255:
                x = i**2 + j**2
                q1.append([i, j, x])
    if(len(q1) != 0):
        q1 = min(q1, key=lambda pair: pair[2])
        q1 = [q1[0], q1[1]]
        print(q1)
    
    q2 = []
    for j in range(0, 40):
        for i in range(360, 400):
            if canny[j, i] == 255:
                x = j**2 + (400 - i)**2
                q2.append([i, j, x])
    if(len(q2) != 0):
        q2 = min(q2, key=lambda pair: pair[2])
        q2 = [q2[0], q2[1]]
        print(q2)
        
    q3 = []
    for j in range(360, 400):
        for i in range(0, 40):
            if canny[j, i] == 255:
                x = i**2 + (400 - j)**2
                q3.append([i, j, x])
    if(len(q3) != 0):
        q3 = min(q3, key=lambda pair: pair[2])
        q3 = [q3[0], q3[1]]
        print(q3)
        
    q4 = []
    for j in range(360, 400):
        for i in range(360, 400):
            if canny[j, i] == 255:
                x = (400 - i)**2 + (400 -j)**2
                q4.append([i, j, x])
    if(len(q4) != 0):                
        q4 = min(q4, key=lambda pair: pair[2])
        q4 = [q4[0], q4[1]]
        print(q4)
        
    q5 = [0, 0]
    j = int((q1[1]+q3[1])/2)
    for i in range(0, 40):
        if canny[j, i] == 255:
            q5[0]=i
            break
    q5[1]=j
    print(q5)
    
    q6 = [0, 0]
    j = int((q2[1]+q4[1])/2)
    for i in range(360, 400):
        if canny[j, i] == 255 and i>q6[0]:
            q6[0]=i
    q6[1]=j
    print(q6)

    src_points = np.float32([q1, q2, q3, q4, q5, q6])  # Note the order: TL, TR, BR, BL
    dst_points = np.float32([[0,0], [400, 0], [0,400], [400,400], [0,200], [400, 200]])

    return src_points, dst_points


def undistort_image(image, k1, k2, p1, p2, k3=0):
    h, w = image.shape[:2]
    # Camera matrix 
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Undistort the image
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)
    
    return undistorted_image


def apply_perspective_transform(image, src_points, dst_points):
    
    # Calculate the homography matrix
    matrix, _ = cv2.findHomography(src_points, dst_points)
    
    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return transformed_image


### main 
image = cv2.imread(r"C:\Users\JoJo\Downloads\qrcode\photo_2024-08-18_09-15-10.jpg")

found = False
qr=[]

start = time.time()

qr = decode(image)

if check_found(qr) == False:
    src_points, dst_points, processed_image = preprocess(image)
    p1 = 0.0
    p2 = 0.0
    for k1 in np.arange(-0.5, 0.55, 0.05):
        for k2 in np.arange(-0.3, 0.35, 0.05):
            undistorted_image = undistort_image(processed_image, k1, k2, p1, p2)
            flattened_image = apply_perspective_transform(undistorted_image, src_points, dst_points)
            qr = decode(flattened_image)
            found = check_found(qr)
            if found == True:
                break
        if found:
            break

stop = time.time()
print(f"time = {(stop - start)*100} ms")


