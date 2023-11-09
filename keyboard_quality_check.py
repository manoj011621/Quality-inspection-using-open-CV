import cv2
from  MyDetectionMethods import MyDetectionMethod
import numpy as np

# Load Aruco detector as per requirement
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

MIN_MATCH = 20 #Minimum match for template to image

w1=43.5 #width of keyboardA
h1=13.5 #height of keyboardA
x1=48.6 #width of keyboardB
y1=15.2 #height of keyboardB

#  Camera High Resolution Setting normal camera is low resolution in openCV
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#  Create a loop for multiple object detection at set speed
while True:
    ret,img = cap.read()
    detect = MyDetectionMethod(img)#object to call class MyDetectionMethod
    
    #Read all the templates needed for detection of Keyboard(A/B),backside of keyboards and missing keys.
    template1 = cv2.imread(cv2.samples.findFile("temp22.jpeg"))# Template for keyboardA
    template2 = cv2.imread(cv2.samples.findFile("temp33.jpeg"))# Template for keyboardB
    updownA=cv2.imread(cv2.samples.findFile("dellback.jpeg"))#Template for back of keyboard A
    updownB=cv2.imread(cv2.samples.findFile("backb.jpeg")) #Template for back of keyboad B
    missing= cv2.imread(cv2.samples.findFile("missing4.png"))#Template for missing Key for A
    missingB=cv2.imread(cv2.samples.findFile("missing.png")) #Template for missing Key for B
    
    #Grayscaling all the templates and video frame.
    templateGray1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
    templateGray2=cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
    updownGrayA = cv2.cvtColor(updownA, cv2.COLOR_BGR2GRAY)
    updownGrayB = cv2.cvtColor(updownB, cv2.COLOR_BGR2GRAY)
    imageGray= detect.gray_scale()
    missinggray=cv2.cvtColor(missing, cv2.COLOR_BGR2GRAY)
    missinggrayB=cv2.cvtColor(missingB, cv2.COLOR_BGR2GRAY)
    
    #width and Height 
    w, h = missinggray.shape[::-1] 
    w2, h2 = missinggrayB.shape[::-1]
    
    #Template matching 
    result1= cv2.matchTemplate(imageGray,missinggray, cv2.TM_CCOEFF_NORMED) 
    result2= cv2.matchTemplate(imageGray,missinggrayB, cv2.TM_CCOEFF_NORMED)
    
    #Feature Matching
    sift = cv2.SIFT_create()#sift feature extractor
    bf=cv2.BFMatcher()#BFMatcher for find match between features.
    kp, des = sift.detectAndCompute(imageGray, None)
    kp1, des1= sift.detectAndCompute(templateGray1, None)
    kp2, des2 = sift.detectAndCompute(templateGray2, None)
    kp3, des3 =sift.detectAndCompute(updownGrayA, None)
    kp4,des4=sift.detectAndCompute(updownGrayB, None)
    matches = bf.knnMatch(des, des1, k=2)
    matches2 = bf.knnMatch(des, des2, k=2)
    matches3 = bf.knnMatch(des, des3, k=2)
    matches4= bf.knnMatch(des, des4, k=2)
    good =[]
    good1=[]
    good2=[]
    good3=[]
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    for m,n in matches2:
        if m.distance < 0.75*n.distance:
            good1.append([m])
    for m,n in matches3:
        if m.distance < 0.75*n.distance:
            good2.append([m])
    for m,n in matches4:
        if m.distance < 0.75*n.distance:
            good3.append([m])
            
    #Condition for matching Keyboard A
    if (len(good)) > MIN_MATCH and (len(good))>(len(good1)):
        cv2.putText(img, "Type A keyboard found",(100,100),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#Display text Type A keyboard found
        thresh1=0.6
        locations=np.where(result1 >= thresh1)
        
        #condition for matching missing key
        if(locations):
            for pt in zip(*locations[::-1]):
                cv2.rectangle(img, pt, (pt[0] + missinggray.shape[1], pt[1] + missinggray.shape[0]), (0, 0, 255), 1) #Draw rectangle arround missing keys
                cv2.putText(img, "Missing Keys",(50,50),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#Display text Missing keys

        

         # Get Aruco Marker corners for highlighting
        corners, _,_ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
        if corners:
        # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter outline function
            aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio 
            pixel_cm_ratio = aruco_perimeter/12.6
            gray= detect.gray_scale()
            blur=detect.remove_noise(gray)
            thres=detect.get_threshold(blur)
            contours=detect.find_contours(thres)
        
        # Draw objects boundaries with minAreaRect function
            for cnt in contours:
        # Get rectangle
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
            # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
            # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w/ pixel_cm_ratio
                object_height = h / pixel_cm_ratio
            
                #Condition statement for Keyboard A detection
                if (object_width >11 and object_width < 15 )and (object_height > 40 and object_height < 46):                                       
                    cv2.polylines(img, [box], True, (255, 0, 0), 2)#Draw border for keyboard A
                    cv2.putText(img, "Width of keyboard A {} cm".format(round(object_width,2)), (int(x - 80),int(y - 15)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)# display width
                    cv2.putText(img, "Height of keyboard A {} cm".format(round(object_height,2)), (int(x - 80),int(y + 25)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display hieght
                    cv2.putText(img, "Angle {} degree".format(round(angle,2)), (int(x - 80),int(y - 50)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display angle
                    
    #condition statement for Keyboard B Detection
    elif(len(good1)) > MIN_MATCH and (len(good1)>(len(good))) :
        cv2.putText(img, "Type B keyboard found",(100,100),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display text for keyboard b
        thresh2=0.7
        locations=np.where(result2 >= thresh2)
        #condition for matching missing key
        if(locations):
            for pt in zip(*locations[::-1]):
                cv2.rectangle(img, pt, (pt[0] + missinggrayB.shape[1], pt[1] + missinggrayB.shape[0]), (0, 0, 255), 1)#Draw rectangle arround missing keys
                cv2.putText(img, "Missing Keys",(50,50),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#Display text Missing keys

        # Get Aruco Marker corners for highlighting
        corners, _,_ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        
        if corners:
           # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

           # Aruco Perimeter outline function
            aruco_perimeter = cv2.arcLength(corners[0], True)

           # Pixel to cm ratio aruco
            pixel_cm_ratio = aruco_perimeter/12.6
            gray= detect.gray_scale()
            blur=detect.remove_noise(gray)
            thres=detect.get_threshold(blur)
            contours=detect.find_contours(thres)
            
            # Draw objects boundaries with minAreaRect function
            for cnt in contours:
            # Get rectangle
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w/ pixel_cm_ratio
                object_height = h / pixel_cm_ratio
                
                #Condition statement for keyboard B 
                if (object_width > 40 and object_width < 52) and (object_height >15 and object_height < 20):                                       
                    cv2.polylines(img, [box], True, (255, 0, 0), 2) #Draw border for keyboard B
                    cv2.putText(img, "Width of keyboard B {} cm".format(round(object_width,2)), (int(x - 80),int(y - 15)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display text for width
                    cv2.putText(img, "Height of keyboard B {} cm".format(round(object_height,2)), (int(x - 80),int(y + 25)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display text for height
                    cv2.putText(img, "Angle {} degree".format(round(angle,2)), (int(x - 80),int(y - 50)), cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#display text for angle
                    
    #condition statement for upside down keyboard
    elif (len(good2)>= MIN_MATCH or len(good3)>= MIN_MATCH):
        cv2.putText(img, "keyboard is upside down",(50,50),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#Display text for both keyboard upside down
    
    else:
        cv2.putText(img, "no type A and B keyboard found",(50,50),cv2.FONT_HERSHEY_TRIPLEX, 1, (100 ,200 , 0), 1)#Display text if no keyboard found
        
    #Video display
    cv2.imshow("Result", img)#output

    #Program will stop when "q" is pressed
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

                
        

