import numpy as np
import cv2

seg2shape = {
    3: "Triangle",
    5: "Pentagon",
    6: "Hexagon",
    7: "Heptagon",
    8: "Octagon",
    9: "Nonagon",
    10: "Decagon"
}

def detect_shape(cnt):
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    num_segments = int(len(approx))
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    if num_segments > 10:
        dist = np.asarray(approx).reshape((len(approx), -1)) - np.asarray([[cx, cy]])
        dist = dist**2
        dist = np.sqrt(dist[:,0] + dist[:,1])
        majorAxis = np.max(dist)
        minorAxis = np.min(dist)
        diff = majorAxis - minorAxis
        a1 = cv2.contourArea(cnt)
        a2 = np.pi*majorAxis*minorAxis
        areaDiff = abs(a2-a1)

        if areaDiff < 500:
            if diff < 12:
                return ["Circle", cx, cy]
            else:
                return ["Ellipse", cx, cy]
        else:
            return ["None", cx, cy]
    elif num_segments > 2:
        if num_segments == 4:
            cos = []
            x = np.asarray([(approx[3][0][0] - approx[0][0][0]), (approx[3][0][1] - approx[0][0][1])])
            y = np.asarray([(approx[1][0][0] - approx[0][0][0]), (approx[1][0][1] - approx[0][0][1])])
            cos.append(abs(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))))
            diff = abs(np.sqrt(np.sum(x*x)) - np.sqrt(np.sum(y*y)))

            for i in range(1, 3):
                x = np.asarray([(approx[i-1][0][0] - approx[i][0][0]), (approx[i-1][0][1] - approx[i][0][1])])
                y = np.asarray([(approx[i+1][0][0] - approx[i][0][0]), (approx[i+1][0][1] - approx[i][0][1])])
                cos.append(abs(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))))

            if (cos[0] <0.01) and (cos[1] <0.01) and (cos[2] <0.01):
                if diff < 6:
                    return ["Square", cx, cy]
                else:
                    return ["Rectangle", cx, cy]
            else:
                return ["Quadrilateral", cx, cy]

        return [seg2shape[num_segments], cx, cy]
    else:
        return ["None", cx, cy]

def find_GS(img, test_img="test", debug_flag=0):
    kernel = np.ones((5,5),np.uint8)
    fimg = np.copy(img)
    final_edges = 0*np.copy(img[:, :, 1])

    for ch in range(3):
        imgray = np.copy(img[:,:,ch])
        imgray = cv2.bilateralFilter(imgray,6,50,50)
        if debug_flag:
                        cv2.imwrite("outputs/Grey_" + test_img + "_" + str(ch) + ".jpg", imgray)
        edges = cv2.Canny(imgray,100,200)
        edges = cv2.dilate(edges,kernel,iterations = 2)
        edges = cv2.erode(edges,kernel,iterations = 2)
        if debug_flag:
            cv2.imwrite("outputs/Edges_" + test_img + "_" + str(ch) + ".jpg", edges)
        final_edges += edges

    final_edges = np.uint8(final_edges)
    if debug_flag:
        cv2.imwrite("outputs/Edges_" + test_img + ".jpg", final_edges)

    contours, hierarchy = cv2.findContours(final_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dimg = cv2.drawContours(np.copy(img), contours, -1, (0,255,0), 3)
    if debug_flag:
        cv2.imwrite("outputs/Contours_" + test_img + ".jpg", dimg)
    dimg = np.copy(img)

    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == -1:
            cnt = contours[i]
            [shape, cx, cy] = detect_shape(cnt)

            if shape != "None":
                fimg = cv2.drawContours(fimg, [cnt], 0, (0,255,0), 3)
                dimg = cv2.drawContours(dimg, [cnt], 0, (0,255,0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(fimg, shape, (cx, cy), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    if debug_flag:
        cv2.imwrite("outputs/Filetered_Contours_" + test_img + ".jpg", dimg)
        cv2.imwrite("outputs/Final_" + test_img + ".jpg", fimg)

    return fimg
