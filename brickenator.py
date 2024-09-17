from os import listdir
from os.path import join

from dataclasses import dataclass, field

import numpy as np
import cv2

VALID_SIZES = [4, 6, 8, 16, 24]

BRICK_PERMUTATIONS = {
    'GREY' : [8],
    'BLACK' : [16],
    'BLUE' : [8],
    'CYAN' : [24],
    'ORANGE' : [4, 8],
    'YELLOW' : [4, 6, 8],
    'LIME' : [4, 8],
    'GREEN' : [16]
}

# Points in a* b* float format, if 3-valued then L* a* b* uint8 format
# Note that opencv's internal order is (y, x).
COLOR_POINTS = {
    'GREY' : np.array([100, 127, 127], dtype=np.float32),
    'BLACK' : np.array([80, 127, 127], dtype=np.float32),

    'MONO' : np.array([0, 0], dtype=np.float32),
    'BLUE' : np.array([0.023, -0.089], dtype=np.float32),
    'CYAN' : np.array([-0.031, -0.064], dtype=np.float32),
    'ORANGE' : np.array([0.138, 0.154], dtype=np.float32),
    'YELLOW' : np.array([0.006, 0.205], dtype=np.float32),
    'LIME' : np.array([-0.062, 0.166], dtype=np.float32),
    'GREEN' : np.array([-0.091, 0.064], dtype=np.float32)
}   

def mergeOverlappingCircles(circles: list, shape: tuple, draw = None, hook = None) -> list:
    assert len(shape) == 2, "Argument 'shape' should have dimension 2, got {}".format(len(shape))

    union = []
    unionspace = np.zeros(shape, dtype='uint8')
    for c in circles:
        # draw the outer circle
        cv2.circle(unionspace,(c[0],c[1]),c[2], 255, cv2.FILLED)
    unionspace = cv2.morphologyEx(unionspace, cv2.MORPH_ERODE, np.ones((3,3), dtype='uint8'))

    if hook == "merge":
        draw[:] = np.logical_or(draw, unionspace)

    contours, _ = cv2.findContours(unionspace, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Rotated BBox
        o, r = cv2.minEnclosingCircle(c)
        union.append([o, r])

    return union

@dataclass
class DuploBrick:
    points: list
    center: tuple
    color: str
    circles: list = field(default_factory=list)
    circleCount: int = -1
    mask : np.array = None

def findColouredRectangles(src, whitepoint: int, minimumArea=576, canny1: int = 20, canny2: int = 28, draw = None, hook = None) -> list:
    assert src.dtype == np.float32, "Expected argument 'src' dtype to be 'float32', got {}".format(src.dtype)

    colorDist = np.linalg.norm((src[:, :, 1:] - 127) / 255, axis=2)

    lumDist = abs(whitepoint - src[:, :, 0]) / 255

    trueDist = ((colorDist + np.clip(lumDist, 0, 1)) * 255).astype('uint8')

    canned = cv2.Canny(cv2.GaussianBlur(trueDist, (3,3), 0), canny1, canny2, L2gradient=True)
    canned = cv2.morphologyEx(canned, cv2.MORPH_DILATE, np.ones((3,3), dtype=np.uint8))

    contours, _ = cv2.findContours(canned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for c in contours:
        if cv2.contourArea(c) < minimumArea:
            continue

        # Rotated BBox
        bbox = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))

        o = np.mean(bbox, axis=0).astype('uint32')
        convex = cv2.convexHull(c)
        mask = np.zeros_like(trueDist, dtype='uint8')
        cv2.drawContours(mask, [convex], -1, 255, cv2.FILLED)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((3,3), dtype=np.uint8))

        h = cv2.calcHist([src], [1,2], mask, [256, 256], [0,255, 0,255])
        histPoint = (np.array(np.unravel_index(np.argmax(h), (256, 256))) - [127, 127]) / 255

        diff = ['NONE', 2]
        for color, point in COLOR_POINTS.items():
            if point.shape[0] == 2:
                dist = np.linalg.norm(point - histPoint)
                if dist < diff[1]:
                    diff = [color, dist]

        L = cv2.calcHist([src], [0], mask, [256], [0,255])

        if diff[0] == 'MONO':
            diff[1] = 255
            Lpoint = np.argmax(L)
            for color, point in COLOR_POINTS.items():
                if point.shape[0] == 3:
                    dist = abs(point[0] - Lpoint)
                    if dist < diff[1]:
                        diff = [color, dist]

        diff[1] = round(diff[1], 3)

        match = DuploBrick(points=bbox, center=tuple(o), color=diff[0], mask=mask)
        found.append(match)
        
        if hook == "contours":
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(draw, [c], -1, (0,0,255), 1, lineType=cv2.LINE_AA)

    valid = []
    for b1 in found:
        for b2 in found:
            if b1.center == b2.center:
                continue
            inside_points = 0
            for p in b1.points:
                if cv2.pointPolygonTest(b2.points, tuple(p), False) >= 0:
                    inside_points += 1
            if inside_points == len(b1.points):
                break
        else:
            valid.append(b1)

    found = valid

    if hook == "contours":
        for b in valid:
            cv2.drawContours(draw, [b.points], -1, (0,255,0), 2, lineType=cv2.LINE_AA)

    if hook == "lab_l":
        draw[:] = src[:, :, 0][:, :, None]
    elif hook == "lab_a":
        draw[:] = src[:, :, 1][:, :, None]
    elif hook == "lab_b":
        draw[:] = src[:, :, 2][:, :, None]

    elif hook == "colorDist":
        draw[:] = (colorDist * 255).astype('uint8')[:, :, None]
    elif hook == "lumDist":
        draw[:] = (lumDist * 255).astype('uint8')[:, :, None]
    elif hook == "trueDist":
        draw[:] = trueDist[:, :, None]

    elif hook == "canned":
        draw[:] = canned[:, :, None]

    return found

KERNEL3X3ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
def findCirclesWithinRectangles(src, whitepoint: int, rectangles: list, enforcePermuationsByColor : list = None, draw = None, hook = None) -> None:
    assert src.dtype == np.float32, "Expected argument 'src' dtype to be 'float32', got {}".format(src.dtype)
    assert not rectangles is None, "Expected list at argument 'rectangles', got None"

    if len(rectangles) == 0:    # There's nothing to do here
        return

    lumDist = 1 - abs(src[:, :, 0] - whitepoint) / 255
    lumDist8 = cv2.normalize(lumDist, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lumDistC = CLAHE.apply(lumDist8)

    for _ in range(2):
        lumDistC = cv2.GaussianBlur(lumDistC, (3, 3), 0)

    lapDist = cv2.Laplacian(lumDistC, cv2.CV_8UC1, ksize=5)
    _, lapDistD = cv2.threshold(lapDist, 80, 255, cv2.THRESH_BINARY)
    lapDistD = cv2.morphologyEx(lapDistD, cv2.MORPH_DILATE, KERNEL3X3ELLIPSE)
    
    proposals1 = cv2.HoughCircles(lumDistC, cv2.HOUGH_GRADIENT_ALT, 1, minDist=18, param1=175, param2=0.5, minRadius=0, maxRadius=16)

    proposals2 = cv2.HoughCircles(lapDist, cv2.HOUGH_GRADIENT_ALT, 1, minDist=18, param1=1200, param2=0.3, minRadius=0, maxRadius=16)

    proposals3 = cv2.HoughCircles(lapDistD, cv2.HOUGH_GRADIENT_ALT, 1, minDist=18, param1=3000, param2=0.25, minRadius=0, maxRadius=16)

    if hook == "circles" or hook == "claheCircles" \
        or hook == "laplaceCircles" or hook == "dilateCircles":
        
        draw[:] = draw // 2

    colors = [(255,255,0), (0,255,0), (0,255,255)]
    for i, circles in enumerate([proposals1, proposals2, proposals3]):
        if circles is not None:
            circles = np.squeeze(circles, axis=1)   # Why does Opencv do this???
            circles = np.uint16(np.around(circles))

            for b in rectangles:
                for c in circles:
                    if cv2.pointPolygonTest(b.points, (c[0],c[1]), False) >= 0:
                        b.circles.append(c)

                        if (hook == "claheCircles" and i == 0) \
                            or (hook == "laplaceCircles" and i == 1) \
                            or (hook == "dilateCircles" and i == 2):
                            
                            # draw the outer circle
                            cv2.circle(draw,(c[0],c[1]),c[2], colors[i],1)
                            # draw the center of the circle
                            cv2.circle(draw,(c[0],c[1]),2, colors[i],1)
            
    _draw = None
    if hook == "merge":
        _draw = np.zeros(draw.shape[:2], dtype='uint8')

    for b in rectangles:
        c_list = mergeOverlappingCircles(b.circles, lumDist8.shape[:2], _draw, hook)
        if hook == "circles":
            for cx in c_list:
                o = (int(cx[0][0]), int(cx[0][1]))
                # draw the outer circle
                cv2.circle(draw, o, int(cx[1]), (255, 255, 255), 1)
                # draw the center of the circle
                cv2.circle(draw, o, 2, (255, 255, 255), 1)

        c = len(c_list)

        permutations = VALID_SIZES
        if enforcePermuationsByColor is not None:
            permutations = enforcePermuationsByColor[b.color]
        
        if c in permutations:
            b.circleCount = c
        else:
            # panik
            for j, current in enumerate(permutations):
                previous = permutations[j-1] if j > 0 else -current
                after = permutations[j+1] if j < len(permutations)-1 else 4*permutations[j]

                lower = (previous + current) // 2
                upper = (current + after) // 2
                if lower < c <= upper:
                    b.circleCount = current
                    break

            else:
                # PANIK
                b.circleCount = -1
    
    
    if hook == "lumDist2":
        draw[:] = (lumDist * 255).astype('uint8')[:, :, None]
    elif hook == "clahe":
        draw[:] = lumDistC[:, :, None]
    elif hook == "laplace":
        draw[:] = lapDist[:, :, None]
    elif hook == "dilate":
        draw[:] = lapDistD[:, :, None]
    elif hook == "merge":
        draw[:] = _draw[:, :, None] * 255



class BrickenatorDetector:
    def __init__(self):
        pass

    def name(self):
        return "Approach A"

    def __call__(self, src, hook = None):
        debug = None
        if isinstance(hook, str):
            if hook == "output":
                hook = None
            else:
                debug = src.copy()

        out = src.copy()
        test_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype('float32')

        whitepoint = 170
        
        bricks = findColouredRectangles(test_lab, whitepoint, draw=debug, hook=hook)

        findCirclesWithinRectangles(test_lab, whitepoint, bricks, enforcePermuationsByColor=BRICK_PERMUTATIONS, draw=debug, hook=hook)

        if hook == "input":
            debug = src

        return bricks, out, debug

def labelBricksWithBBox(src, bricks, translation = None):
    out = src.copy()

    for b in bricks:
        color = (0,255,0) if b.circleCount > 0 else (0,0,255)

        try:
            name = translation[b.color]
        except KeyError:
            name = b.color.capitalize()

        try:
            label = " ".join([name, translation[str(b.circleCount)]])
        except KeyError:
            label = " ".join([name, str(b.circleCount)])

        l_size = cv2.getTextSize(label, 0, fontScale=0.75, thickness=2)

        cv2.drawContours(out, [b.points], -1, color, 2)

        new_center = (b.center[0] - l_size[0][0]//2, b.center[1] + l_size[0][1]//2)

        cv2.putText(out, label, new_center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out

### Are we overfitting?
# Single contrast                               :   94.83%
# Input-CLAHE-Laplacian Multicontrast, Maximum  :   96.84%
# Input-CLAHE-Laplacian Multicontrast, Union    :   98.56%
# CLAHE-Laplacian-Dilate Multicontrast, Union   :   99.71%
#       175, 175, 175 / 0.5, 0.3, 0.5 (?)
# CLAHE-Laplacian-Dilate Multicontrast, Union   :   99.71%(?)
#       1200, 3000, 175 / 0.3, 0.23, 0.5