import sys
import numpy as np
import cv2
import math

class JugglingBallTracker(object):
    WHITE = (255,0,0)
    colors = [(255,0,0), (125,125,125), (0,255,0), (0,0,255), (125,125,0)]
    new = 0
    track = []
    radius = 18
    thickness = 5
    
    FIRST_CAMERA = 0
    
    def __init__(self, path_to_video=None):
        if path_to_video is None:
            path_to_video = self.FIRST_CAMERA
        
        self.current_frame = None
        self.video_source = None
        self.path_to_video = path_to_video
        self.initialize_video_input()
    
    def initialize_video_input(self):
        self.video_source = cv2.VideoCapture(self.path_to_video)
        if not self.video_source.isOpened():
            sys.exit("couldn't open video or video capture device")
        # auto closese device on teardown
    
    def read_frame(self):
        ret, self.current_frame = self.video_source.read()
        self.current_frame = self.current_frame[:, ::-1]
        if not ret:
            sys.exit("can't read frame")
        self.original_capture = self.current_frame
        return self.current_frame
    
    def crop_frame(self, x_or_y, y_or_x):
        self.current_frame = self.current_frame[x_or_y, y_or_x]
        return self.current_frame
    
    def to_grey_scale(self):
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        return self.current_frame
    
    def apply_threshold(self, threshold):
        ret, self.current_frame = cv2.threshold(self.current_frame, threshold, 255, cv2.THRESH_BINARY)
        if not ret:
            sys.exit("can't apply threshold")
        return self.current_frame
    
    def find_contours(self):
        contours, hierarchy = cv2.findContours(self.current_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def locate_balls(self, minimum_area=300, maximum_area=3000):
        contours = self.find_contours()
        locations = []
        for contour in contours:
            # Check if it is a juggling ball - this is gonna need improvement
            area = cv2.contourArea(contour)
            if minimum_area < area < maximum_area:
                locations.append((contour, cv2.minEnclosingCircle(contour)))
        
        return locations
    
    def highlight_contours(self, balls):
        contours = map(lambda x: x[0], balls)
        cv2.drawContours(self.current_frame, contours, -1, color=self.WHITE, thickness=3)
    
    def highlight_balls(self, balls):
        balls = map(lambda x: x[1], balls)
        for center, radius in balls:
            cv2.circle(self.current_frame, (int(center[0]), int(center[1])), int(radius), color=self.colors[1], thickness=3)
    
    def annotate_balls(self, balls):
        for contour, ball in balls: 
            area = cv2.contourArea(contour)
            center = ball[0]
            text = "%s" % area
            position = (int(center[0]), int(center[1]) + int(ball[1]))
            self.show_text(text, position)
    
    def show_text(self, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = .5
        color = (255,255,255)
        fontLineThickness = 1
        lineType = cv2.CV_AA
        cv2.putText(self.current_frame, text, position, font, fontSize, color, fontLineThickness, lineType)
        

"""
# What would be a good methapher?

* I would like to express pipelines of data processing
* I would like to output intermediate stepps / processed data to ease visual debugging
* I would like to track balls probabilistic
    * If a ball isn't found for a number of frames, it is discarded
    * If a new ball is found he gets a new color

# Algorithmic ideas

* filter detected balls by history.
    * If position is identical with old position, consider keepin it
    * If position change is insufficient, for some time, remove point
    * If position change is consistent with first or second aproximation of an existing ball, keep
* Check out deep learning with cv2 or alternatives
    * would be great if the old recognition algorithm can be used to train against existing movies
    * different lighting conditions
    * different balls
    * start with simplest patterns only
    * define tests how near different extraction parameters and/ or parameter changes are best 
      to get good recognition
* Could use 
"""
        
threshold = 210
minimum_area=75
maximum_area=10000

tracker = JugglingBallTracker()
# for i in range(930): tracker.read_frame()
while True:
    tracker.read_frame()
    # tracker.crop_frame(0:480, 100:700)
    tracker.to_grey_scale()
    tracker.apply_threshold(threshold)
    # threshold %= 255
    # threshold += 1
    # REFACT: locate should have access to some part of the history of located balls, 
    # and use it to detect balls.
    balls = tracker.locate_balls(minimum_area, maximum_area)
    tracker.highlight_contours(balls)
    tracker.highlight_balls(balls)
    tracker.annotate_balls(balls)
    
    tracker.show_text("threshold %s, minimum_area %s, maximum_area %s" % (threshold, minimum_area, maximum_area), (15, len(tracker.current_frame) - 20))
    
#     if len(locations) < len(track):
#         print 'problem'
#         min_distances = []
#         for i in track:
#             x,y = i[1][0], i[1][1]
#             distance = []
#             for c in locations:
#                 x1, y1 = c[0], c[1]
#                 distance.append(math.sqrt((x-x1)**2 + (y-y1)**2))
#             min_distances.append(min(distance))
#         index = min_distances.index(max(min_distances))
#         locations.append(track[index][1])
#     if new == 0:
#         for i in range(len(locations)):
#             track.append((colors[i], locations[i]))
#         new = 1
#         num_balls = len(locations)
#     else:
#         new_track = []
#         for i in locations:
#             x,y = i[0], i[1]
#             distance = []
#             for c in track:
#                 x1, y1 = c[1][0], c[1][1]
#                 distance.append(math.sqrt((x-x1)**2+(y-y1)**2))
#             index = distance.index(min(distance))
#             new_track.append((track[index][0], (x,y)))
#         track = new_track
#
#     for i in track:
#         center = int(i[1][0]),int(i[1][1])
#         cv2.circle(img, center, radius, i[0], thickness)
    # cv2.imshow('Detected Balls', img)
    cv2.imshow('Detected Balls', tracker.current_frame)
    # cv2.waitKey(100) # delay in milliseconds
cv2.destroyAllWindows()
# cv2.release(cap)
