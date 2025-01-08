import cv2
import numpy as np
from utils.bbox_utils import measure_distance
from sklearn.cluster import KMeans


class IsBall:
    def __init__(self, min_ball_size=5, max_ball_size=20, color_threshold=(20, 100, 100, 30, 255, 255), stationary_threshold=5, penalty_threshold=1.5):
        self.min_ball_size = min_ball_size
        self.max_ball_size = max_ball_size
        self.color_threshold = color_threshold
        self.stationary_threshold = stationary_threshold
        self.penalty_threshold = penalty_threshold

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def _is_valid_ball(self, ball_bbox):
        min_ball_size = 5
        max_ball_size = 20
        ball_width = ball_bbox[2] - ball_bbox[0]
        ball_height = ball_bbox[3] - ball_bbox[1]
        return min_ball_size < ball_width < max_ball_size and min_ball_size < ball_height < max_ball_size
    
    def get_ball_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_ball_cluster = max(set(corner_clusters),key=corner_clusters.count)
        ball_cluster = 1 - non_ball_cluster
        ball_color = kmeans.cluster_centers_[ball_cluster]

        return ball_color


    def _filter_by_color(self, frame, object_bbox):
        x1, y1, x2, y2 = map(int, object_bbox)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        object_mask = hsv_frame[y1:y2, x1:x2]
        lower_ball_color = self.get_ball_color(frame, object_bbox) 
        upper_ball_color = self.get_ball_color(frame, object_bbox)
        ball_mask = cv2.inRange(object_mask, lower_ball_color, upper_ball_color)
        return cv2.countNonZero(ball_mask) > 0


    def _classify_ball(self, ball_bbox, frame, tracked_positions):
        score = 0
        if self._is_valid_ball(ball_bbox):
            score += 1
        if self._filter_by_color(frame, ball_bbox):
            score += 1

        return score > 1
