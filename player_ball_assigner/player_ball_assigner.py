import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance, get_bbox_height

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        avg_player_bbox_height = 0
        if len(players) > 0:
            for _, player in players.items():
                player_bbox = player['bbox']
                avg_player_bbox_height += get_bbox_height(player_bbox)

            avg_player_bbox_height /= (len(players)*2)
        else:
            avg_player_bbox_height = 70

        self.max_player_ball_distance = avg_player_bbox_height
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player