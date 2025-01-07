import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from utils import get_foot_position, get_center_of_bbox, measure_distance

class PassStatsTracker:
    def __init__(self):
        self.pass_stats = {}

    def initialize_player_stats(self, player_id):
        if player_id not in self.pass_stats:
            self.pass_stats[player_id] = {
                'total_passes': 0,
                'successful_passes': 0,
                'received_passes': 0
            }

    def get_live_stats(self):
        stats_text = []
        for player_id, stats in self.pass_stats.items():
            stats_text.append(
                f"Player {player_id}: Passes: {stats['total_passes']} | Successful: {stats['successful_passes']} | Received: {stats['received_passes']}"
            )
        return stats_text
    
    # Print pass statistics for all players.
    def print_stats(self):
        for player_id, stats in self.pass_stats.items():
            print(
                f"Player {player_id}: Total Passes = {stats['total_passes']}, "
                f"Successful Passes = {stats['successful_passes']}, "
                f"Received Passes = {stats['received_passes']}"
            )
    
    def update_frame_stats(self, tracks, frame_num, team_ball_control):
        if frame_num >= len(tracks['players']):
            return

        assigned_player_id = None
        for player_id, player_data in tracks['players'][frame_num].items():
            if player_data.get('has_ball', False):
                assigned_player_id = player_id
                break

        if assigned_player_id is None:
            return

        self.initialize_player_stats(assigned_player_id)

        # Check if possession changes to another player
        if frame_num + 1 < len(tracks['players']):
            for other_player_id, other_player_data in tracks['players'][frame_num + 1].items():
                if other_player_data.get('has_ball', False) and assigned_player_id != other_player_id:
                    ball_speed = tracks['ball'][frame_num].get(1, {}).get('speed', 0)
                    if ball_speed > 5:
                        self.initialize_player_stats(other_player_id)
                        
                        if (
                            tracks['players'][frame_num][assigned_player_id]['team']
                            == tracks['players'][frame_num + 1][other_player_id]['team']
                        ):
                            self.pass_stats[assigned_player_id]['successful_passes'] += 1
                            self.pass_stats[other_player_id]['received_passes'] += 1

                        self.pass_stats[assigned_player_id]['total_passes'] += 1
                        break

    
