from is_ball.is_ball import IsBall
from pass_stats_tracker import PassStatsTracker
from player_heatmap.player_heatmap import PlayerHeatmap
from tactical_analysis.pass_network import PassNetwork
from tactical_analysis.space_occupancy_analyzer import SpaceOccupancyAnalyzer
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/yolov8.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs_08fd33_4.pkl')
    # Get object positions with kalman filter
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=False,
                                                                                stub_path='stubs/camera_movement_stub_bayernv2.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #for frame_num in range(len(tracks["ball"])):
    #    print(f"Ball transformed position at frame {frame_num}: {tracks['ball'][frame_num].get(1, {}).get('position_transformed', 'Missing')}")

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    """
    # Initialize IsBall
    is_ball_detector = IsBall()

    for frame_num, ball_track in enumerate(tracks['ball']):
        for track_id, track_info in ball_track.items():
            ball_bbox = track_info['bbox']
            tracked_positions = [track_info['position_transformed']] 
            frame = video_frames[frame_num]

            is_ball = is_ball_detector._classify_ball(ball_bbox, frame, tracked_positions)

            tracks['ball'][frame_num][track_id]['is_valid_ball'] = is_ball
            #validity = "valid" if is_ball else "invalid"
            #print(f"Frame {frame_num}: ball is {validity}")

    # Validate and remove invalid ball positions
    for frame_num, ball_track in enumerate(tracks['ball']):
        for track_id, track_info in list(ball_track.items()):
            if not track_info.get('is_valid_ball', False):
                del tracks['ball'][frame_num][track_id]
    

    # Interpolate missing ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Optional: Recheck ball validity if necessary
    for frame_num, ball_track in enumerate(tracks['ball']):
        for track_id, track_info in ball_track.items():
            ball_bbox = track_info['bbox']
            tracked_positions = [track_info['position_transformed']] 
            frame = video_frames[frame_num]

            is_ball = is_ball_detector._classify_ball(ball_bbox, frame, tracked_positions)
            tracks['ball'][frame_num][track_id]['is_valid_ball'] = is_ball
            validity = "valid" if is_ball else "invalid"
            print(f"Frame {frame_num}: ball is {validity}")
    """

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    # Initialize Tactical Analysis Components
    pass_network = PassNetwork()
    #space_occupancy_analyzer = SpaceOccupancyAnalyzer()


    # Pass Network Analysis
    pass_network_graph = pass_network.construct_pass_network(tracks['players'], team_ball_control)
    pass_network.visualize(pass_network_graph, 'output_images/pass_network.png')

    # Space Occupancy Analysis
    #space_occupancy_maps = space_occupancy_analyzer.analyze_space_control(tracks['players'])
    #space_occupancy_analyzer.visualize_space_occupancy(space_occupancy_maps, 'output_images/space_occupancy.png')


    # Initialize PassStatsTracker
    pass_stats_tracker = PassStatsTracker()

   # Draw stats on each frame with a semi-transparent background
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    line_type = 1

    for frame_num, frame in enumerate(video_frames):
        # Update pass stats for the current frame
        pass_stats_tracker.update_frame_stats(tracks, frame_num, team_ball_control)
        live_stats = pass_stats_tracker.get_live_stats()

        # Determine the background rectangle dimensions
        frame_height = frame.shape[0] - 100
        y_offset = frame_height - (len(live_stats) * 20 + 10)
        rect_height = len(live_stats) * 20 + 20
        rect_width = 450 

        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, y_offset - 10),
            (10 + rect_width, y_offset + rect_height - 10),
            (255, 255, 255),
            -1,
        )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw stats text on top of the semi-transparent background
        for line in live_stats:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                font,
                font_scale,
                font_color,
                line_type,
            )
            y_offset += 18  # Adjust spacing between lines

    pass_stats_tracker.print_stats()


    # Generate heatmaps for each team
    teams = set(player['team'] for frame in tracks['players'] for player in frame.values() if 'team' in player)
    frame_size = (video_frames[0].shape[0], video_frames[0].shape[1])
    pitch_image_path = "images/football_pitch.png"

    for team in teams:
        # Initialize heatmap for the team
        team_heatmap = PlayerHeatmap(frame_size, pitch_image_path=pitch_image_path)
        team_heatmap.update_heatmap(tracks, team=team)
        output_path = f"output_images/team_{team}_heatmap.png"
        team_heatmap.save_heatmap_on_pitch(output_path)
        print(f"Heatmap for team {team} saved at {output_path}")

    # Generate heatmap for all players
    all_players_heatmap = PlayerHeatmap(frame_size, pitch_image_path=pitch_image_path)
    all_players_heatmap.update_heatmap(tracks)
    output_path_all = "output_images/all_players_heatmap.png"
    all_players_heatmap.save_heatmap_on_pitch(output_path_all)
    print(f"Heatmap for all players saved at {output_path_all}")



    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)


    # Save video
    save_video(output_video_frames, 'output_videos/08fd33_4_v9.avi')

if __name__ == '__main__':
    main()