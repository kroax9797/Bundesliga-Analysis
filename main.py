from utils import read_video , save_video
from trackers import Tracker
from tqdm import tqdm
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import cameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    # Read Video 
    video_frames = read_video('./input_videos/08fd33_4.mp4')
    print("Video read")

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    #Track the players , refereees and ball
    tracks = tracker.get_object_tracks(video_frames , read_from_stub=True , stub_path='stubs/tracks_stubs.pkl')
    
    # Get object positions
    tracker.add_positions_to_tracks(tracks)
    print("tracked")

    #Camera movement 
    camera_movement_estimator = cameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames , read_from_stubs=True , stub_path='./stubs/camera_movement.pkl')

    # Adjusting positions relative to camera
    camera_movement_estimator.adjust_positions_to_tracks(tracks , camera_movement_per_frame)

    # Transform the view 
    view_transformer = ViewTransformer()
    view_transformer.transformed_position_to_tracks(tracks)

    # Estimate speed and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.speed_and_distance_to_track(tracks)

    # Interpolate the ball positions : 
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    ### Code for extracting a sample player image from the video
    # for track_id , bbox in tracks['players'][0].items():
    #     frame = video_frames[0]
    #     bbox = bbox['bbox']

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]) , int(bbox[0]):int(bbox[2])]

    #     cv2.imwrite("./output_videos/player_image.jpg" , cropped_image)
    #     break

    #Assign Team Colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0] , tracks['players'][0])
    
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign ball to player
    player_assigner = PlayerBallAssigner()
    ball_possesion = []

    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track , ball_bbox)

        # Check if the ball_bbox is valid
        if ball_bbox is not None:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Check if assigned_player is valid
        if assigned_player != -1 and assigned_player in player_track:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            ball_possesion.append(tracks['players'][frame_num][assigned_player]['team'])
        else : 
            ball_possesion.append(ball_possesion[-1])

        

    #Draw output 
    output_video_frames = tracker.draw_annotations(video_frames , tracks , ball_possesion)
    print("annotated")

    #Draw camera movment
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    print("Camera movment annotated")

    # Draw speed and distance 
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames , tracks)

    # Save Video
    save_video(output_video_frames , 'output_videos/output_video.avi')
    print("Video saved")
    

if __name__ == "__main__":
    main() 