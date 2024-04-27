import pickle
import cv2
import numpy as np
from utils import measure_distance
import os
from tqdm import tqdm


class cameraMovementEstimator():
    def __init__(self,frame):

        self.minimum_distance = 3

        first_frame_grayscale = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1 
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,    
            minDistance = 3 ,
            blockSize = 7 ,
            mask = mask_features
        )

        self.lk_params = dict(
            winSize=(15,15),
            maxLevel=2,
            criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

    def get_camera_movement(self , frames , read_from_stubs = False , stub_path = None):
        
        if read_from_stubs and os.path.exists(stub_path):
            with open(stub_path , 'rb') as f : 
                print("Camera stubs found !")
                return pickle.load(f)

        cameraMovement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in tqdm(range(1,len(frames)) , desc = "Creating camera movement stubs : "):
            frame_gray = cv2.cvtColor(frames[frame_num] , cv2.COLOR_BGR2GRAY)
            new_features , _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
            max_distance = 0
            camera_movement_x , camera_movement_y = 0,0

            for i , (old,new) in enumerate(zip(old_features,new_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = new_features_point[0] - old_features_point[0]
                    camera_movement_y = new_features_point[1] - old_features_point[1]
                
            if max_distance > self.minimum_distance:
                cameraMovement[frame_num] = [camera_movement_x , camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray , **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None : 
            with open(stub_path , 'wb') as file : 
                pickle.dump(cameraMovement , file)

        return cameraMovement
    
    def draw_camera_movement(self , frames , camera_movement_per_frame):
        output_frames = []

        for frame_num , frame in tqdm(enumerate(frames) , desc = "Adding camera movment"):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay , (0,0) , (500,100) , (255,255,255) , -1)
            alpha=0.6
            cv2.addWeighted(overlay , alpha , frame , 1-alpha , 0 , frame)

            dx , dy = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame , f"Camera Movement X : {dx:.2f}" , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0) , 3)
            frame = cv2.putText(frame , f"Camera Movement Y : {dy:.2f}" , (10,60) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0) , 3)

            output_frames.append(frame)

        return output_frames
    
    def adjust_positions_to_tracks(self , tracks , camera_movement_per_frame):
        for object , object_tracks in tracks.items():
            for frame_num , track in enumerate(object_tracks):
                for track_id , track_info in track.items():
                    position = track_info['position']
                    position_adjusted = (position[0]-camera_movement_per_frame[frame_num][0] , position[1]-camera_movement_per_frame[frame_num][1])
                    tracks[object][frame_num][track_id]['position'] = position_adjusted
        pass