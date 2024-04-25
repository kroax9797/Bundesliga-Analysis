import cv2
from matplotlib import pyplot as plt

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True :
        ret , frame = cap.read()
        if not ret : 
            break
        frames.append(frame)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame_rgb)
        # plt.show()
    return frames

def save_video(output_video_frames, output_video_path):
    if len(output_video_frames) == 0:
        raise ValueError("No frames to save.")

    # Proper way to generate fourcc
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using 'XVID' as the codec
    
    # Define frame size (width, height) from the first frame
    # print("TYPES")
    # for ff in output_video_frames: 
    #     print("hehe")
    #     print(ff)
    # exit()
    
    frame_height, frame_width = output_video_frames[0].shape[:2]
    
    # Set up VideoWriter with correct arguments
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    
    # Write each frame to the video
    for frame in output_video_frames:
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
