# Bundesliga-Analysis
 AI ML model to analyse the bundesliga football clips .
 
Welcome to the Football Player Detection and Analysis project! This repository contains a YOLO-based model for detecting football players in broadcast camera angle streams. Along with player detection, the project includes additional features for analyzing player movement, speed, and distance. This README provides an overview of the repository, installation instructions, and a guide on how to run the code.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Components](#components)
- [Usage](#usage)
- [Contributing](#contributing)
- [Useful Links](#UsefulLinks)

## Introduction
This project leverages the YOLO (You Only Look Once) model for real-time object detection to detect football players in a broadcast camera angle. Additional components include:
- A model to assign the ball to the nearest player.
- A speed and distance estimator.
- Object tracking for maintaining consistency across frames.
- View Transformer
- Player Ball Assigner
- Team Assigner

## Requirements
- Python 3.7 or higher
- OpenCV
- Ultralytics
- Additional libraries: numpy, scipy, etc.

## Installation
1. Clone the repository to your local machine:
   git clone https://github.com/kroax9797/Bundesliga-Analysis.git
   cd Bundesliga-Analysis
2. Install the required packages
   pip install -r requirements.txt

## Components
### Description and functionality of various directories 
1. main.py: Entry point for running the full code. This script integrates all the functionalities and produces output videos.
2. camera_movement : Detects the camera movement in the broadcast view using cv2 and its extraction of feature method .
3. input_videos : You can put the videos you wish to analyze over here .
4. models : Contains trained models , best.pt referes to my trained yolo model for the dataset provided in the training directory .
5. output_video : Your analyzed video will be saved over here . A sample video is also provided with the source code .
6. player_ball_assigner : Assigns the ball to the nearest player to it . Calculates the possession stats as well .
7. runs : Your yolo inferences will be stored over here .
8. speed_and_distance_estimator : Estimates speed and distance covered by a player using view transformer .
9. stubs : caches the tracks for a faster run .
10. team_assigner : Assigns the team to a player using k-means clustering algorithm with 2 clusters over here .
11. trackers : tracks and stores all the objects detected by the yolo model(best.pt over here) .
12. training : the yolo model is trainied on this dataset for finetuning it to football scenarios in broadcast camera angle .
13. utils : contains useful functions
14. view_transformer : Transforms the view and helps in calculating real distance metrics from the tilted view of the camera .

## Usage 
Run the main.py script to execute the full pipeline:
    python main.py

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your suggestions or bug reports.

## Useful Links : 
- Bundesliga Dataset : https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips
- Training dataset for Yolo : https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1


