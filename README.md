# HandTrackingProject

**Hand Detection and Finger Counting Application**  

This project performs real-time hand detection and finger counting using **OpenCV** and **Mediapipe**. It detects hand positions from the camera feed and displays the number of fingers on the screen in each frame.  

## Features
- Real-time hand and finger detection  
- Display finger count visually with overlays  
- FPS (Frames Per Second) display for performance monitoring  
- Detects thumb and 4 fingers for the right hand (can be adapted for the left hand)

![Demo Image]("C:\Users\ahsen\Pictures\Screenshots\ekrangoruntusu.png")

## Setup
```bash
git clone https://github.com/username/HandTrackingProject.git
cd HandTrackingProject
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install opencv-python mediapipe

