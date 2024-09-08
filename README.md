# Person-detection

Project Overview
This project involves creating a robust person detection and tracking pipeline capable of identifying and tracking children and therapists in long-duration videos. The system assigns unique IDs to individuals and re-tracks them if they re-enter or become occluded. The results are visualized with bounding boxes and ID labels overlaying the video frames.

Problem Statement
The goal is to develop an optimized inference pipeline that can:

Detect children and therapists.
Assign unique IDs to detected individuals.
Track these individuals throughout the video, including handling re-entries and post-occlusion scenarios.
This pipeline is intended for use in analyzing behaviors, emotions, and engagement levels of children with Autism Spectrum Disorder and their therapists, to assist in creating tailored treatment plans.

Requirements
Python 3.7 or higher
PyTorch
Ultralytics YOLO
OpenCV
NumPy
SORT (Simple Online and Realtime Tracking)
Additional dependencies (listed in requirements.txt)
Installation
Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_directory>
Create and activate a virtual environment:

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Prepare the test videos:

Download the test video from this Google Drive link and place it in the videos/ directory.

Run the inference script:

bash
Copy code
python detectdata.py
This script will process the video and save the output with predictions to output/processed_output.mp4.

Inference Script Explanation
Device Configuration: The script initializes the YOLO model on GPU if available; otherwise, it defaults to CPU.

Model Initialization: YOLO model is loaded and prepared for inference.

SORT Tracker Initialization: SORT (Simple Online and Realtime Tracking) is used for tracking detected individuals.

Video Processing:

Frames are read from the video.
Each frame is resized and passed through the YOLO model for detection.
Detected bounding boxes are filtered to include only 'person' class.
Bounding boxes are converted to the format required by SORT.
The SORT tracker updates with new detections and assigns IDs.
Bounding boxes and IDs are drawn on the frames.
Output Video: The processed frames are saved to the specified output file.

Testing
Test the pipeline on the provided video file to ensure that it correctly detects, tracks, and labels individuals. The output video should display bounding boxes and unique IDs for each detected person.

Troubleshooting
Dependencies Issue: Ensure all required packages are installed. If any package conflicts occur, resolve them by adjusting versions or updating packages.
Video Processing Errors: Check if the video file path is correct and that the video is not corrupted.
Deliverables
Source Code:

detectdata.py (Inference script)
sort.py (SORT tracker implementation)
Test Video Outputs: Processed video saved in output/processed_output.mp4.

Requirements File: requirements.txt listing all necessary packages.

README.md: This file with detailed instructions and explanations.

Conclusion
This pipeline offers a reliable solution for detecting and tracking individuals in long-duration videos, which can be valuable for behavioral analysis and treatment planning.

