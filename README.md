# PiVision: Raspberry Pi Facial Analysis for Marketing

**Project:** [SE2025] - [Final Project]

## Project Summary

PiVision is a Raspberry Pi-based system designed to analyze customer demographics in a retail setting.  It uses a camera to capture a live video feed and performs facial analysis using OpenCV and DeepFace.

## Objectives

* To demonstrate the feasibility of using inexpensive equipment such as a Raspberry Pi for facial recognition in businesses.
* To explore the potential applications of facial analysis in understanding customer behaviour.
* To gain practical experience with computer vision techniques.

## System Design

* **Input:**  Live video stream from a connected camera.
* **Processing:**  Raspberry Pi running OpenCV for facial detection and DeepFace for analysis.
* **Output:**  Demographic data (age, gender, race, emotion) displayed directly on the video feed.

## Dependencies

* **Via requirements.txt:**`pip install -r requirements.txt`

* **OpenCV:**  `pip install opencv-python`
* **DeepFace:** `pip install deepface`
  
## Usage Instructions

1.  Clone the repository: `git clone [repository URL]`
2.  Install the required libraries (see Dependencies above).
3.  Run the main script: `python face_metrics.py`

##  Ethical Note

This project is solely for educational purposes.  The ethical implications and legal requirements of facial recognition technology should be carefully considered before any real-world deployment.

## Features

* **Real-time Face Detection:** Identifies faces in the camera feed.
* **Demographic Analysis:** Predicts age, gender, and race of detected individuals.
* **Emotion Recognition:**  Detects the dominant emotion expressed by the person.
* **Direct Visualization:**  Displays the analysis results (age, gender, race, emotion) directly on the video feed.

## Limitations

* **Accuracy:** The accuracy of the analysis may vary depending on lighting conditions, camera quality, and facial expressions.
* **Performance:** Processing speed may be limited by the Raspberry Pi's hardware capabilities.
* **Ethical Considerations:**  The use of facial recognition technology raises privacy concerns.  This project is for educational purposes and should be used responsibly.
