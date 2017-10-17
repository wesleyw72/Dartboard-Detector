# Image Processing and Computer Vision
A dartboard detector written in C++ in collaboration with joromybouk.
Makes use of a Hough transform for shape detection with a Sobel filter used for edge detection, implementations are included.
Also used Viola-Jones and SURF for detecting areas of interest.

## Requirements
* OpenCV (developed on 2.4.13.1 but other versions may work)
## How to run
* Compile using Makefile. (Makefile uses pkg-config)
* Run ./viola dart1.jpg (replace number for other images).
* See bounding boxes drawin in detected.jpg
## Performance
* Manages to correctly detect most dartboards.
* Runs in a reasonable amount of time (few seconds on a MacBook Pro 2017).
* Overall Precision = 0.68
* Overall Recall = 0.85
* Total F1 Score = 0.76
## Report
A small report on this project is available upon request.
