#ifndef IPCV_DETECTOR_H_
#define IPCV_DETECTOR_H_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hough.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
namespace ipcv {
class Detector {
 public:
  Detector(string file_name);
  void WriteImage();
  void Detect();

 private:
  CascadeClassifier cascade;
  Mat frame;
  void CalcTPR(int num_detections, Rect* detected, int num_dart_boards,
               Rect* ground_truth);
  void CalcF1Scores(char* ground_truth_filename, std::vector<Rect> detections);
  bool PointInRect(Point2f pt, Rect rectangle);
  std::vector<Point2f> getFFDPoints(Mat img_scene, Mat img_object);
  int getFFDScore(Rect rectangle, Mat img_scene, Mat img_object);
};
}  // namespace ipcv
#endif
