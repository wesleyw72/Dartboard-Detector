#ifndef IPCV_HOUGH_H_
#define IPCV_HOUGH_H_
#include <opencv/cv.h>       //you may need to
#include <opencv/cxcore.h>   //depending on your machine setup
#include <opencv/highgui.h>  //adjust import locations
#include <stdio.h>
using namespace cv;
#define RAD(x, y, radius, numX, numY) ((radius * numX * numY) + (y * numX) + x)
namespace ipcv {
class Hough {
 public:
  // Based on input params, determines whether circles are in image.
  bool Circles(int min_radius, int max_radius, int step, int threshold);
  // Based on input thesholds, determines whether lines are in image.
  bool Lines(int hough_space_thresh, int angle_thresh, int count_thresh,
             int accum_thresh);
  Hough(cv::Mat input, int thresholdMagVal, cv::Mat outSection, int sectionID);

 private:
  int sectionID;
  cv::Mat original;
  cv::Mat input;
  cv::Mat gray_input;
  cv::Mat threshold_magnitude_image;
  cv::Mat gradient;
  cv::Mat magnitude;

  // Utility for drawing detected hough lines
  // Outputs to houghLines.jpg file
  void DrawLines(std::vector<Vec2f> lines);

  // Computes Sobel for an individual dimension.
  // Takes the image and the kernel to be applied
  void SobelIndividual(cv::Mat &input, cv::Mat kernel, cv::Mat &result);
  // Computes Sobel for both dimensions, calculates the magnitude gradient image
  // and thresholds it. Takes the threshold magnitude value.
  void Sobel(int thresholdMagVal);
  void CalcMagnitudeAndGradImage(cv::Mat xGrad, cv::Mat yGrad);
  cv::Mat Threshold(int thresholdValue, cv::Mat &image);
  cv::Mat houghLinesVotes(int angleThresh);
  cv::Mat LineWeights(int threshold, int angleThresh);
  std::vector<Vec2f> thresholdLines(int threshold, int angleThresh);
  void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurred_output);
};
}  // namespace ipcv
#endif
