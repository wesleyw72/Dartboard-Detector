#include "hough.h"
namespace ipcv {
Hough::Hough(cv::Mat input, int thresholdMagVal, cv::Mat outSection,
             int sectionID) {
  this->sectionID = sectionID;
  original = outSection;
  this->input = cv::Mat(input.clone());
  cvtColor(this->input, gray_input, CV_BGR2GRAY);
  equalizeHist(gray_input, gray_input);
  Sobel(thresholdMagVal);
}
void Hough::SobelIndividual(cv::Mat &input, cv::Mat kernel, cv::Mat &result) {
  result.create(input.size(), input.type());
  int kernel_radius_x = (kernel.size[0] - 1) / 2;
  int kernel_radius_y = (kernel.size[1] - 1) / 2;
  cv::Mat padded_input;
  cv::copyMakeBorder(input, padded_input, kernel_radius_x, kernel_radius_x,
                     kernel_radius_y, kernel_radius_y, cv::BORDER_REPLICATE);

  // now we can do the convoltion
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double sum = 0.0;
      for (int m = -kernel_radius_x; m <= kernel_radius_x; m++) {
        for (int n = -kernel_radius_y; n <= kernel_radius_y; n++) {
          // find the correct indices we are using
          int imagex = i + m + kernel_radius_x;
          int imagey = j + n + kernel_radius_y;
          int kernelx = m + kernel_radius_x;
          int kernely = n + kernel_radius_y;
          // get the values from the padded image and the kernel
          int imageval = (int)padded_input.at<uchar>(imagex, imagey);
          double kernalval = kernel.at<double>(kernelx, kernely);
          // do the multiplication
          sum += imageval * kernalval;
        }
      }
      // set the output value as the sum of the convolution
      // divide by 8 and add 128 to prevent arithmetic errors
      result.at<uchar>(i, j) = (uchar)((sum / 8) + 128);
    }
  }
}

void Hough::Sobel(int thresholdMagVal) {
  cv::Mat blurredImage;
  // Calc x and y kernel for sobel
  GaussianBlur(gray_input, 3, blurredImage);
  cv::Mat kernelX = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  cv::Mat kernelY = kernelX.t();
  cv::Mat xGrad, yGrad;
  SobelIndividual(blurredImage, kernelX, xGrad);
  SobelIndividual(blurredImage, kernelY, yGrad);
  gradient.create(blurredImage.size(), blurredImage.type());
  CalcMagnitudeAndGradImage(xGrad, yGrad);
  threshold_magnitude_image = Threshold(thresholdMagVal, magnitude);
}
void Hough::CalcMagnitudeAndGradImage(cv::Mat xGrad, cv::Mat yGrad) {
  // Calculates magnitude and gradient image from the x and y gradient images
  magnitude.create(xGrad.size(), xGrad.type());
  for (int i = 0; i < xGrad.rows; i++) {
    for (int j = 0; j < xGrad.cols; j++) {
      magnitude.at<uchar>(i, j) = sqrt(
          ((yGrad.at<uchar>(i, j) - 127)) * ((yGrad.at<uchar>(i, j) - 127)) +
          ((xGrad.at<uchar>(i, j) - 127)) * ((xGrad.at<uchar>(i, j) - 127)));
      float val;
      if (xGrad.at<uchar>(i, j) - 127 == 0) {
        val = 0.00001;
        // avoid dividing by 0
      } else {
        val = xGrad.at<uchar>(i, j) - 127;
      }
      gradient.at<uchar>(i, j) =
          (atan((yGrad.at<uchar>(i, j) - 127) / (val)) * 40.6051) + 128;
    }
  }
}
cv::Mat Hough::Threshold(int threshold_value, cv::Mat &image) {
  // set any values above threshold to 255, rest to 0
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      if (image.at<uchar>(i, j) > threshold_value) {
        image.at<uchar>(i, j) = 255;
      } else {
        image.at<uchar>(i, j) = 0;
      }
    }
  }
  return image;
}
bool Hough::Circles(int min_radius, int max_radius, int step, int threshold) {
  bool correct_box_detection = false;
  int *votes;
  votes = (int *)calloc(threshold_magnitude_image.cols *
                            threshold_magnitude_image.rows * (max_radius + 1) *
                            sizeof(int),
                        sizeof(int));
  for (int i = 0; i < threshold_magnitude_image.rows - 1; i++) {
    for (int j = 0; j < threshold_magnitude_image.cols - 1; j++) {
      if (threshold_magnitude_image.at<uchar>(i, j) == 255) {
        float Cgradient = (float)((gradient.at<uchar>(i, j) - 128) / 40.6051);
        for (int r = min_radius; r <= max_radius; r += step) {
          int x0 = j + (int)(r * cos(Cgradient));
          int y0 = i + (int)(r * sin(Cgradient));
          int x0m = j - (int)(r * cos(Cgradient));
          int y0m = i - (int)(r * sin(Cgradient));
          if (!((x0m < 0 || y0 >= threshold_magnitude_image.rows || y0 < 0 ||
                 x0m >= threshold_magnitude_image.cols))) {
            votes[RAD(x0m, y0, r, threshold_magnitude_image.cols,
                      threshold_magnitude_image.rows)]++;
          }
          if (!(x0 >= threshold_magnitude_image.cols ||
                y0 >= threshold_magnitude_image.rows || y0 < 0 || x0 < 0)) {
            (votes[RAD(x0, y0, r, threshold_magnitude_image.cols,
                       threshold_magnitude_image.rows)])++;
          }
          if (!(y0m < 0 || x0 >= threshold_magnitude_image.cols ||
                y0m >= threshold_magnitude_image.rows || x0 < 0)) {
            votes[RAD(x0, y0m, r, threshold_magnitude_image.cols,
                      threshold_magnitude_image.rows)]++;
          }
          if (!(x0m < 0 || y0m < 0 || x0m >= threshold_magnitude_image.cols ||
                y0m >= threshold_magnitude_image.rows)) {
            (votes[RAD(x0m, y0m, r, threshold_magnitude_image.cols,
                       threshold_magnitude_image.rows)])++;
          }
        }
      }
    }
  }
  cv::Mat hough_space;
  hough_space.create(threshold_magnitude_image.size(),
                     threshold_magnitude_image.type());
  for (int i = 0; i < threshold_magnitude_image.rows; i++) {
    for (int j = 0; j < threshold_magnitude_image.cols; j++) {
      int position_total = 0;
      for (int r = min_radius; r <= max_radius; r += step) {
        position_total += votes[RAD(j, i, r, threshold_magnitude_image.cols,
                                    threshold_magnitude_image.rows)];
      }
      hough_space.at<uchar>(i, j) = (uchar)position_total;
    }
  }
  int correctCount = 0;
  for (int i = 0; i < threshold_magnitude_image.rows; i++) {
    for (int j = 0; j < threshold_magnitude_image.cols; j++) {
      if (hough_space.at<uchar>(i, j) > 0) {
        for (int r = min_radius; r <= max_radius; r += step) {
          if (votes[RAD(j, i, r, threshold_magnitude_image.cols,
                        threshold_magnitude_image.rows)] > threshold) {
            correctCount++;
          }
        }
      }
    }
  }
  if (correctCount > 0) {
    correct_box_detection = true;
  }
  free(votes);
  return correct_box_detection;
}
cv::Mat Hough::houghLinesVotes(int angle_thresh) {
  // Function: votes for hough lines in an image, returns the hough space
  // Calculate longest possible line size
  int dim = ceil(hypot(gradient.rows, gradient.cols));
  cv::Mat houghSp;
  int size[] = {dim, 360};
  houghSp.create(2, size, CV_32SC1);
  houghSp = Scalar(0);
  for (int i = 0; i < gradient.rows; i++) {
    for (int j = 0; j < gradient.cols; j++) {
      if (threshold_magnitude_image.at<uchar>(i, j) == 255) {
        // convert dartboard info
        int Cgradient =
            ((gradient.at<uchar>(i, j) - 128) / 40.6051) * 180 / CV_PI;
        for (int theta = Cgradient - angle_thresh;
             theta <= Cgradient + angle_thresh; theta += 1) {
          int rho =
              (j * cos(theta * CV_PI / 180) + i * sin(theta * CV_PI / 180));
          if (rho >= 0) {
            houghSp.at<int>(rho, theta + 180) += 1;
          }
        }
      }
    }
  }
  return houghSp;
}
cv::Mat Hough::LineWeights(int threshold, int angle_thresh) {
  // Function: Accumulates the lines, in order to find the points with the most
  // intersections
  std::vector<Vec2f> lines = thresholdLines(threshold, angle_thresh);
  cv::Mat houghWeights;
  houghWeights.create(input.size(), input.type());
  houghWeights = Scalar(0);
  std::vector<Vec2f> cartLines;
  // create equations of all the linees
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt;
    double a = cos((theta - 180) * CV_PI / 180),
           b = sin((theta - 180) * CV_PI / 180);
    pt.x = cvRound(a * rho);
    pt.y = cvRound(b * rho);
    if (pt.y != 0) {
      float grad = -(float)pt.x / (float)pt.y;
      if ((int)grad != 0) {
        float c = pt.y - grad * pt.x;
        Vec2f temp;
        temp[0] = grad;
        temp[1] = c;
        cartLines.push_back(temp);
      }
    }
  }
  // vote for each point that the line goes through
  for (int i = 0; i < input.cols; i++) {
    for (int l = 0; l < cartLines.size(); l++) {
      int y = cvRound(cartLines[l][0] * i + cartLines[l][1]);
      if (0 < y && y < input.rows) {
        houghWeights.at<uchar>(y, i)++;
      }
    }
  }

  return houghWeights;
}
std::vector<Vec2f> Hough::thresholdLines(int threshold, int angle_thresh) {
  // Function: Returns a vector of lines that are over a threshold in hough
  // space.
  cv::Mat houghVotes = houghLinesVotes(angle_thresh);
  std::vector<Vec2f> result;
  for (int i = 0; i < houghVotes.rows; i++) {
    for (int j = 0; j < houghVotes.cols; j++) {
      if (houghVotes.at<int>(i, j) > threshold) {
        Vec2f temp;
        temp[0] = i;
        temp[1] = j;
        result.push_back(temp);
      }
    }
  }

  return result;
}
bool Hough::Lines(int hough_space_thresh, int angle_thresh, int count_thresh,
                  int accum_thresh) {
  int count = 0;
  cv::Mat houghLWeights = LineWeights(hough_space_thresh, angle_thresh);
  for (int x = 0; x < houghLWeights.cols; x++) {
    for (int j = 0; j < houghLWeights.rows; j++) {
      if (houghLWeights.at<uchar>(j, x) > accum_thresh) {
        count++;
      }
    }
  }
  if (count > count_thresh) {
    return true;
  }
  return false;
}
void Hough::GaussianBlur(cv::Mat &input, int size, cv::Mat &blurred_output) {
  // intialise the output using the input
  blurred_output.create(input.size(), input.type());
  // create the Gaussian kernel in 1D
  cv::Mat kX = cv::getGaussianKernel(size, -1);
  cv::Mat kY = cv::getGaussianKernel(size, -1);
  // make it 2D multiply one by the transpose of the other
  cv::Mat kernel = kX * kY.t();
  // Create padded version to stop border effects
  int kernel_radius_x = (kernel.size[0] - 1) / 2;
  int kernel_radius_y = (kernel.size[1] - 1) / 2;
  cv::Mat padded_input;
  cv::copyMakeBorder(input, padded_input, kernel_radius_x, kernel_radius_x,
                     kernel_radius_y, kernel_radius_y, cv::BORDER_REPLICATE);
  // Covolution
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double sum = 0.0;
      for (int m = -kernel_radius_x; m <= kernel_radius_x; m++) {
        for (int n = -kernel_radius_y; n <= kernel_radius_y; n++) {
          // find the correct indices we are using
          int imagex = i + m + kernel_radius_x;
          int imagey = j + n + kernel_radius_y;
          int kernelx = m + kernel_radius_x;
          int kernely = n + kernel_radius_y;

          // get the values from the padded image and the kernel
          int imageval = (int)padded_input.at<uchar>(imagex, imagey);
          double kernalval = kernel.at<double>(kernelx, kernely);

          // do the multiplication
          sum += imageval * kernalval;
        }
      }
      // set the output value as the sum of the convolution
      blurred_output.at<uchar>(i, j) = (uchar)sum;
    }
  }
}
void Hough::DrawLines(std::vector<Vec2f> lines) {
  Mat workingCopy = Mat(input.clone());
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos((theta - 180) * CV_PI / 180),
           b = sin((theta - 180) * CV_PI / 180);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(workingCopy, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
  }
  char buff[100];
  sprintf(buff, "houghLines%d.jpg", sectionID);
  imwrite(buff, workingCopy);
}
}  // namespace ipcv
