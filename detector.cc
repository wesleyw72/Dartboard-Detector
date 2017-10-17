#include "detector.h"
namespace ipcv {
Detector::Detector(string file_name) {
  frame = imread(file_name, CV_LOAD_IMAGE_COLOR);
  string cascade_name = "dartcascade/cascade.xml";
  if (!cascade.load(cascade_name)) {
    printf("--(!)Error loading\n");
    exit(1);
  };
}
void Detector::WriteImage() { imwrite("detected.jpg", frame); }
// Utility function for calculating F1 scores from ground truth.
void Detector::CalcF1Scores(char* ground_truth_filename,
                            std::vector<Rect> detections) {
  FILE* fid;
  int x, y, w, h, numDartBoards;
  fid = fopen(ground_truth_filename, "r");
  if (fid == NULL) {
    return;
  }
  int numDetections = detections.size();
  Rect* detectedRectangles = new Rect[numDetections];
  for (int i = 0; i < numDetections; i++) {
    detectedRectangles[i] = detections[i];
  }
  fscanf(fid, "%d", &numDartBoards);
  Rect* groundTruthRect = new Rect[numDartBoards];
  for (int i = 0; i < numDartBoards; i++) {
    fscanf(fid, "%d %d %d %d", &x, &y, &w, &h);
    Rect detectedBox = Rect(x, y, w, h);
    groundTruthRect[i] = detectedBox;
  }
  fclose(fid);
  CalcTPR(numDetections, detectedRectangles, numDartBoards, groundTruthRect);
  free(groundTruthRect);
  free(detectedRectangles);
}

void Detector::CalcTPR(int num_detections, Rect* detected, int num_dart_boards,
                       Rect* ground_truth) {
  int count = 0;
  double threshold = 0.5;

  for (int i = 0; i < num_dart_boards; i++) {
    for (int j = 0; j < num_detections; j++) {
      double intersectionArea = (ground_truth[i] & detected[j]).area();
      double union_area = (ground_truth[i] | detected[j]).area();

      if (union_area > 0) {
        double ratio = intersectionArea / union_area;
        if ((ratio > threshold)) {
          count++;
        }
      }
    }
  }
  int TP = count;
  int FP = num_detections - TP;
  int FN = num_dart_boards - TP;
  double truePositiveRate = (double)TP / (TP + FN);
  double f1_score = (double)2 * TP / (2 * TP + FN + FP);
  printf("TP: %d FP: %d FN: %d F1: %lf \n", TP, FP, FN, f1_score);
}
bool Detector::PointInRect(Point2f pt, Rect rectangle) {
  if (pt.x > rectangle.x && pt.x < rectangle.x + rectangle.width &&
      pt.y > rectangle.y && pt.y < rectangle.y + rectangle.height) {
    return true;
  }
  return false;
}
std::vector<Point2f> Detector::getFFDPoints(Mat img_scene, Mat img_object) {
  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 20;
  FastFeatureDetector detector(minHessian);
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  detector.detect(img_object, keypoints_object);
  detector.detect(img_scene, keypoints_scene);
  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  Mat descriptors_object, descriptors_scene;
  extractor.compute(img_object, keypoints_object, descriptors_object);
  extractor.compute(img_scene, keypoints_scene, descriptors_scene);
  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;
  matcher.match(descriptors_object, descriptors_scene, matches);
  double min_dist = 1;
  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector<DMatch> good_matches;

  for (int i = 0; i < descriptors_object.rows; i++) {
    if (matches[i].distance < 3 * min_dist) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  for (int i = 0; i < good_matches.size(); i++) {
    scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
  }
  return scene;
}
// Gets score from Fast Feature Detector
int Detector::getFFDScore(Rect rectangle, Mat img_scene, Mat img_object) {
  int count = 0;
  std::vector<Point2f> points = getFFDPoints(img_scene, img_object);
  for (int i = 0; i < points.size(); i++) {
    if (PointInRect(points[i], rectangle)) {
      count++;
    }
  }
  return count;
}
void Detector::Detect() {
  std::vector<Rect> detections;
  std::vector<Rect> faces;
  Mat input = frame.clone();
  Mat frame_gray;
  Mat dartImage;
  dartImage = imread("dartc.jpg", CV_LOAD_IMAGE_COLOR);
  // 1. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor(input, frame_gray, CV_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE,
                           Size(50, 50), Size(500, 500));
  for (int i = 0; i < faces.size(); i++) {
    int radius = faces[i].width / 2;
    int maxRadius = radius * 1.1;
    int minRadius = radius * 0.9;
    Mat section, sectionGray, outSection;
    int x = faces[i].x - (maxRadius - radius);
    int y = faces[i].y - (maxRadius - radius);
    if (x < 0) {
      x = 0;
    }
    if (y < 0) {
      y = 0;
    }

    int width = maxRadius * 2;
    int height = maxRadius * 2;
    section = input(Rect(x, y, width, height));
    outSection = frame(Rect(x, y, width, height));
    Hough* currentSection = new Hough(section, 15, outSection, i);
    bool detected = false;
    Rect tempDetec =
        Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
    int ffd_score = getFFDScore(tempDetec, frame, dartImage);
    if (ffd_score < 95 && ffd_score > 67) {
      detected = true;
    }
    if (currentSection->Circles(minRadius, maxRadius, 1, 14)) {
      detected = true;
    }
    if (currentSection->Lines(40, 5, 15, 7)) {
      detected = true;
    }
    if (currentSection->Lines(20, 5, 10, 2)) {
      if (ffd_score > 40 && ffd_score < 52) {
        detected = true;
      }
    }

    if (detected) {
      Rect tempDetec =
          Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      detections.push_back(tempDetec);
      rectangle(
          frame, Point(faces[i].x, faces[i].y),
          Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
          Scalar(0, 255, 0), 2);
    }
    free(currentSection);
  }
  // CalcF1Scores(groundTruthFile,detections);
}
}  // namespace ipcv
int main(int argc, const char** argv) {
  ipcv::Detector DartDetector(argv[1]);
  DartDetector.Detect();
  DartDetector.WriteImage();
  return 0;
}
