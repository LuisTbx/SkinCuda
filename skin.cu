// Losely based on https://github.com/abubakr-shafique/Image_Inversion_CUDA_CPP/blob/master/kernel.cu
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "SkinDetector.h"


int main(int argc, char* argv[])
{
  std::cout << "Running with: " << argc << " arguments:" << std::endl;
  std::string capture_name = "0";
  int apiID = cv::CAP_ANY;

  if (argc < 2){
    std::cout << "Atempting to initialize camera" << std::endl;
  }
  else{
    capture_name = argv[1];
  }

  std::cout << "Atempting to open video: " << capture_name << std::endl;

  cv::VideoCapture cap;
  cap.open(capture_name, apiID);

  int rows = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int cols = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  cv::Mat image;
  cv::Mat outImage = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

  cv::Mat covariance = (cv::Mat_<float>(2,2) << 0.0038,-0.0009,-0.0009, 0.0009 );
  cv::Mat meanCV = (cv::Mat_<float>(2,1) << 0.4404, 0.3111);
  cv::Mat inverseCovariance;
  cv::invert(covariance, inverseCovariance);
  float threshold = 0.33f;

  SkinDetector det = SkinDetector((float*)inverseCovariance.data, (float*)meanCV.data, threshold, cols, rows);

  if(!cap.isOpened()){
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }

  for(;;){
    cap.read(image);
    if (image.empty()) {
      std::cout << "End of video\n";
      break;
    }
    det.skinMap(image.data);
    cv::imshow("SKINMAP",image);
    char key = cv::waitKey(20);
    if (key == 27){
      break;
    }
  }
  cap.release();
  image.release();

  return 0;
}