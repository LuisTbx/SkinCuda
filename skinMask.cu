#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "SkinDetector.h"


int main(int argc, char* argv[])
{
  std::cout << "Running with: " << argc << " arguments:" << std::endl;
  if (argc < 2){
    std::cout << "Please provide an image to process" << std::endl;
  }
  else{
    for (int i = 0; i < argc; ++i) {
      std::cout << argv[i] << std::endl;
    }
    std::string image_name = argv[1];
    cv::Mat image = cv::imread(image_name, -1);
    // Single channel output mask
    cv::Mat outImage = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);

    cv::Mat covariance = (cv::Mat_<float>(2,2) << 0.0038,-0.0009,-0.0009, 0.0009 );
    cv::Mat meanCV = (cv::Mat_<float>(2,1) << 0.4404, 0.3111);
    cv::Mat inverseCovariance;
    cv::invert(covariance, inverseCovariance);
    float threshold = 0.33f;

    SkinDetector det = SkinDetector((float*)inverseCovariance.data, (float*)meanCV.data, threshold, image.cols, image.rows);

    det.skinMask(image.data, outImage.data);

    cv::imwrite("skinMask.jpg", outImage);
  }
  return 0;
}