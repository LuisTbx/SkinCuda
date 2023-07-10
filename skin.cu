// Losely based on https://github.com/abubakr-shafique/Image_Inversion_CUDA_CPP/blob/master/kernel.cu
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "SkinDetector.h"


int main(void)
{

  cv::Mat image = cv::imread("test_images/fr.jpeg", -1);
  cv::Mat outImage = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

  cv::Mat covariance = (cv::Mat_<float>(2,2) << 0.0038,-0.0009,-0.0009, 0.0009 );
  cv::Mat meanCV = (cv::Mat_<float>(2,1) << 0.4404, 0.3111);
  cv::Mat inverseCovariance;
  cv::invert(covariance, inverseCovariance);
  float threshold = 0.33f;

  SkinDetector det = SkinDetector((float*)inverseCovariance.data, (float*)meanCV.data, threshold, image.cols, image.rows);

  det.skinMap(image.data);

  cv::imwrite("map.jpeg", image);
  return 0;
}