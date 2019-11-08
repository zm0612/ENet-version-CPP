//
// Created by meng on 2019/11/7.
//

#ifndef ORB_SLAM2_SEMANTIC_H
#define ORB_SLAM2_SEMANTIC_H

#include <iostream>
#include <ctime>
#include <string>
#include <opencv2/opencv.hpp>

class SemanticSeg{
public:
    void LoadNet(const std::string& _netAddr);

    cv::Mat forward(const cv::Mat& _image);

    //The following three functions are used to display the segmentation results.
    void LoadColorAndLabel(const std::string& colorFile, const std::string& classFile);

    cv::Mat CreatLegend();

    cv::Mat MaskImage(const cv::Mat& _image);


private:
    cv::dnn::Net net;//the e-net's model
    std::vector<std::string> classes;//classes file
    std::vector<cv::Vec3b> colors;//colors file
    cv::Mat maxClass;//Final classification result for each pixel
};
#endif //ORB_SLAM2_SEMANTIC_H
