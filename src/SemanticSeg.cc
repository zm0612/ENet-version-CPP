//
// Created by meng on 2019/11/7.
//
#include "SemanticSeg.h"

    void SemanticSeg::LoadNet(const std::string& _netAddr) {
        net = cv::dnn::readNet(_netAddr);
        assert(!net.empty());
        std::cout << "Load net model successfully" << std::endl;
    }

    cv::Mat SemanticSeg::forward(const cv::Mat& _image) {

        cv::Mat blob;
        cv::Mat output;
        blob = cv::dnn::blobFromImage(_image, 1/255.0, cv::Size(1024, 512), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        output = net.forward();

        const int layers = output.size[1];
        const int rows = output.size[2];
        const int cols = output.size[3];

        maxClass = cv::Mat::zeros(rows, cols, CV_8UC1);
        cv::Mat maxValue(rows, cols, CV_32FC1, output.data);

        for(int layer =1; layer<layers; layer++){
            for (int row = 0; row < rows; ++row) {
                const float *ptrOutput = output.ptr<float>(0, layer, row);
                uint8_t *ptrMaxClass = maxClass.ptr<uint8_t>(row);
                float *ptrMaxValue = maxValue.ptr<float_t>(row);
                for(int col=0; col < cols; col++){
                    if(ptrOutput[col] > ptrMaxValue[col]){
                        ptrMaxValue[col] = ptrOutput[col];
                        ptrMaxClass[col] = (uchar)layer;
                    }
                }
            }
        }

        cv::resize(maxClass, maxClass, cv::Size(_image.cols, _image.rows), 0, 0, cv::INTER_NEAREST);

        return maxClass;
    }

    void SemanticSeg::LoadColorAndLabel(const std::string& colorFile, const std::string& classFile) {

        //load the class label names
        std::ifstream classNamesFile(classFile.c_str());
        assert(classNamesFile.is_open());
        std::string className;
        while(std::getline(classNamesFile, className)) {
            classes.push_back(className);
        }

        // load color file;
        std::ifstream colorNameFile(colorFile.c_str());
        assert(colorNameFile.is_open());
        std::string line;
        while(std::getline(colorNameFile, line)){
            auto pos1 = line.find_first_of(',');
            auto pos2 = line.find_last_of(',');
            int b = std::stoi(line.substr(0, pos1));
            int g = std::stoi(line.substr(pos1+1, pos2-pos1-1));
            int r = std::stoi(line.substr(pos2+1, line.size()-pos2-1));
            colors.push_back(cv::Vec3b(b,g,r));
        }
    }

    cv::Mat SemanticSeg::CreatLegend() {
        cv::Mat legend(colors.size()*25 + 25, 300, CV_8UC3, cv::Scalar(0,0,0));
        for (uint i = 0; i < classes.size(); ++i) {
            cv::putText(legend, classes[i], cv::Point(5, (i*25)+17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            cv::rectangle(legend, cv::Point(100, (i * 25)), cv::Point(300, (i * 25) + 25), colors[i], -1);
        }
        return legend;
    }

    cv::Mat SemanticSeg::MaskImage(const cv::Mat& _image){
        cv::Mat mask(maxClass.rows, maxClass.cols, CV_8UC3);
        for(int row=0; row<maxClass.rows; row++){
            for (int col = 0; col < maxClass.cols; ++col) {
                mask.at<cv::Vec3b>(row, col) = colors[maxClass.at<uchar>(row, col)];
            }
        }

        cv::Mat weightedImage;

        cv::addWeighted(_image, 0.4, mask, 0.6, 0, weightedImage);

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("use time: %.2f ms", t);
        cv::putText(weightedImage, label, cv::Point(0,15), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 255, 0));
        return  weightedImage;
    }

