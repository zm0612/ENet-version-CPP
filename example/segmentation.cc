//
// Created by meng on 2019/11/8.
//
#include "SemanticSeg.h"

int main()
{
    std::string addr = "/home/meng/codes/segmentation/enet_file/";

    cv::Mat image;
    image = cv::imread("../images/img_2374.jpg");
    SemanticSeg seg;

    seg.LoadNet(addr + "enet-model.net");

    seg.LoadColorAndLabel(addr+"enet-colors.txt", addr+"enet-classes.txt");

    cv::Mat legend, output, mask;
    legend = seg.CreatLegend();

    output = seg.forward(image);

    mask = seg.MaskImage(image);

    cv::imshow("legend", legend);
    cv::imshow("output", mask);
    cv::imshow("original image", image);
    cv::waitKey(0);
    return 0;
}

