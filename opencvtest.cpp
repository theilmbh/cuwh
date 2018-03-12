/*
 * =====================================================================================
 *
 *       Filename:  opencvtest.cpp
 *
 *    Description:  testing opencv
 *
 *        Version:  1.0
 *        Created:  03/09/2018 01:34:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    if (argc != 2){
        printf("usage: cuWH <image path>\n");
    }

    cv::Mat image;
    image = cv::imread(argv[1], 1);

    if (!image.data){
        printf("No Image Data \n");
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);

    cv::waitKey(0);
    return 0;
}
