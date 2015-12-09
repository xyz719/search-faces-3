//#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
using namespace cv;


void detectAndDisplay(Mat picture);
String pCascadeFrontal = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";

int main(void)
{
	if (!face_cascade.load(pCascadeFrontal))
	{
		printf("--(!)Error loading\n");
		return -1;
	};
	IplImage* img = cvLoadImage("3.jpg", CV_LOAD_IMAGE_COLOR);
	Mat picture(img);
	if (!picture.empty())
	{
		detectAndDisplay(picture);
 
	}
	else
	{
 
		printf("--(!)Error!\n");
 
	}
	waitKey(0);
	cvDestroyWindow(window_name.c_str());
	cvReleaseImage(&img); 
	return 0;
 
}

void detectAndDisplay(Mat picture)
{
	std::vector<Rect> faces;
	Mat picture_gray;
	cvtColor(picture, picture_gray, CV_BGR2GRAY);
	double t = (double)cvGetTickCount();
	face_cascade.detectMultiScale(picture_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	t = (double)cvGetTickCount() - t;
	printf("%g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2); 
		printf("Found a face at (%d, %d)\n", center.x, center.y);
		ellipse(picture, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255),5, 2, 0);
	}
	resize(picture,picture,Size(650,500),0,0,CV_INTER_LINEAR);
	imshow(window_name,picture);
 
}