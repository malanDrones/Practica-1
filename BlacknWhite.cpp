/*
* File:   main.cpp
* Author: sagar
*
* Created on 10 September, 2012, 7:48 PM
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat cameraFrame;
Mat imagen_final;

//Funcion que saca blanco y negro de imagen
void blanco_negro(const Mat &sourceImage, Mat &destinationImage)
{

int prom = 0;

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols ; ++x){
            //se suman los valores r g y b de cada pixel y se promedian
            prom = 0;
			for (int i = 0; i < sourceImage.channels(); ++i){
                prom += sourceImage.at<Vec3b>(y, x)[i];
			}prom = prom/3;
			for (int i = 0; i < sourceImage.channels(); ++i){
                destinationImage.at<Vec3b>(y, x)[i] = prom;
			}
        }

}

//Esta funcion toma una matriz y llama a las demás para modificarla
void tratamiento_imagen(const Mat &sourceImage)
{
blanco_negro(sourceImage,imagen_final);
}

int main() {

VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

if (!stream1.isOpened()) { //check if video device has been initialised
cout << "cannot open camera";
}

//unconditional loop
while (true) {
stream1.read(cameraFrame);

tratamiento_imagen(cameraFrame);

imshow("cam", imagen_final);
if (waitKey(30) >= 0)
break;
}
return 0;
}



