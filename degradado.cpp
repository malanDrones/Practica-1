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
Mat i_og;
Mat i_bn;
Mat i_hsv;
Mat i_yiq;
Mat i_bin;
Mat i_sep;
Mat blue, green, red;
int treshold = 125;
int Px;
int Py;
int vR;
int vG;
int vB;
int xA=0;
int yA=0;
int val_red[10];
int val_green[10];
int val_blue[10];
int n=0;

int histSize = 256;
float range[] = { 0, 256 };
const float* histRange = { range };
int hist_w = 512; int hist_h = 400;
int bin_w = cvRound( (double) hist_w/histSize );
Mat histBlue( hist_h+40, hist_w, CV_8UC3, Scalar( 0,0,0) );
Mat histGreen( hist_h+40, hist_w, CV_8UC3, Scalar( 0,0,0) );
Mat histRed( hist_h+40, hist_w, CV_8UC3, Scalar( 0,0,0) );

void degradados(Mat &histBlue, Mat &histGreen, Mat &histRed){
	int x=0;//start at y=50, then increment
	int b=0;
	int g=0;
	int r=0;
	while(b<255)//run till green color reaches 0
	{
		b++;
		r++;
		g++;
		Scalar colorB=Scalar(b,0,0);
		rectangle(histBlue,Point(x+(b-1)*bin_w,400),Point(x+b*(bin_w),440),colorB,-1);
		Scalar colorG=Scalar(0,g,0);
		rectangle(histGreen,Point(x+(b-1)*bin_w,400),Point(x+b*(bin_w),440),colorG,-1);
		Scalar colorR=Scalar(0,0,r);
		rectangle(histRed,Point(x+(b-1)*bin_w,400),Point(x+b*(bin_w),440),colorR,-1);
	}
}

int max(int list[]){
	int val = list[0];
	for (int m=0;m<10;m++){
			if(list[m] > val)
				val = list[m];
		}
	return val;
}

int min(int list[]){
	int val = list[0];
	for (int m=0;m<10;m++){
			if(list[m] < val)
				val = list[m];
		}
	return val;
}

void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    uchar* destination;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            Px=x;
            Py=y;
            destination = (uchar*) i_og.ptr<uchar>(Py);
            vB=destination[Px * 3];
            vG=destination[Px*3+1];
            vR=destination[Px*3+2];
            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
        case CV_EVENT_RBUTTONDOWN:
        //flag=!flag;
            break;
        
    }
}

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
	imshow("BN", i_bn);
}

void binarizar(const Mat &sourceImage, Mat &destinationImage)
{
	namedWindow("BIN",1);
	createTrackbar("Treshold", "BIN", &treshold, 255);
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
		
	for(int x=0; x<sourceImage.rows;++x)
		for(int y=0; y<sourceImage.cols; ++y){
			if(sourceImage.at<Vec3b>(x, y)[0] < treshold){
				for (int i = 0; i < sourceImage.channels(); ++i)
					destinationImage.at<Vec3b>(x, y)[i] = 0;
			}else{
				for (int i = 0; i < sourceImage.channels(); ++i)
					destinationImage.at<Vec3b>(x, y)[i] = 255;
			}
		}
	imshow("BIN", i_bin);
}

void separarRGB(const Mat &sourceImage, Mat &destinationImage, int rM, int rm, int gM, int gm, int bM, int bm)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
		
	for(int x=0; x<sourceImage.rows;++x)
		for(int y=0; y<sourceImage.cols; ++y){
			if(sourceImage.at<Vec3b>(x, y)[0] < bM && sourceImage.at<Vec3b>(x, y)[0] > bm &&
			sourceImage.at<Vec3b>(x, y)[1] < gM && sourceImage.at<Vec3b>(x, y)[1] > gm &&
			sourceImage.at<Vec3b>(x, y)[2] < rM && sourceImage.at<Vec3b>(x, y)[2] > rm){
				for (int i = 0; i < sourceImage.channels(); ++i)
					destinationImage.at<Vec3b>(x, y)[i] = sourceImage.at<Vec3b>(x, y)[i];
			}else{
				for (int i = 0; i < sourceImage.channels(); ++i)
					destinationImage.at<Vec3b>(x, y)[i] = 0;
			}
		}
	imshow("SEPARADO", i_sep);
}

//Esta funcion toma una matriz y llama a las demás para modificarla
void tratamiento_imagen(const Mat &sourceImage)
{
	blanco_negro(sourceImage,i_bn);
	binarizar(i_bn, i_bin);
	cvtColor(sourceImage, i_hsv, COLOR_BGR2HSV);
	cvtColor(sourceImage, i_yiq, COLOR_BGR2YCrCb);
}
int main() {
	bool uniform = true; bool accumulate = false;
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}
	//unconditional loop
	while (true) {
		stream1.read(cameraFrame);
		imshow("Camara", cameraFrame);
		char key = waitKey(5);
		if(key == 'a'){
			stream1 >> i_og;
			vector<Mat> bgr_planes;
			split(i_og, bgr_planes);
			imshow("Original", i_og);
			tratamiento_imagen(i_og);
			imshow("HSV", i_hsv);
			imshow("YIQ", i_yiq);
			
			calcHist( &bgr_planes[0], 1, 0, Mat(), blue, 1, &histSize, &histRange, uniform, accumulate );
			calcHist( &bgr_planes[1], 1, 0, Mat(), green, 1, &histSize, &histRange, uniform, accumulate );
			calcHist( &bgr_planes[2], 1, 0, Mat(), red, 1, &histSize, &histRange, uniform, accumulate );
			
			normalize(blue, blue, 0, histBlue.rows-40, NORM_MINMAX, -1, Mat() );
			normalize(green, green, 0, histGreen.rows-40, NORM_MINMAX, -1, Mat() );
			normalize(red, red, 0, histRed.rows-40, NORM_MINMAX, -1, Mat() );
			
			histBlue.setTo(Scalar(0,0,0));
			histRed.setTo(Scalar(0,0,0));
			histGreen.setTo(Scalar(0,0,0));
			
			for( int i = 1; i < histSize; i++ )
			{
				line( histBlue, Point( bin_w*(i-1), hist_h - cvRound(blue.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(blue.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
				line( histGreen, Point( bin_w*(i-1), hist_h - cvRound(green.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(green.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
				line( histRed, Point( bin_w*(i-1), hist_h - cvRound(red.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(red.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
			}
			degradados(histBlue, histGreen, histRed);
			namedWindow("Blue Histogram", CV_WINDOW_AUTOSIZE );
			imshow("Blue Histogram", histBlue );
			namedWindow("Green Histogram", CV_WINDOW_AUTOSIZE );
			imshow("Green Histogram", histGreen );
			namedWindow("Red Histogram", CV_WINDOW_AUTOSIZE );
			imshow("Red Histogram", histRed );
		}
			setMouseCallback("Original", mouseCoordinatesExampleCallback);
			if(xA != Px || yA != Py){
				xA = Px;
				yA = Py;
				val_red[n] = vR;
				val_green[n] = vG;
				val_blue[n] = vB;
				n++;
				histBlue.setTo(Scalar(0,0,0));
				histRed.setTo(Scalar(0,0,0));
				histGreen.setTo(Scalar(0,0,0));
				for( int i = 1; i < histSize; i++ )
				{
					line( histBlue, Point( bin_w*(i-1), hist_h - cvRound(blue.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(blue.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
					line( histGreen, Point( bin_w*(i-1), hist_h - cvRound(green.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(green.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
					line( histRed, Point( bin_w*(i-1), hist_h - cvRound(red.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(red.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
				}
				degradados(histBlue, histGreen, histRed);
				circle(histBlue, Point( bin_w*(vB), hist_h - cvRound(blue.at<float>(vB)) ),7,Scalar(255,0,0), CV_FILLED);
				circle(histGreen, Point( bin_w*(vG), hist_h - cvRound(green.at<float>(vG)) ),7,Scalar(0,255,0), CV_FILLED);
				circle(histRed, Point( bin_w*(vR), hist_h - cvRound(red.at<float>(vR)) ),7,Scalar(0,0,255), CV_FILLED);
				imshow("Blue Histogram", histBlue );
				imshow("Green Histogram", histGreen );
				imshow("Red Histogram", histRed );
			}
			if(n == 10){
				cout<<"R MAX:" << max(val_red) <<" R MIN:" << min(val_red) << endl;
				cout<<"G MAX:" << max(val_green) <<" G MIN:" << min(val_green) << endl;
				cout<<"B MAX:" << max(val_blue) <<" B MIN:" << min(val_blue) << endl;
				n=0;
				separarRGB(i_og, i_sep, max(val_red)+10, min(val_red)-10, max(val_green)+10, min(val_green)-10,max(val_blue)+10, min(val_blue)-10);
			}
	}
	return 0;
}
