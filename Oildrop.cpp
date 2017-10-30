#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "SDL/SDL.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <utility>
#include <time.h>
//#include <cstdlib>
using namespace cv;
using namespace std;

Mat Oimg;
Mat Aimg;
Mat Gimg;
Mat Mimg;
Mat Limg;
Mat Dimg;
Mat Himg;
Mat Cimg;
Mat Eimg;
Mat Dilimg;
Mat Bimg;
Mat dst;
Mat Mix;
Mat cameraFrame;
Mat grad;
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
int Px;
int Py;
int vR;
int vG;
int vB;
int binTreshold = 125;
int lowThreshold;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;
int srcColor = 1;  //0=Gray 1=color
bool stop = false;
int Wsize[2] = {500, 500};  

Mat Color;
int color = 255;
int Loop = 0;
int const MaxLoop = 200;
//unsigned long int Area = 0;
unsigned long int Area[MaxLoop];
int Seed[2] = {0,0};
int Pc [2] = {0,0}; 
int Pn [2] = {0,0}; 
int Po [2] = {0,0}; 
int Ps [2] = {0,0};
int Pe [2] = {0,0};
vector<pair<int, int> > frontier;  
int n=0;
int expo = 0;

//const char *imageName = "/home/luis/Desktop/Visión para robots/Parrot/src/main/Louvre2.JPG";
const char *srcImg = "circles.jpg";


void MouseEvent(int event, int x, int y, int flags, void* param)
{
	Mat* rgb = (Mat*) param;
    switch (event)
    {
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONDOWN: 
            Px=x;
            Py=y;
            
            vB=(int)(*rgb).at<Vec3b>(Point(x,y))[0];
            vG=(int)(*rgb).at<Vec3b>(Point(x,y))[1];
            vR=(int)(*rgb).at<Vec3b>(Point(x,y))[2];
			
            cout << "  X:" << x << " Y:" << y ;
        	if(srcColor == 0){
        		cout << "  Val:" << (vR+vG+vB)/3 ;
        		cout << endl;
        	} else {
               	cout << "  --  R:" << vR << " G:" << vG << " B:" <<vB ;
        		cout << endl;  
        	}
        case CV_EVENT_RBUTTONDOWN:
        	//flag=!flag;
            break;
        case CV_EVENT_LBUTTONDBLCLK:
        	break;
        
    }
}

void CannyThreshold(int, void*){
    GaussianBlur(Oimg, Dimg, Size(3,3), 0, 0);  /// Reduce noise with a kernel 3x3
    Canny( Dimg, Cimg, lowThreshold, lowThreshold*3, 3);  /// Canny detector
    dst = Scalar::all(0);  /// Using Canny's output as a mask, we display our result
    Oimg.copyTo(dst,Cimg);
    addWeighted(Cimg, 1, Oimg, 0.7, 0, Mix, -1);
	imshow("Enhancement", Mix);
 }

 /**  @function Erosion  */
void Erosion( int, void* ){
    int erosion_type;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  	else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  	else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  	Mat element = getStructuringElement(erosion_type,Size(2*erosion_size+1,2*erosion_size+1),Point(erosion_size,erosion_size));
  	erode( Oimg, Eimg, element ); /// Apply the erosion operation
  	imshow( "Erosión", Eimg );
}

/** @function Dilation */
void Dilation( int, void* ){
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement(dilation_type,Size(2*dilation_size+1,2*dilation_size+1),Point(dilation_size,dilation_size));
  dilate( Oimg, Dilimg, element ); /// Apply the dilation operation
  imshow( "Dilatación", Dilimg );
}

/*
// Convert CRawImage to Mat  //need CHeli.h
void rawToMat( Mat &destImage, CRawImage* sourceImage)
{	
	uchar *pointerImage = destImage.ptr(0);
	
	for (int i = 0; i < 240*320; i++)
	{
		pointerImage[3*i] = sourceImage->data[3*i+2];
		pointerImage[3*i+1] = sourceImage->data[3*i+1];
		pointerImage[3*i+2] = sourceImage->data[3*i];
	}
}
*/

void binarizar(const Mat &sourceImage, Mat &destinationImage)
{
    //namedWindow("BIN",1);
    namedWindow("Binarizada", WINDOW_NORMAL);
	resizeWindow("Binarizada", Wsize[0], Wsize[1]);
	setMouseCallback("Binarizada", MouseEvent, &destinationImage);
    createTrackbar("Treshold", "Binarizada", &binTreshold, 255);
    if (destinationImage.empty())
        destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
        
    for(int y=0; y<sourceImage.rows;++y)
        for(int x=0; x<sourceImage.cols; ++x){
            if(sourceImage.at<Vec3b>(Point(x,y))[0] < binTreshold){
                for (int i = 0; i < sourceImage.channels(); ++i)
                    destinationImage.at<Vec3b>(Point(x,y))[i] = 0;
            }else{
                for (int i = 0; i < sourceImage.channels(); ++i)
                    destinationImage.at<Vec3b>(Point(x,y))[i] = 255;
            }
        }
    imshow("Binarizada", destinationImage);
}

void Expand(pair <int, int> Pc){
	pair <int, int> Pn (Pc.first,Pc.second-1);
	pair <int, int> Po (Pc.first-1,Pc.second);
	pair <int, int> Ps (Pc.first,Pc.second+1);
	pair <int, int> Pe (Pc.first+1,Pc.second);

	if ((Bimg.at<Vec3b>(Pn.second, Pn.first)[0] == 0) && (Color.at<Vec3b>(Pn.second, Pn.first)[0] == 0)){
		Color.at<Vec3b>(Pn.second, Pn.first)[0] = color;
		frontier.push_back(Pn);
		Area[n]++;
	}
	if ((Bimg.at<Vec3b>(Po.second, Po.first)[0] == 0) && (Color.at<Vec3b>(Po.second, Po.first)[0] == 0)){
		Color.at<Vec3b>(Po.second, Po.first)[0] = color;
		frontier.push_back(Po);
		Area[n]++;
	}
	if ((Bimg.at<Vec3b>(Ps.second, Ps.first)[0] == 0) && (Color.at<Vec3b>(Ps.second, Ps.first)[0] == 0)){
		Color.at<Vec3b>(Ps.second, Ps.first)[0] = color;
		frontier.push_back(Ps);
		Area[n]++;
	}
	if ((Bimg.at<Vec3b>(Pe.second, Pe.first)[0] == 0) && (Color.at<Vec3b>(Pe.second, Pe.first)[0] == 0)){
		Color.at<Vec3b>(Pe.second, Pe.first)[0] = color;
		frontier.push_back(Pe);
		Area[n]++;
	}
}


int main() {
	//bool uniform = true; bool accumulate = false;
	/*VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}*/
	srand (time(NULL));
	namedWindow("Original", WINDOW_NORMAL);
	resizeWindow("Original", Wsize[0], Wsize[1]);
	setMouseCallback("Original", MouseEvent, &Oimg);
	//rawToMat(Oimg,srcImg); //need CHeli.h
	Oimg = imread(srcImg, srcColor);  
	dst.create( Oimg.size(), Oimg.type() );
	//unconditional loop

	if(!Oimg.data){
   		printf( " No image data \n " );
   		stop = true;
 	}
	imshow("Original", Oimg);
	binarizar(Oimg,Bimg);
	Color = Bimg.clone();

	while (stop == false) {
		char key = waitKey(5);
		//stream1.read(cameraFrame);
		
		while(Loop <= MaxLoop){
			Seed [0] = rand()%Bimg.cols;
			Seed [1] = rand()%Bimg.rows;			
			cout << "  X:" << Seed[0] << "   Y:" << Seed[1] ;
        	cout << endl; 
        	int object = Bimg.at<Vec3b>(Seed[1], Seed[0])[0];
        	int colors = Bimg.at<Vec3b>(Seed[1], Seed[0])[0];
			cout << "  F(x,y):" << object << "   Color(x,y):" << colors ;
        	cout << endl; 
			
			if ((Bimg.at<Vec3b>(Seed[1], Seed[0])[0] == 0) && (Color.at<Vec3b>(Seed[1], Seed[0])[0] == 0)) {
				pair <int, int> Pc(Seed[0], Seed[1]);
				Color.at<Vec3b>(Pc.second,Pc.first)[1] = color;
				frontier.push_back(Pc);
				Area[n]++;
				//cout << " Area:" << Area[n] ;
				//cout << endl;
				while (!frontier.empty()){
					Expand(Pc);
					frontier.erase(frontier.begin());
					Pc = frontier.front();
					cout << "  Expand:" << expo ;
        			cout << endl;
        			expo++; 
				}
				n++;
			}
			Loop++;
			cout << "  N: " << n << "   Loop:" << Loop  << "   Total area: " << Area[0]+Area[1]+Area[2] ;
        	cout << endl; 
		}

		namedWindow("Color", WINDOW_NORMAL);
		resizeWindow("Color", Wsize[0], Wsize[1]);
		setMouseCallback("Color", MouseEvent, &Color);
		imshow("Color", Color);
		
		switch (key)
		{
			case 'z':
				stop = true;
				break;

			case '1':  //Promedio
				//boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), bool normalize=true, int borderType=BORDER_DEFAULT )
				boxFilter(Oimg, Aimg, -1, Size(10,10), Point(-1,-1), true, BORDER_REPLICATE);
				namedWindow("AverageBlur", WINDOW_NORMAL);
				resizeWindow("AverageBlur", Wsize[0], Wsize[1]);
				imshow("AverageBlur", Aimg);
				break;
			case '2':  //Gausiano
				//GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
				GaussianBlur(Oimg, Gimg, Size(11,11), 0, 0);
				namedWindow("GaussianBlur", WINDOW_NORMAL);
				resizeWindow("GaussianBlur", Wsize[0], Wsize[1]);
				imshow("GaussianBlur", Gimg);
				break;
			case '3':  //Mediano
				//medianBlur(InputArray src, OutputArray dst, int ksize)
				medianBlur(Oimg, Mimg, 11);
				namedWindow("MedianBlur", WINDOW_NORMAL);
				resizeWindow("MedianBlur", Wsize[0], Wsize[1]);
				imshow("MedianBlur", Mimg);
				break;
			case '4':  //Laplaciano
				//Laplacian(InputArray src, OutputArray dst, int ddepth, int ksize=1, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
				Laplacian(Oimg, Limg, -1, 3, 1, 0, BORDER_DEFAULT);
				namedWindow("Laplacian", WINDOW_NORMAL);
				resizeWindow("Laplacian", Wsize[0], Wsize[1]);
				imshow("Laplacian", Limg);
				break;
			case '5':  //Sombrero
				GaussianBlur(Oimg, Dimg, Size(3,3), 0, 0);
				Laplacian(Dimg, Himg, -1, 3, 1, 0, BORDER_DEFAULT);
				namedWindow("Hat", WINDOW_NORMAL);
				resizeWindow("Hat", Wsize[0], Wsize[1]);
				imshow("Hat", Himg);
				break;
			case '6':  //Detector de bordes (2 direcciones)
				// Gradient X
				GaussianBlur(Oimg, Dimg, Size(3,3), 0, 0);
				//Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
				Sobel(Dimg, grad_x, -1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs( grad_x, abs_grad_x );
				// Gradient Y
				Sobel(Dimg, grad_y, -1, 0, 1, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs( grad_y, abs_grad_y );
				addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
				namedWindow("EdgeDetector", WINDOW_NORMAL);
				resizeWindow("EdgeDetector", Wsize[0], Wsize[1]);
				imshow("EdgeDetector", grad);
				break;
			case '7':  //Enfatizador
				namedWindow("Enhancement", WINDOW_NORMAL);
    			resizeWindow("Enhancement", Wsize[0], 550);
				createTrackbar( "Min Threshold:", "Enhancement", &lowThreshold, 100, CannyThreshold);
				CannyThreshold(0, 0);
				break;
			case '8':  //Derivador (Detector de bordes X)
				GaussianBlur(Oimg, Dimg, Size(3,3), 0, 0);
				Sobel(Dimg, grad_x, -1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs( grad_x, abs_grad_x );
				namedWindow("DerivatorX", WINDOW_NORMAL);
				resizeWindow("DerivatorX", Wsize[0], Wsize[1]);
				imshow("DerivatorX", abs_grad_x);
				//Derivador (Detector de bordes y)
				Sobel(Dimg, grad_y, -1, 0, 1, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs( grad_y, abs_grad_y );
				namedWindow("DerivatorY", WINDOW_NORMAL);
				resizeWindow("DerivatorY", Wsize[0], Wsize[1]);
				imshow("DerivatorY", abs_grad_y);
				break;
			case '9':  //Dilatación 
				namedWindow("Dilatación", WINDOW_NORMAL);
				resizeWindow("Dilatación", Wsize[0], 650);
				createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse","Dilatación",&dilation_elem,max_elem,Dilation);
  				createTrackbar("Kernel size:\n 2n +1","Dilatación",&dilation_size,max_kernel_size,Dilation);
  				Dilation( 0, 0 );
				break;
			case '0':  //Erosión
				namedWindow("Erosión", WINDOW_NORMAL);
				resizeWindow("Erosión", Wsize[0], 650);
				createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse","Erosión",&erosion_elem,max_elem,Erosion);
				createTrackbar( "Kernel size:\n 2n +1", "Erosión",&erosion_size, max_kernel_size,Erosion);
				Erosion( 0, 0 );
				break;

			/*default:
				destroyWindow("AverageBlur");
				destroyWindow("GaussianBlur");
				destroyWindow("MedianBlur");
				destroyWindow("Laplacian");
				destroyWindow("Hat");
				break;*/
		}
	}
	destroyAllWindows();
	return 0;
}
