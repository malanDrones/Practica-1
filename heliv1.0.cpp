#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
/*
 * A simple interface to the ARDrone, v0.2 
 * check out the CHeli class and main() to see 
 * Modified on 11 September, 2017
 */

#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

bool stop = false;
CRawImage *image;
CHeli *heli;
float pitch, roll, yaw, height;
int hover=0;
// Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
int joypadRoll, joypadPitch, joypadVerticalSpeed, joypadYaw;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
string ultimo = "init";

bool uniform = true;
bool accumulate1 = false;

int Px;
int Py;
int vR;
int vG;
int vB;
int treshold = 125;
int xA=0;
int yA=0;
int val_red[10];
int val_green[10];
int val_blue[10];
int n=0;
int clicks=10;

Mat cameraFrame;
Mat i_og;
Mat i_bn;
Mat i_hsv;
Mat i_yiq;
Mat i_bin;
Mat i_sep;
Mat blue, green, red;

int histSize = 256;
float range[] = { 0, 256 };
const float* histRange = { range };
int hist_w = 512; int hist_h = 300;
int bin_w = cvRound( (double) hist_w/histSize );
Mat histBlue( hist_h+30, hist_w, CV_8UC3, Scalar( 0,0,0) );
Mat histGreen( hist_h+30, hist_w, CV_8UC3, Scalar( 0,0,0) );
Mat histRed( hist_h+30, hist_w, CV_8UC3, Scalar( 0,0,0) );

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
		rectangle(histBlue,Point(x+(b-1)*bin_w,hist_h),Point(x+b*(bin_w),hist_h+30),colorB,-1);
		Scalar colorG=Scalar(0,g,0);
		rectangle(histGreen,Point(x+(b-1)*bin_w,hist_h),Point(x+b*(bin_w),hist_h+30),colorG,-1);
		Scalar colorR=Scalar(0,0,r);
		rectangle(histRed,Point(x+(b-1)*bin_w,hist_h),Point(x+b*(bin_w),hist_h+30),colorR,-1);
	}
}

int max(int list[]){
    int val = list[0];
    for (int m=0;m<clicks;m++){
            if(list[m] > val)
                val = list[m];
        }
    return val;
}

int min(int list[]){
    int val = list[0];
    for (int m=0;m<clicks;m++){
            if(list[m] < val)
                val = list[m];
        }
    return val;
}

// Convert CRawImage to Mat
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

//codigo del click en pantalla
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

int main(int argc,char* argv[])
{
	//establishing connection with the quadcopter
	heli = new CHeli();
	
	//this class holds the image from the drone	
	image = new CRawImage(320,240);
	
	// Initial values for control	
    pitch = roll = yaw = height = 0.0;
    joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

	// Destination OpenCV Mat	
	Mat currentImage = Mat(240, 320, CV_8UC3);
	// Show it	
	imshow("ParrotCam", currentImage);

    // Initialize joystick
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
    useJoystick = SDL_NumJoysticks() > 0;
    if (useJoystick)
    {
        SDL_JoystickClose(m_joystick);
        m_joystick = SDL_JoystickOpen(0);
    }

    namedWindow("ParrotCam");
    //setMouseCallback("ParrotCam", mouseCoordinatesExampleCallback);

    while (stop == false)
    {

        // Clear the console
        printf("\033[2J\033[1;1H");

        if (useJoystick)
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            joypadRoll = SDL_JoystickGetAxis(m_joystick, 2);
            joypadPitch = SDL_JoystickGetAxis(m_joystick, 3);
            joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
            joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
            joypadTakeOff = SDL_JoystickGetButton(m_joystick, 1);
            joypadLand = SDL_JoystickGetButton(m_joystick, 2);
            joypadHover = SDL_JoystickGetButton(m_joystick, 0);
        }

        // prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
        printf("===================== Parrot Basic Example =====================\n\n");
        fprintf(stdout, "Angles  : %.2lf %.2lf %.2lf \n", helidata.phi, helidata.psi, helidata.theta);
        fprintf(stdout, "Speeds  : %.2lf %.2lf %.2lf \n", helidata.vx, helidata.vy, helidata.vz);
        fprintf(stdout, "Battery : %.0lf \n", helidata.battery);
        fprintf(stdout, "Hover   : %d \n", hover);
        fprintf(stdout, "Joypad  : %d \n", useJoystick ? 1 : 0);
        fprintf(stdout, "  Roll    : %d \n", joypadRoll);
        fprintf(stdout, "  Pitch   : %d \n", joypadPitch);
        fprintf(stdout, "  Yaw     : %d \n", joypadYaw);
        fprintf(stdout, "  V.S.    : %d \n", joypadVerticalSpeed);
        fprintf(stdout, "  TakeOff : %d \n", joypadTakeOff);
        fprintf(stdout, "  Land    : %d \n", joypadLand);
        fprintf(stdout, "Navigating with Joystick: %d \n", navigatedWithJoystick ? 1 : 0);
        cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor RGB: ("<<vR<<","<<vG<<","<<vB<<")"<<endl;
        
	   //cout<<"Prueba Luis"<<endl;
	
		//image is captured
		heli->renewImage(image);

		// Copy to OpenCV Mat
		rawToMat(currentImage, image);
		imshow("ParrotCam", currentImage);
        i_og=currentImage;
        //imshow("Click", i_og);

        char key = waitKey(5);
		switch (key) {
			case 'a': yaw = -20000.0; break;
			case 'd': yaw = 20000.0; break;
			case 'w': height = -20000.0; break;
			case 's': height = 20000.0; break;
			case 'q': heli->takeoff(); break;
			case 'e': heli->land(); break;
			case 'z': heli->switchCamera(0); break;
			case 'x': heli->switchCamera(1); break;
			case 'c': heli->switchCamera(2); break;
			case 'v': heli->switchCamera(3); break;
			case 'j': roll = -20000.0; break;
			case 'l': roll = 20000.0; break;
			case 'i': pitch = -20000.0; break;
			case 'k': pitch = 20000.0; break;
            case 'h': hover = (hover + 1) % 2; break;
            case 27: stop = true; break;
            default: pitch = roll = yaw = height = 0.0;
		}

        if (joypadTakeOff) {
            heli->takeoff();
        }
        if (joypadLand) {
            heli->land();
        }
        //hover = joypadHover ? 1 : 0;

        //setting the drone angles
        if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
        {
            heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
            navigatedWithJoystick = true;
        }
        else
        {
            heli->setAngles(pitch, roll, yaw, height, hover);
            navigatedWithJoystick = false;
        }



        if(key == 'p'){
            //stream1 >> i_og;
            vector<Mat> bgr_planes;
            split(i_og, bgr_planes);
            imshow("Original", i_og);
            tratamiento_imagen(i_og);
            imshow("HSV", i_hsv);
            imshow("YIQ", i_yiq);
            
            calcHist( &bgr_planes[0], 1, 0, Mat(), blue, 1, &histSize, &histRange, uniform, accumulate1 );
            calcHist( &bgr_planes[1], 1, 0, Mat(), green, 1, &histSize, &histRange, uniform, accumulate1 );
            calcHist( &bgr_planes[2], 1, 0, Mat(), red, 1, &histSize, &histRange, uniform, accumulate1 );
            
            normalize(blue, blue, 0, histBlue.rows-30, NORM_MINMAX, -1, Mat() );
            normalize(green, green, 0, histGreen.rows-30, NORM_MINMAX, -1, Mat() );
            normalize(red, red, 0, histRed.rows-30, NORM_MINMAX, -1, Mat() );
            
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
            for( int i = 1; i < histSize; i++ ){
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

        if(n == clicks){
            cout<<"R MAX:" << max(val_red) <<" R MIN:" << min(val_red) << endl;
            cout<<"G MAX:" << max(val_green) <<" G MIN:" << min(val_green) << endl;
            cout<<"B MAX:" << max(val_blue) <<" B MIN:" << min(val_blue) << endl;
            n=0;
            separarRGB(i_og, i_sep, max(val_red)+10, min(val_red)-10, max(val_green)+10, min(val_green)-10,max(val_blue)+10, min(val_blue)-10);
        }

        usleep(15000);
        if (key == '1') 
            stop = true;
	}
	
	heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
	delete image;
	return 0;
}
