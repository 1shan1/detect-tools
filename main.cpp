#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include "CrowdEstimate.h"
//#include "frcnn_utils.hpp"
//#include"frcnn_param.hpp"

using namespace caffe;
using namespace  std;

int main()
{
    CrowdEstimate::ins();

    string imglist = "/home/pengshanzhen/try/try/jianshezhijia-frame.txt";
    string out_folder_path="/data_2/crowd-predict/out_img_2";



    //boost::filesystem::path tmppath(imglist);
    //string outfile = out_folder_path +"/"+tmppath.stem().string()+ ".jpg";
    //std::ofstream outfile(savepath,ios::app);


    std::ifstream list(imglist);

    string mean_file = "";
    string img_path;
    //cv::Mat densitymap;
    //float headcount;
    while(list >> img_path)
    {
        cv::Mat densitymap;
        float headcount = 0;
        boost::filesystem::path tmppath(img_path);
        string outfile = out_folder_path +"/"+tmppath.stem().string()+ ".jpg";
        cv::Mat img=cv::imread(img_path);
        if (img.data == NULL)
        {
            cout << endl << "read image error" << endl;
            getchar();
        }
        int64 t1 = cvGetTickCount();

        CrowdEstimate::ins().process(img,densitymap,headcount);
        //process(img,densitymap,headcount);
        int64 t2 = cvGetTickCount();
        cv::putText(img,std::to_string(headcount), cv::Point(img.cols-400, 100), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 23, 0), 5, 5);
        cout << "time cost:" << (t2-t1)/cvGetTickFrequency() / 1000000 << "s" << endl;
        cout << "per img people count:" << headcount <<endl;
        cv::imwrite(outfile,img);

        //cvNamedWindow("new",CV_WINDOW_NORMAL);
        //cv::imshow("new", img);
        //cv::waitKey(10000);




    }
    return 0;


}










