#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

using namespace cv;
using namespace std;
using namespace cv::line_descriptor;

void ExtractLineSegment(const Mat &img, const Mat &image2, vector<KeyLine> &keylines,vector<KeyLine> &keylines2)
{
    Mat mLdesc,mLdesc2;
    vector<vector<DMatch>> lmatches;
    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();

    lsd->detect(img, keylines, 1.2,1);
    lsd->detect(image2,keylines2,1.2,1);
    int lsdNFeatures = 50;
    if(keylines.size()>lsdNFeatures){
        sort(keylines.begin(), keylines.end(),[](const KeyLine &a,const KeyLine &b){return a.response > b.response;});
        keylines.resize(lsdNFeatures);
        for( int i=0; i<lsdNFeatures; i++)
            keylines[i].class_id = i;
    }
    if(keylines2.size()>lsdNFeatures){
        sort(keylines2.begin(), keylines2.end(), [](const KeyLine &a,const KeyLine &b){return a.response > b.response;});
        keylines2.resize(lsdNFeatures);
        for(int i=0; i<lsdNFeatures; i++)
            keylines2[i].class_id = i;
    }

    lbd->compute(img, keylines, mLdesc);
    lbd->compute(image2,keylines2,mLdesc2);
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    vector<DMatch> matches;
    for(size_t i=0;i<lmatches.size();i++)
    {
        const DMatch& bestMatch = lmatches[i][0];
        const DMatch& betterMatch = lmatches[i][1];
        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < 0.7)
            matches.push_back(bestMatch);
    }

    cv::Mat outImg;
    std::vector<char> mask( lmatches.size(), 1 );
    drawLineMatches( img, keylines, image2, keylines2, matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask, DrawLinesMatchesFlags::DEFAULT );
    imshow( "Matches", outImg );
    waitKey(0);
    imwrite("Line_Matcher.png",outImg);
}

int main(int argc, char**argv)
{
    if(argc != 3){
        cerr << endl << "Usage: ./Line path_to_image1 path_to_image2" << endl;
        return 1;
    }
    string imagePath1=string(argv[1]);
    string imagePath2=string(argv[2]);
    cout<<"import two images"<<endl;
    Mat image1=imread(imagePath1);
    Mat image2=imread(imagePath2);

    imshow("img1",image1);
    imshow("img2",image2);
    waitKey(0);
    destroyWindow("img1");
    destroyWindow("img2");

    vector<KeyLine> keylines,keylines2;
    ExtractLineSegment(image1,image2,keylines,keylines2);
    return 0;
}

