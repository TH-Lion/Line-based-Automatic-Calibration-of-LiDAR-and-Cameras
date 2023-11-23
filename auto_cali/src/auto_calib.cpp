#include "auto_calib.h"

bool AutoCalib::getPointcloud(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr ptcloud)
{
    // Load the actual pointcloud.
    const size_t kMaxNumberOfPoints = 1e6; // From Readme for raw files.
    ptcloud->clear();
    ptcloud->reserve(kMaxNumberOfPoints);
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    if (!input)
    {
        std::cout << "Could not open pointcloud file.\n";
        return false;
    }

    // From yanii's kitti-pcl toolkit:
    // https://github.com/yanii/kitti-pcl/blob/master/src/kitti2pcd.cpp
    for (size_t i = 0; input.good() && !input.eof(); i++)
    {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        if (pcl_isfinite(point.x) && pcl_isfinite(point.y) && pcl_isfinite(point.z) && pcl_isfinite(point.intensity))
        {
           ptcloud->push_back(point); 
        }
    }
    input.close();
    return true;
}
void AutoCalib::getOxtsData(std::string filename, std::vector<float> &oxts_vec)
{
    std::ifstream oxtsfile(filename);
    std::string line_data;
    if (oxtsfile)
    {
        while (getline(oxtsfile, line_data))
        {
            // std::cout<<"line_data = ";
            // for(int i = 0; i < line_data.size(); i++)
            // {
            //     std::cout<<line_data[i]<<" ";
            // }
            // std::cout<<std::endl;
            //string转char
            char *lineCharArray;
            const int len = line_data.length();
            lineCharArray = new char[len + 1];
            strcpy(lineCharArray, line_data.c_str());

            char *p;                        //分隔后的字符串
            p = strtok(lineCharArray, " "); //按照spaceChar分割
            //将数据加入vector中
            while (p)
            {
                // std::cout<<atof(p)<<" ";
                oxts_vec.push_back(atof(p));
                p = strtok(NULL, " ");
            }
        }
    }
    else
    {
        std::cout << "Can not open oxts file" << std::endl;
    }
}

bool AutoCalib::point_cmp(pcl::PointXYZI a, pcl::PointXYZI b)
{
    return a.z < b.z;
}
void AutoCalib::point_cb(pcl::PointCloud<pcl::PointXYZI>::Ptr data, pcl::PointCloud<pcl::PointXYZI>::Ptr final_no_ground)
{
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // For mark ground points and hold all points
    pcl::PointCloud<pcl::PointXYZI>::Ptr data_org(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*data, *data_org);

    float angle;
    PointXYZIRL point;
    pcl::PointCloud<PointXYZIRL>::Ptr g_all_pc(new pcl::PointCloud<PointXYZIRL>);

    for (size_t i = 0; i < data->points.size(); i++)
    {
        point.x = data->points[i].x;
        point.y = data->points[i].y;
        point.z = data->points[i].z;
        point.intensity = data->points[i].intensity;

        point.label = 0u; // 0 means uncluster
        g_all_pc->points.push_back(point);
    }
    //std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
    // 2.Sort on Z-axis value.
    sort(data_org->points.begin(), data_org->points.end(), AutoCalib::point_cmp);
    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    pcl::PointCloud<pcl::PointXYZI>::iterator it = data_org->points.begin();
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < -1.5 * 2.0)
        {
            it++;
        }
        else
        {
            break;
        }
    }
    data_org->erase(data_org->points.begin(), it);
    // 4. Extract init ground seeds.
    double sum = 0;
    int cnt = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_seeds_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // Calculate the mean height value.
    for (int i = 0; i < data_org->points.size() && cnt < 20; i++)
    {
        sum += data_org->points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < data_org->points.size(); i++)
    {
        if (data_org->points[i].z < lpr_height + 0.4)
        {
            g_seeds_pc->points.push_back(data_org->points[i]);
        }
    }

    g_ground_pc = g_seeds_pc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr g_not_ground_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // 5. Ground plane fitter mainloop
    float d_, th_dist_d_;
    Eigen::MatrixXf normal_;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Matrix3f cov;
        Eigen::Vector4f pc_mean;
        pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean);
        // Singular Value Decomposition: SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
        // use the least singular vector as normal
        normal_ = (svd.matrixU().col(2));
        // mean ground seeds value
        Eigen::Vector3f seeds_mean = pc_mean.head<3>();

        // according to normal.T*[x,y,z] = -d
        d_ = -(normal_.transpose() * seeds_mean)(0, 0);
        // set distance threhold to `th_dist - d`
        th_dist_d_ = 0.3 - d_;

        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(data->points.size(), 3);
        int j = 0;
        for (auto p : data->points)
        {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0; r < result.rows(); r++)
        {
            if (result[r] < th_dist_d_)
            {
                g_all_pc->points[r].label = 1u; // means ground
                g_ground_pc->points.push_back(data->points[r]);
            }
            else
            {
                g_all_pc->points[r].label = 0u; // means not ground and non clusterred
                g_not_ground_pc->points.push_back(data->points[r]);
            }
        }
    }

    pcl::copyPointCloud(*g_not_ground_pc, *final_no_ground);

    // ROS_INFO_STREAM("origin: "<<g_not_ground_pc->points.size()<<" post_process: "<<final_no_ground->points.size());

    // publish ground points
    //    sensor_msgs::PointCloud2 ground_msg;
    //    pcl::toROSMsg(*g_ground_pc, ground_msg);
    //    ground_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    ground_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_ground_.publish(ground_msg);
    //
    //    // publish not ground points
    //    sensor_msgs::PointCloud2 groundless_msg;
    //    pcl::toROSMsg(*final_no_ground, groundless_msg);
    //    groundless_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    groundless_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_no_ground_.publish(groundless_msg);
    //
    //    // publish all points
    //    sensor_msgs::PointCloud2 all_points_msg;
    //    pcl::toROSMsg(*g_all_pc, all_points_msg);
    //    all_points_msg.header.stamp = in_cloud_ptr->header.stamp;
    //    all_points_msg.header.frame_id = in_cloud_ptr->header.frame_id;
    //    pub_all_points_.publish(all_points_msg);
    //std::cout << g_ground_pc->size() << std::endl;
}

// txtname: /home/zh/code/useful_tools/auto_calibration/data/list.txt
// foldername: /home/zh/code/useful_tools/auto_calibration/data/
// oxtsfolder: /home/zh/data/kitti/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/oxts/data/
int AutoCalib::getData(std::string txtName, std::string folderName, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pointclouds,
                       std::vector<cv::Mat> &images, std::vector<float> &oxts_vec)
{
    std::cout << "Start Read Data ..." << std::endl;
    std::string filename;
    std::ifstream readtxt;
    readtxt.open(txtName);
    if (!readtxt)
    {
        std::cout << "\033[31mgetData Error: Open txt file faile!\033[0m" << std::endl;
        std::exit(0);
    }
    int n = 0;
    while (readtxt >> filename)
    {
        filename = folderName + filename;

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        if (!getPointcloud(filename, raw))
        {
            std::cout << "\033[31mgetData Error: Could not open " << filename << " !\033[0m" << std::endl;
            std::exit(0);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        if (config["delete_ground"].as<bool>())
        {
            point_cb(raw, pc);
        }
        else
        {
            pcl::copyPointCloud(*raw, *pc);
        }
        pointclouds.push_back(pc);

        if (!(readtxt >> filename))
        {
            std::cout << "\033[31mgetData Error: no image!\033[0m" << std::endl;
            std::exit(0);
        }
        std::string result_file_temp = folderName + "result/" + filename;
        filename = folderName + filename;
        result_file.push_back(result_file_temp);
        cv::Mat image = cv::imread(filename);
        //cv::imshow("image",image);
        cv::Mat dst=cv::Mat(image.rows,image.cols,CV_8UC1);
	//cv::Mat dst
	//std::cout << "rows= " << image.rows << std::endl;
	//std::cout << "cols= " << image.cols << std::endl;
	
        cv::Mat intrinsic_matrix=(cv::Mat_<double>(3, 3) << fx,0,cx,0,fy,cy,0,0,1);
        cv::Mat distortion_coeffs=(cv::Mat_<double>(1, 4) << k1,k2,p1,p2);
        //cv::Mat distortion_coeffs=(cv::Mat_<double>(1, 5) << k1,k2,-2.1117168284769821e-02,p1,p2);
        cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
        //cv::fisheye::undistortImage(image, dst, intrinsic_matrix, distortion_coeffs,intrinsic_matrix,image.size());

	cv::Mat map1, map2;
	const int ImgActWid = 1242;//1856
	const int ImgActHei = 375;//1071
	//const double DisAlp = 1;
	//cv::Size imageSize(ImgActWid, ImgActHei);
    	//cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(intrinsic_matrix,distortion_coeffs, imageSize, DisAlp, imageSize, 0);
   	//initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
	//remap(image, dst, map1, map2, cv::INTER_LINEAR);


        //cv::undistort(image,dst,intrinsic_matrix,distortion_coeffs);
        //cv::imshow("dst",dst);
	//cv::imwrite("/home/lion/wxh/auto_cali_cha/test/dist_img.jpg",dst);
        images.push_back(image);

        if (!(readtxt >> filename))
        {
            std::cout << "\033[31oxtsData Error: no oxts!\033[0m" << std::endl;
            std::exit(0);
        }
        else
        {
            filename = folderName + filename;
            // std::cout<<"finename = "<<filename<<std::endl;
            getOxtsData(filename, oxts_vec);
	    
        }
        n++;
	//std::cout<<"numbers = "<< n <<std::endl;
    }
    // std::cout<<"oxts_vec size = " << oxts_vec.size() << std::endl;
    // for(int i = 0; i < oxts_vec.size(); i++)
    // {
    //     std::cout << "oxts = " << oxts_vec[i] << std::endl;
    // }
    // std::cout<<"images.size = "<<images.size()<<" " << "pc.size = " << pointclouds.size() << std::endl;
    std::cout << "Finish Read Data." << std::endl;
    return n;
}

void AutoCalib::find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                                     std::vector<cv::KeyPoint> &keypoints_1,
                                     std::vector<cv::KeyPoint> &keypoints_2,
                                     std::vector<cv::DMatch> &matches)
{

    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // #define ROI_LEFT 0
    // #define ROI_TOP images[ppp].rows/4
    // #define ROI_WIDTH images[ppp].cols
    // #define ROI_HEIGHT images[ppp].rows/2

    // std::vector<cv::KeyPoint> keypoints_1_temp, keypoints_2_temp;

    // for(int i = 0; i < keypoints_1.size(); ++i)
    // {
    //     if(keypoints_1[i].pt.x > 0+20 && keypoints_1[i].pt.x < 0+img_1.cols-20
    //        && keypoints_1[i].pt.y > img_1.rows/2+20 && keypoints_1[i].pt.y < img_1.rows/2+img_1.rows/2-20)
    //     {
    //         keypoints_1_temp.push_back(keypoints_1[i]);
    //     }
    // }
    // for(int i = 0; i < keypoints_2.size(); ++i)
    // {
    //     if(keypoints_2[i].pt.x > 0+20 && keypoints_2[i].pt.x < 0+img_2.cols-20
    //        && keypoints_2[i].pt.y > img_2.rows/2+20 && keypoints_2[i].pt.y < img_2.rows/2+img_2.rows/2-20)
    //     {
    //         keypoints_2_temp.push_back(keypoints_2[i]);
    //     }
    // }
    // keypoints_1.clear();
    // keypoints_2.clear();
    // keypoints_1 = keypoints_1_temp;
    // keypoints_2 = keypoints_2_temp;

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= std::max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

// using namespace std;
// using namespace cv;
int AutoCalib::recoverPose(cv::InputArray &E, cv::InputArray &_points1, cv::InputArray &_points2, cv::OutputArray &_R,
                           cv::OutputArray &_t, double &focal, cv::Point2d &pp, cv::InputOutputArray &_mask)
{
    cv::Mat points1, points2, cameraMatrix;
    cameraMatrix = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);
    cv::Mat P0 = cv::Mat::eye(3, 4, R1.type());
    cv::Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P1.col(3) = t * 1.0;
    P2(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
    P2.col(3) = t * 1.0;
    P3(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P3.col(3) = -t * 1.0;
    P4(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
    P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    cv::Mat Q;
    cv::triangulatePoints(P0, P1, points1, points2, Q);
    cv::Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    cv::triangulatePoints(P0, P2, points1, points2, Q);
    cv::Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    cv::triangulatePoints(P0, P3, points1, points2, Q);
    cv::Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    cv::triangulatePoints(P0, P4, points1, points2, Q);
    cv::Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty())
    {
        cv::Mat mask = _mask.getMat();
        CV_Assert(mask.size() == mask1.size());
        cv::bitwise_and(mask, mask1, mask1);
        cv::bitwise_and(mask, mask2, mask2);
        cv::bitwise_and(mask, mask3, mask3);
        cv::bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask4.copyTo(_mask);
        return good4;
    }
}

void AutoCalib::pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                                     std::vector<cv::KeyPoint> keypoints_2,
                                     std::vector<cv::DMatch> matches,
                                     cv::Mat &R, cv::Mat &t)
{
    // 相机内参,TUM Freiburg2
    // Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    // 相机内参
    //float cx = 325.5;
    //float cy = 253.5;
    //float fx = 518.0;
    //float fy = 519.0;
    //float depth_scale = 1000.0;

    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0, 0, 1;

    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    // cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    // Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
    // double focal_length = 521;			//相机焦距, TUM dataset标定值
    cv::Point2d principal_point(325.5, 253.5); //相机光心, TUM dataset标定值
    double focal_length = 519;                 //相机焦距, TUM dataset标定值
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    // cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    // cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "ORB -- R is " << std::endl
              << R << std::endl;
    std::cout << "ORB -- t is " << std::endl
              << t << std::endl;

    Eigen::Matrix3f eigen_rot; // = Eigen::Matrix3f::Identity();
    eigen_rot << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0),
        R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(2, 0),
        R.at<double>(2, 1), R.at<double>(2, 2);
    Eigen::Vector3f euler_ang_xyz = eigen_rot.eulerAngles(0, 1, 2); //x y z
    Eigen::Vector3f euler_ang_zyx = eigen_rot.eulerAngles(2, 1, 0); //z y x
    // std::cout << "euler_ang_xyz = " << std::endl << euler_ang_xyz * 180.0 / M_PI << std::endl;
    std::cout << "ORB -- euler_ang_zyx is " << std::endl
              << euler_ang_zyx * 180.0 / M_PI << std::endl;
}

void AutoCalib::send_transform_thread()
{
    tf::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf_trans);
    std::cout << "send transform thread" << std::endl;
}

void AutoCalib::down_sample_pc(pcl::PointCloud<pcl::PointXYZI>::Ptr &in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out, double leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZI> filter;
    filter.setInputCloud(in);
    filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    filter.filter(*out);
}

bool AutoCalib::pointcmp(PointXYZIA a, PointXYZIA b)
{
    return a.cosangle < b.cosangle;
}

void AutoCalib::extract_pc_feature_6(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature)
{
    float factor_t = ((upperBound - lowerBound) / (rings - 1));
    factor = ((rings - 1) / (upperBound - lowerBound));
    YAML::Node local_config = config["extract_pc_edges"];
    float dis_threshold = config["dis_threshold"].as<float>();
    std::vector<std::vector<float>> pc_image;
    std::vector<std::vector<float>> pc_image_copy;
    pc_image.resize(1000);
    pc_image_copy.resize(1000);
    // resize img and set to -1
    for (int i = 0; i < pc_image.size(); i++)
    {
        pc_image[i].resize(rings);
        pc_image_copy[i].resize(rings);
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image[i][j] = -1;
    }
    // convert pointcloud from 3D to 2D img
    for (size_t i = 0; i < pc->size(); i++)
    {
        float theta = 0;
        if (pc->points[i].y == 0)
            theta = 90.0;
        else if (pc->points[i].y > 0)
        {
            float tan_theta = pc->points[i].x / pc->points[i].y;
            theta = 180 * std::atan(tan_theta) / M_PI;
        }
        else
        {
            float tan_theta = -pc->points[i].y / pc->points[i].x;
            theta = 180 * std::atan(tan_theta) / M_PI;
            theta = 90 + theta;
        }
        int col = cvFloor(theta / 0.18); // theta [0, 180] ==> [0, 1000]
        if (col < 0 || col > 999)
            continue;
        float hypotenuse = std::sqrt(std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].y, 2));
        float angle = std::atan(pc->points[i].z / hypotenuse);
        int ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5);
        if (ring_id < 0 || ring_id > rings - 1)
            continue;
        float dist = std::sqrt(std::pow(pc->points[i].y, 2) + std::pow(pc->points[i].x, 2) + std::pow(pc->points[i].z, 2));
        if (dist < 2)
            continue; //10
        if (pc_image[col][ring_id] == -1)
        {
            pc_image[col][ring_id] = dist; //range
        }
        else if (dist < pc_image[col][ring_id])
        {
            pc_image[col][ring_id] = dist; //set the nearer point
        }
    }

    // // show pc_image by cv::imshow
    // cv::Mat pc_img = cv::Mat::zeros(1000, 64, CV_8UC1);
    // int cnt = 0;
    // float max_range = 0;
    // std::cout<<"0"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         if((int)pc_image[i][j] > max_range) max_range = (int)pc_image[i][j];
    //         if((int)pc_image[i][j] == -1) pc_image[i][j] = 0;
    //     }
    // }
    // std::cout<<"1"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         pc_image[i][j] = pc_image[i][j] / max_range * 255;
    //     }
    // }
    // std::cout<<"2"<<std::endl;
    // for(int i = 0; i < 1000; i++){
    //     for(int j = 0; j < 64; j++){
    //         pc_img.at<uchar>(i, j) = (int)pc_image[i][j];//这里的赋值不对
    //         std::cout << "pc_Img = " << (int)pc_image[i][j] << std::endl;
    //         if((int)pc_image[i][j] != -1) cnt++;
    //     }
    // }
    // std::cout<<"cnt = "<<cnt<<std::endl;
    // cv::namedWindow("pc_img", CV_WINDOW_NORMAL);
    // cv::imshow("pc_img", pc_img);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/pc_img.png", pc_img);
    // cv::waitKey(0);

    // copy
    for (int i = 0; i < pc_image.size(); i++)
    {
        for (int j = 0; j < pc_image[i].size(); j++)
            pc_image_copy[i][j] = pc_image[i][j];
    }
    for (int i = 1; i < rings - 1; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            float sum_dis = 0.0;
            int sum_n = 0;
            float far_sum_dis = 0.0;
            int far_sum_n = 0;
            float near_sum_dis = 0.0;
            int near_sum_n = 0;
            if (pc_image_copy[j - 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i - 1] - pc_image[j][i] > dis_threshold)
                    { //如果相邻点比此点远于一定阈值
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i - 1] > dis_threshold)
                    { //如果此点比相邻点远于一定阈值
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i];
                sum_n++;
            }
            if (pc_image_copy[j - 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j - 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j - 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j - 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j - 1][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j][i + 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i - 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i - 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i - 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i - 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i - 1];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i];
                sum_n++;
            }
            if (pc_image_copy[j + 1][i + 1] != -1)
            {
                if (pc_image[j][i] != -1)
                {
                    if (pc_image_copy[j + 1][i + 1] - pc_image[j][i] > dis_threshold)
                    {
                        far_sum_n++;
                        far_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                    else if (pc_image[j][i] - pc_image_copy[j + 1][i + 1] > dis_threshold)
                    {
                        near_sum_n++;
                        near_sum_dis += pc_image_copy[j + 1][i + 1];
                    }
                }
                sum_dis += pc_image_copy[j + 1][i + 1];
                sum_n++;
            }
            if (sum_n >= 5 && pc_image[j][i] == -1)
            {                                     //>=5
                pc_image[j][i] = sum_dis / sum_n; //如果周围点都有并且此点为-1则距离为平均
                continue;
            }
            if (near_sum_n > sum_n / 2)
            {
                pc_image[j][i] = near_sum_dis / near_sum_n; //如果周围点大多比此点近
            }
            if (far_sum_n > sum_n / 2)
            {
                pc_image[j][i] = far_sum_dis / far_sum_n; //如果周围点大多比此点远
            }
        }
    }

    //pc_image data structure
    //  **
    //  **   1000*64
    //  **

    //如果周围的点都为-1则为-1
    //   *
    //  *#*
    //   *
    for (int i = 0; i < rings; i++)
    {
        if (i == 0)
        { //处理第一列
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                { //处理pc_image第一行
                    if (pc_image[j][i + 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1;              //如果下一列和下一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i]; //否则为pc_image
                }
                else if (j == pc_image.size() - 1)
                { //处理pc_image最后一行
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //如果下一列和上一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //如果下一列和上一行和下一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else if (i == rings - 1)
        { //处理最后一列
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //如果上一列和下一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1)
                        pc_image[j][i] = -1; //如果上一列和上一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //如果上一列和上一行和下一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
        else
        {
            for (int j = 0; j < pc_image.size(); j++)
            {
                if (j == 0)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j + 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //如果上一列和下一行和下一列都为-1.则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else if (j == pc_image.size() - 1)
                {
                    if (pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j][i + 1] == -1)
                        pc_image[j][i] = -1; //如果上一列和上一行和下一列都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
                else
                {
                    if (pc_image[j][i + 1] == -1 && pc_image[j][i - 1] == -1 && pc_image[j - 1][i] == -1 && pc_image[j + 1][i] == -1)
                        pc_image[j][i] = -1; //如果下一列和上一列和上一行和下一行都为-1，则为-1
                    pc_image_copy[j][i] = pc_image[j][i];
                }
            }
        }
    }

    for (int i = 0; i < rings; i++)
    {
        for (int j = 1; j < pc_image.size() - 1; j++)
        {
            if (pc_image[j][i] == -1)
            { //如果此点为-1，但上一行和下一行都不为-1，取上一行或下一行较小的那个
                if (pc_image_copy[j - 1][i] != -1 && pc_image_copy[j + 1][i] != -1)
                {
                    pc_image[j][i] = pc_image_copy[j - 1][i] > pc_image_copy[j + 1][i] ? pc_image_copy[j + 1][i] : pc_image_copy[j - 1][i];
                }
            }
        }
    }

    // cv::Mat pc_img = cv::Mat::zeros();

    std::vector<std::vector<float>> mk_rings;
    for (int j = 0; j < pc_image.size(); j++)
    { //1000
        std::vector<float> mk_ring;
        mk_ring.clear();
        for (int i = 0; i < rings; i++)
        {
            // if(pc_image[j][i] != -1)//计算此列不等于-1的点
            {
                mk_ring.push_back(pc_image[j][i]); //存储一列rings个距离数据
                // mk_ring.push_back(j);
            }
        }
        mk_rings.push_back(mk_ring); //存储1000个列
    }
    // std::cout<<"0"<<std::endl;

    //pc_image data structure  竖直特征
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < rings; i++) //i<64
    {
        std::vector<float> mk;
        for (int j = 0; j < pc_image.size(); j++)
        { //j<1000
            if (pc_image[j][i] != -1)
            {                                 //计算此列不等于-1的点
                mk.push_back(pc_image[j][i]); //存距离(index: 0 2 4 6 ...)
                mk.push_back(j);              //存序号 0-999(index: 1 3 5 7 ...)
            }
        }
        if (mk.size() < 6)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++)
        { //mk的size等于一列(1000)个中不等于-1的个数×2
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //水平左边点距离此点大于一定阈值(与距离相关)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //水平右边点距离此点大于一定阈值(与距离相关)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || // && std::abs(mk[(j-1)*2+1]-mk[(j)*2+1])==1) || //水平左边点距离此点大于一定阈值(与距离相关)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())) || //&& std::abs(mk[(j+1)*2+1]-mk[(j)*2+1])==1) || //水平右边点距离此点大于一定阈值(与距离相关)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //水平角度距离大于一定阈值
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                if (i == 0) // bottom
                {
                    // 中间点距离 上 左上 右上 大于阈值   或者    中间点距离 上上 左上上 右上上  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    // ↓
                    //  **
                    // #** →上
                    //  **
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];

                    if ((abs(up - cen) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 上上 左上上 右上上  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    // * **
                    // *#**  ->上
                    // * **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 1)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // 中间点距离 下 左下 右下 大于阈值   或者    中间点距离 下下 左下下 右下下  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    // **
                    // **#  ->上
                    // **
                    if ((abs(cen - dw) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                if (i == rings - 2)
                {
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 下下 左下下 右下下  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    // ** *
                    // **#*  ->上
                    // ** *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (i > 1 && i < rings - 2)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[mk[2 * j + 1]][i];
                    float up = mk_rings[mk[2 * j + 1]][i + 1];
                    float dw = mk_rings[mk[2 * j + 1]][i - 1];
                    float lu = mk_rings[mk[2 * j - 1]][i + 1];
                    float ru = mk_rings[mk[2 * j + 3]][i + 1];
                    float ld = mk_rings[mk[2 * j - 1]][i - 1];
                    float rd = mk_rings[mk[2 * j + 3]][i - 1];
                    float uu = mk_rings[mk[2 * j + 1]][i + 2];
                    float dd = mk_rings[mk[2 * j + 1]][i - 2];
                    float luu = mk_rings[mk[2 * j - 1]][i + 2];
                    float ruu = mk_rings[mk[2 * j + 3]][i + 2];
                    float ldd = mk_rings[mk[2 * j - 1]][i - 2];
                    float rdd = mk_rings[mk[2 * j + 3]][i - 2];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 上上 左上上 右上上 下下 左下下 右下下  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    // ** **
                    // **#**  ->上
                    // ** **
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS) || (abs(uu - cen) > DIS && abs(luu - cen) > DIS && abs(ruu - cen) > DIS && abs(dd - cen) > DIS && abs(ldd - cen) > DIS && abs(rdd - cen) > DIS))
                    {
                        continue;
                    }
                }
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p;
                p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180.0) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180.0);
                p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180.0);
                // i->64  j->1000   mk[dis,j, dis,j, ...]
                //判断水平特征是否大于一定阈值，intensity取差值大的那一个，论文中差值越大表明越可能是一个边
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //取差值大的，存距离值
                }
                // TODO: 竖直方向的差值存到intensity里
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //取差值大的，存距离值
                }
                pc_feature->push_back(p); //delete horizontal features temply
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }

    //pc_image data structure  水平特征
    //  **
    //  **   1000*64
    //  **
    for (int i = 0; i < pc_image.size(); i++) //i<1000
    {
        std::vector<float> mk;

        for (int j = 0; j < rings; j++) //j<64
        {
            if (pc_image[i][j] != -1) //计算此列不等于-1的点
            {
                mk.push_back(pc_image[i][j]); //存距离(index: 0 2 4 6 ...)
                mk.push_back(j);              //存序号 0-64(index: 1 3 5 7 ...)
            }                                 //检查序号
        }

        if (mk.size() < 2)
            continue;

#define DIS 0.1
        for (int j = 1; j < (mk.size() / 2) - 1; j++) //mk的size等于一列(64)个中不等于-1的个数×2
        {                                             //j=1;j<64-1;j++
            // if(mk[(j-1)*2]!=-1 && (
            if (
                // mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //水平左边点距离此点大于一定阈值(与距离相关)
                // mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || //水平右边点距离此点大于一定阈值(与距离相关)
                ((mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || // && (std::abs(mk[(j-1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //水平左边点距离此点大于一定阈值(与距离相关)
                ((mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() * 1.5)) || //&& (std::abs(mk[(j+1)*2+1] - mk[(j)*2+1]) == 1 ))||//mk[(j)*2] / config["dis_threshold"].as<float>() || //水平右边点距离此点大于一定阈值(与距离相关)
                // mk[j*2+1] - mk[(j-1)*2+1] > local_config["angle_pixel_dis"].as<int>() || //水平角度距离大于一定阈值
                // mk[(j+1)*2+1] - mk[j*2+1] > local_config["angle_pixel_dis"].as<int>() ||
                local_config["show_all"].as<bool>())
            {

                // if(j == 0) // bottom
                // {
                //     // 中间点距离 上 左上 右上 大于阈值   或者    中间点距离 上上 左上上 右上上  大于阈值
                //     // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                //     //  **
                //     // #**
                //     //  **
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float up = mk_rings[i+1][mk[2*j+1]];
                //     float lu = mk_rings[i+1][mk[2*j-1]];
                //     float ru = mk_rings[i+1][mk[2*j+3]];
                //     float uu = mk_rings[i+2][mk[2*j+1]];
                //     float luu = mk_rings[i+2][mk[2*j-1]];
                //     float ruu = mk_rings[i+2][mk[2*j+3]];

                //     if( (abs(up-cen)>DIS&&abs(lu-cen)>DIS&&abs(ru-cen)>DIS)
                //         || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }

                //          j=[dis,j,dis,j...]
                //          ***
                // i=1000   ***
                //          ***
                //          ***

                if (j == 1 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 上上 左上上 右上上  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    /*// ****
                    // *#**  ->上
                    // *****/

                    // ***
                    // *#*  ->上
                    // ***
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS //不要左右的点，因为远处打到地面的点会被计算到这个里边，因为太远了，两线之间距离超过了DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // if(j == rings-1)
                // {
                //     float cen = mk_rings[i][mk[2*j+1]];
                //     float dw = mk_rings[i-1][mk[2*j+1]];
                //     float ld = mk_rings[i-1][mk[2*j-1]];
                //     float rd = mk_rings[i-1][mk[2*j+3]];
                //     float dd = mk_rings[i-2][mk[2*j+1]];
                //     float ldd = mk_rings[i-2][mk[2*j-1]];
                //     float rdd = mk_rings[i-2][mk[2*j+3]];

                //     // 中间点距离 下 左下 右下 大于阈值   或者    中间点距离 下下 左下下 右下下  大于阈值
                //     // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                //     // **
                //     // **#  ->上
                //     // **
                //     if( (abs(cen-dw)>DIS&&abs(ld-cen)>DIS&&abs(rd-cen)>DIS)
                //         || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                //     )
                //     {
                //         continue;
                //     }
                // }
                if (j == rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];
                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 下下 左下下 右下下  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    /* // ** *
                    // **#*  ->上
                    // ** *   */

                    // * *
                    // *#*  ->上
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // std::cout<<"1"<<std::endl;
                if (j > 1 && j < rings - 2 && i > 0 && i < pc_image.size() - 1)
                {
                    // std::cout << "mk size = " << mk.size() << std::endl;
                    // std::cout << "mk[(j+1)*2+1]  = " << mk[(j+1)*2+1] << std::endl;
                    // std::cout << "rings size = " << mk_rings.size() << std::endl;
                    // std::cout<<"j i = " << j << " " << i << std::endl;
                    float cen = mk_rings[i][mk[2 * j + 1]];
                    float up = mk_rings[i][mk[2 * j + 3]];
                    float dw = mk_rings[i][mk[2 * j - 1]];
                    float lu = mk_rings[i - 1][mk[2 * j + 3]];
                    float lt = mk_rings[i - 1][mk[2 * j + 1]];
                    float ru = mk_rings[i + 1][mk[2 * j + 3]];
                    float rt = mk_rings[i + 1][mk[2 * j + 1]];
                    float ld = mk_rings[i - 1][mk[2 * j - 1]];
                    float rd = mk_rings[i + 1][mk[2 * j - 1]];
                    float uu = mk_rings[i][mk[2 * j + 5]];
                    float dd = mk_rings[i][mk[2 * j - 3]];
                    float luu = mk_rings[i - 1][mk[2 * j + 5]];
                    float ruu = mk_rings[i + 1][mk[2 * j + 5]];
                    float ldd = mk_rings[i - 1][mk[2 * j - 3]];
                    float rdd = mk_rings[i + 1][mk[2 * j - 3]];

                    // std::cout << "cen = " << cen<<" "<<up << " "<<dw<<" "<<lu<<" "<<ru<<" "<<ld<<" "<<rd<<std::endl;
                    // if(abs(mk_rings[mk[2*j+1]][i-1]-mk_rings[mk[2*j+1]][i])>0.2 || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i])>0.2
                    // ||  abs(mk_rings[mk[2*j+1]][i-2]-mk_rings[mk[2*j+1]][i])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i])>0.24
                    //    || abs(mk_rings[mk[2*j+1]][i+1]-mk_rings[mk[2*j+1]][i-1])>0.24 || abs(mk_rings[mk[2*j+1]][i+2]-mk_rings[mk[2*j+1]][i-2])>0.3
                    // 中间点距离 上 下 左上 右上 左下 右下 大于阈值   或者    中间点距离 上上 左上上 右上上 下下 左下下 右下下  大于阈值
                    // 去除离群点，保证当前点周围两圈(除去两条边)至少有一个点
                    /*  // ** **
                    // **#**  ->上
                    // ** **  */

                    // * *
                    // *#*  ->上
                    // * *
                    if ((abs(up - cen) > DIS && abs(cen - dw) > DIS && abs(lu - cen) > DIS && abs(ru - cen) > DIS && abs(ld - cen) > DIS && abs(rd - cen) > DIS
                         // &&abs(lt-cen)>DIS&&abs(rt-cen)>DIS
                         )
                        // || (abs(uu-cen)>DIS&&abs(luu-cen)>DIS&&abs(ruu-cen)>DIS && abs(dd-cen)>DIS&&abs(ldd-cen)>DIS&&abs(rdd-cen)>DIS)
                    )
                    {
                        continue;
                    }
                }
                // i->1000  j->64   mk[dis,j, dis,j, ...]
                // recover the point from distance, the layer, and angle of the point
                pcl::PointXYZI p; //mk[(j)*2]是点的距离信息
                p.x = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::cos((i * 0.18 - 90) * M_PI / 180);
                p.y = mk[(j)*2] * std::cos((j * factor_t + lowerBound) * M_PI / 180) * std::sin(-(i * 0.18 - 90) * M_PI / 180);
                p.z = mk[(j)*2] * std::sin((mk[j * 2 + 1] * factor_t + lowerBound) * M_PI / 180);

                //判断水平特征是否大于一定阈值，intensity取差值大的那一个，差值越大，越可能是一个边缘
                if (mk[(j - 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>() || mk[(j + 1) * 2] - mk[(j)*2] > mk[(j)*2] / config["dis_threshold"].as<float>())
                {
                    // if(mk[(j-1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>() || mk[(j+1)*2] - mk[(j)*2] > config["dis_threshold"].as<float>()){
                    p.intensity = (mk[(j - 1) * 2] - mk[(j)*2]) > (mk[(j + 1) * 2] - mk[(j)*2]) ? (mk[(j - 1) * 2] - mk[(j)*2]) : (mk[(j + 1) * 2] - mk[(j)*2]); //取大，存距离差值
                }
                //
                else if (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1] > local_config["angle_pixel_dis"].as<int>() || mk[(j + 1) * 2 + 1] - mk[j * 2 + 1] > local_config["angle_pixel_dis"].as<int>())
                {
                    p.intensity = (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) > (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]) ? (mk[j * 2 + 1] - mk[(j - 1) * 2 + 1]) : (mk[(j + 1) * 2 + 1] - mk[j * 2 + 1]); //取大，存距离差值
                }
                // p.intensity = int(mk[j*2+1] * 30) % 255;
                p.intensity = 0.1;        // TEST: set 0.1 to label horizontal line features
                pc_feature->push_back(p); // not push back 水平特征
            }
            else if (local_config["add_edge"].as<bool>())
            {
                if (i != 0 && i != rings - 1)
                {
                    if (pc_image[mk[j * 2 + 1]][i + 1] != -1 && pc_image[mk[j * 2 + 1]][i + 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                    else if (pc_image[mk[j * 2 + 1]][i - 1] != -1 && pc_image[mk[j * 2 + 1]][i - 1] - pc_image[mk[j * 2 + 1]][i] > local_config["shuzhi_dis_th"].as<float>() * pc_image[mk[j * 2 + 1]][i])
                    {
                        pcl::PointXYZI p;
                        p.x = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::cos((mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.y = mk[(j)*2] * std::cos((i * factor_t + lowerBound) * M_PI / 180) * std::sin(-(mk[j * 2 + 1] * 0.18 - 90) * M_PI / 180);
                        p.z = mk[(j)*2] * std::sin((i * factor_t + lowerBound) * M_PI / 180);
                        p.intensity = 0.5;
                        pc_feature->push_back(p);
                    }
                }
            }
        }
    }

    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*pc_feature, *pc_cluster);
    if (pc_cluster->points.size() > 0)
    {
        kdtree->setInputCloud(pc_cluster);
    }
    std::vector<pcl::PointIndices> local_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclid;
    euclid.setInputCloud(pc_cluster);
    euclid.setClusterTolerance(0.3);
    euclid.setMinClusterSize(3);
    euclid.setMaxClusterSize(1000);
    euclid.setSearchMethod(kdtree);
    euclid.extract(local_indices);
    // std::cout<<"local_indices size = " << local_indices.size() << std::endl;

    std::vector<Detected_Obj> obj_list;
    for (size_t i = 0; i < local_indices.size(); i++)
    {
        // the structure to save one detected object
        Detected_Obj obj_info;

        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();

        for (auto pit = local_indices[i].indices.begin(); pit != local_indices[i].indices.end(); ++pit)
        {
            //fill new colored cluster point by point
            pcl::PointXYZ p;
            p.x = pc_feature->points[*pit].x;
            p.y = pc_feature->points[*pit].y;
            p.z = pc_feature->points[*pit].z;

            obj_info.centroid_.x += p.x;
            obj_info.centroid_.y += p.y;
            obj_info.centroid_.z += p.z;

            if (p.x < min_x)
                min_x = p.x;
            if (p.y < min_y)
                min_y = p.y;
            if (p.z < min_z)
                min_z = p.z;
            if (p.x > max_x)
                max_x = p.x;
            if (p.y > max_y)
                max_y = p.y;
            if (p.z > max_z)
                max_z = p.z;
        }

        //min, max points
        obj_info.min_point_.x = min_x;
        obj_info.min_point_.y = min_y;
        obj_info.min_point_.z = min_z;

        obj_info.max_point_.x = max_x;
        obj_info.max_point_.y = max_y;
        obj_info.max_point_.z = max_z;

        //calculate centroid, average
        if (local_indices[i].indices.size() > 0)
        {
            obj_info.centroid_.x /= local_indices[i].indices.size();
            obj_info.centroid_.y /= local_indices[i].indices.size();
            obj_info.centroid_.z /= local_indices[i].indices.size();
        }

        //calculate bounding box
        double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
        double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
        double height_ = obj_info.max_point_.z - obj_info.min_point_.z;

        // obj_info.bounding_box_.header = "object";

        obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
        obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
        obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

        obj_info.bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
        obj_info.bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
        obj_info.bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

        if (obj_info.bounding_box_.dimensions.x > 0.5 || obj_info.bounding_box_.dimensions.y > 0.5 || obj_info.bounding_box_.dimensions.z > 0.5)
        {
            obj_list.push_back(obj_info);
        }
    }
    // std::cout<<"obj list size = " << obj_list.size() << std::endl;
    // for(size_t i = 0; i < obj_list.size(); ++i)
    // {
    //     std::cout<<"obj list xyz = " <<obj_list[i].bounding_box_.pose.position.x<<" " <<obj_list[i].bounding_box_.pose.position.y<<
    //     " "<< obj_list[i].bounding_box_.pose.position.z << " " <<obj_list[i].bounding_box_.dimensions.x << " " << obj_list[i].bounding_box_.dimensions.y << " "
    //      << obj_list[i].bounding_box_.dimensions.z << std::endl;
    // }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_cluster(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t j = 0; j < obj_list.size(); ++j)
    {
        float x_min = obj_list[j].bounding_box_.pose.position.x - obj_list[j].bounding_box_.dimensions.x / 2;
        float x_max = obj_list[j].bounding_box_.pose.position.x + obj_list[j].bounding_box_.dimensions.x / 2;
        float y_min = obj_list[j].bounding_box_.pose.position.y - obj_list[j].bounding_box_.dimensions.y / 2;
        float y_max = obj_list[j].bounding_box_.pose.position.y + obj_list[j].bounding_box_.dimensions.y / 2;
        float z_min = obj_list[j].bounding_box_.pose.position.z - obj_list[j].bounding_box_.dimensions.z / 2;
        float z_max = obj_list[j].bounding_box_.pose.position.z + obj_list[j].bounding_box_.dimensions.z / 2;
        for (size_t i = 0; i < pc_feature->points.size(); ++i)
        {
            float pc_x = pc_feature->points[i].x;
            float pc_y = pc_feature->points[i].y;
            float pc_z = pc_feature->points[i].z;
            if (pc_x > x_min - 0.1 && pc_x < x_max + 0.1 && pc_y > y_min - 0.1 && pc_y < y_max + 0.1 && pc_z > z_min - 0.1 && pc_z < z_max + 0.1)
            {
                pcl::PointXYZI p;
                p.x = pc_x;
                p.y = pc_y;
                p.z = pc_z;
                p.intensity = pc_feature->points[i].intensity;
                // std::cout<<"xyz = "<<p.x<< " " <<  p.y << " " << p.z<<std::endl;
                pc_feature_cluster->push_back(p);
                // std::cout<<"after"<<std::endl;
            }
        }
    }
    if (config["cluster_pointcloud"].as<bool>())
    {
        pcl::copyPointCloud(*pc_feature_cluster, *pc_feature); // whether use the cluster method
    }

    // ros::Publisher pub_;
    // pub_ = nh.advertise<sensor_msgs::PointCloud2>("pc_feature_cluster", 50);

    // sensor_msgs::PointCloud2 cloud_msg;
    // pcl::toROSMsg(*pc_feature_cluster, cloud_msg);
    // cloud_msg.header.stamp = ros::Time::now();
    // cloud_msg.header.frame_id = "velodyne";

    // ros::Rate loop_rate(50);
    // while(ros::ok())
    // {
    //     ros::spinOnce();
    //     pub_.publish(cloud_msg);
    //     std::cout << "publish" << std::endl;
    //     loop_rate.sleep();
    // }

    //  //显示提取效果
    // pcl::visualization::PCLVisualizer viewer("pc Viewer");
    // pcl::visualization::PCLVisualizer viewer_feature("pc_feature Viewer");
    // pcl::visualization::PCLVisualizer viewer_feature_cluster("pc_feature Viewer Cluster");
    // //设置窗口背景颜色，范围为0-1
    // viewer.setBackgroundColor(0, 0, 0);
    // viewer_feature.setBackgroundColor(0, 0, 0);
    // viewer_feature_cluster.setBackgroundColor(0, 0, 0);
    // //添加坐标轴
    // viewer.addCoordinateSystem(1);
    // viewer_feature.addCoordinateSystem(1);
    // viewer_feature_cluster.addCoordinateSystem(1);
    // //根据点云里某个字段大小设置颜色
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pc, "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor_feature(pc_feature, "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor_feature_cluster(pc_feature_cluster, "z");
    // //往窗口添加点云并设置颜色
    // viewer.addPointCloud(pc, fildColor, "cloud");
    // viewer_feature.addPointCloud(pc_feature, fildColor_feature, "cloud_feature");
    // viewer_feature_cluster.addPointCloud(pc_feature_cluster, fildColor_feature, "cloud_feature_cluster");
    // //添加点云后，通过点云ID来设置显示大小
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
    // viewer_feature.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_feature");
    // viewer_feature_cluster.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_feature_cluster");
    // //重置相机，将点云显示到窗口
    // viewer.resetCamera();
    // viewer_feature.resetCamera();
    // viewer_feature_cluster.resetCamera();
    // while (!viewer.wasStopped())
    // {
    //   viewer.spinOnce();
    //   viewer_feature.spinOnce();
    //   viewer_feature_cluster.spinOnce();
    // }
}

float AutoCalib::dist2Point(int x1, int y1, int x2, int y2)
{
    return std::sqrt(double(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void AutoCalib::extract_image_feature(cv::Mat &img, cv::Mat &image2, std::vector<cv::line_descriptor::KeyLine> &keylines, std::vector<cv::line_descriptor::KeyLine> &keylines2,
                                      cv::Mat &outimg, cv::Mat &outimg_thin)
{
    cv::Mat mLdesc, mLdesc2;
    std::vector<std::vector<cv::DMatch>> lmatches;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    // std::cout<<"img channels = " << img.channels() << std::endl;
    // std::cout << (img.type() == CV_8UC1) << std::endl;
    // if(img.channels()==1)
    // {
    //     // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
    //     cv::Mat img_temp(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0));
    //     for (int i = 0; i < img.cols; i++)
    //     {
    //         for (int j = 0; j < img.rows; j++)
    //         {
    //             img_temp.at<cv::Vec3b>(j,i)[2] = img_temp.at<cv::Vec3b>(j,i)[1] =
    //             img_temp.at<cv::Vec3b>(j,i)[0] = (int) img.at<uchar>(j, i);
    //         }
    //     }
    //     img_temp.copyTo(img);
    //     // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
    //     // cv::imshow("cvt", output_image);
    //     // cv::waitKey(0);
    //     // std::cout << "after cvt" << std::endl;
    // }
    // if(image2.channels()==1)
    // {
    //     // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
    //     cv::Mat img_temp(image2.rows, image2.cols, CV_8UC3, cv::Scalar::all(0));
    //     for (int i = 0; i < image2.cols; i++)
    //     {
    //         for (int j = 0; j < image2.rows; j++)
    //         {
    //             img_temp.at<cv::Vec3b>(j,i)[2] = img_temp.at<cv::Vec3b>(j,i)[1] =
    //             img_temp.at<cv::Vec3b>(j,i)[0] = (int) image2.at<uchar>(j, i);
    //         }
    //     }
    //     img_temp.copyTo(image2);
    //     // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
    //     // cv::imshow("cvt", output_image);
    //     // cv::waitKey(0);
    //     // std::cout << "after cvt" << std::endl;
    // }

    // std::cout << (img.type() == CV_8UC1) << std::endl;

    lsd->detect(img, keylines, 1.2, 1);
    lsd->detect(image2, keylines2, 1.2, 1);
    int lsdNFeatures = 50;
    if (keylines.size() > lsdNFeatures)
    {
        std::sort(keylines.begin(), keylines.end(), [](const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
                  { return a.response > b.response; });
        keylines.resize(lsdNFeatures);
        for (int i = 0; i < lsdNFeatures; i++)
            keylines[i].class_id = i;
    }
    if (keylines2.size() > lsdNFeatures)
    {
        std::sort(keylines2.begin(), keylines2.end(), [](const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b)
                  { return a.response > b.response; });
        keylines2.resize(lsdNFeatures);
        for (int i = 0; i < lsdNFeatures; i++)
            keylines2[i].class_id = i;
    }
    // cv::Mat drawLines(img);
    // lsd->drawSegments(drawLines, keylines);
    // cv::imshow("lsd", drawLines);
    // cv::waitKey(0);

    // Create and LSD detector with standard or no refinement.
#if 0
    cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);//或者两种LSD算法，这边用的是standard的
#else
    cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
#endif
    std::vector<cv::Vec4f> lines_std;
    // Detect the lines
    cv::Mat img_gray;
    if (img.channels() == 1)
    {
        img.copyTo(img_gray);
    }
    else
    {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    }

    ls->detect(img_gray, lines_std); //这里把检测到的直线线段都存入了lines_std中，4个float的值，分别为起止点的坐标
    // Show found lines
    // cv::Mat drawnLines(img);
    // ls->drawSegments(drawnLines, lines_std);

    // cv::Mat img_draw = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    cv::Mat img_draw = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar::all(0));
    // cv::Mat::zeros(1000, 64, CV_8UC1);
    // for(int i = 0; i < img.cols; ++i)
    // {
    //     for(int j =0; j < img.rows; ++j)
    //     {
    //             if (img.at<uchar>(j, i) < 0 || img.at<uchar>(j, i) > 255) {
    //                 std::cout << "error" << std::endl;
    //                 exit(0);
    //             }
    //             img.at<uchar>(j, i) = 0;//255 - (int) edge_distance_image2.at<uchar>(j, i);
    //     }
    // }
    //draw
    for (int i = 0; i < lines_std.size(); ++i)
    {
        if (dist2Point(lines_std[i][0], lines_std[i][1], lines_std[i][2], lines_std[i][3]) > 7)
            cv::line(img_draw, cv::Point(lines_std[i][0], lines_std[i][1]), cv::Point(lines_std[i][2], lines_std[i][3]), cv::Scalar(255, 255, 255), 1, CV_AA);
    }
    // cv::imshow("Standard refinement", drawnLines);
    // cv::imshow("Standard refinement", img_draw);
    // std::cout<<"img draw " << img_draw.channels() << " " << img_draw.type() << std::endl;
    cv::threshold(img_draw, img_draw, 100, 255, cv::THRESH_BINARY);
    // cv::waitKey(0);

    static int m = 0;
    std::string m_string = std::to_string(m);

    std::string line_feature_string = "../data" + std::to_string(frame_cnt) + "/result/line_feature_" + m_string + ".png";
    cv::imwrite(line_feature_string, img_draw);
    m++;

    cv::Mat img_draw_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw.cols; i++)
    {
        for (int j = 0; j < img_draw.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw.cols - 5) && j <= (img_draw.rows - 5))
            {
                float cen = (int)img_draw.at<uchar>(j, i);
                float up = (int)img_draw.at<uchar>(j - 1, i);
                float dw = (int)img_draw.at<uchar>(j + 1, i);
                float lt = (int)img_draw.at<uchar>(j, i - 1);
                float rt = (int)img_draw.at<uchar>(j, i + 1);
                float lu = (int)img_draw.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw.at<uchar>(j + 1, i + 1);
                float uu = (int)img_draw.at<uchar>(j - 2, i);
                float dd = (int)img_draw.at<uchar>(j + 2, i);
                float luu = (int)img_draw.at<uchar>(j - 2, i - 1);
                float ruu = (int)img_draw.at<uchar>(j - 2, i + 1);
                float ldd = (int)img_draw.at<uchar>(j + 2, i - 1);
                float rdd = (int)img_draw.at<uchar>(j + 2, i + 1);
                float lluu = (int)img_draw.at<uchar>(j - 2, i - 2);
                float rruu = (int)img_draw.at<uchar>(j - 2, i + 2);
                float llu = (int)img_draw.at<uchar>(j - 1, i - 2);
                float rru = (int)img_draw.at<uchar>(j - 1, i + 2);
                float ll = (int)img_draw.at<uchar>(j, i - 2);
                float rr = (int)img_draw.at<uchar>(j, i + 2);
                float lld = (int)img_draw.at<uchar>(j + 1, i - 2);
                float rrd = (int)img_draw.at<uchar>(j + 1, i + 2);
                float lldd = (int)img_draw.at<uchar>(j + 2, i - 2);
                float rrdd = (int)img_draw.at<uchar>(j + 2, i + 2);

                // 竖直特征
                if (cen == 255 && (lu == 255 || up == 255 || ru == 255 || ld == 255 || dw == 255 || rd == 255) && lluu == 0 && llu == 0 && ll == 0 && lld == 0 && lldd == 0 && rruu == 0 && rru == 0 && rr == 0 && rrd == 0 && rrdd == 0)
                {
                    img_draw_copy.at<uchar>(j, i) = (int)img_draw.at<uchar>(j, i);
                }

                // // 水平特征
                if (cen == 255 && (lu == 255 || lt == 255 || ld == 255 || rt == 255 || ru == 255 || rd == 255 || rru == 255 || rr == 255 || rrd == 255) && lluu == 0 && luu == 0 && uu == 0 && ruu == 0 && rruu == 0 && lldd == 0 && ldd == 0 && dd == 0 && rdd == 0 && rrdd == 0)
                {
                    img_draw_copy.at<uchar>(j, i) = (int)img_draw.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imshow("canny_filter", image_edge_copy);
    // cv::waitKey(0);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter.png", image_edge_copy);

    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy.png", img_draw_copy);

    cv::Mat img_draw_copy_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw.cols; i++)
    {
        for (int j = 0; j < img_draw.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw.cols - 5) && j <= (img_draw.rows - 5))
            {
                float cen = (int)img_draw_copy.at<uchar>(j, i);
                float up = (int)img_draw_copy.at<uchar>(j - 1, i);
                float dw = (int)img_draw_copy.at<uchar>(j + 1, i);
                float lt = (int)img_draw_copy.at<uchar>(j, i - 1);
                float rt = (int)img_draw_copy.at<uchar>(j, i + 1);
                float lu = (int)img_draw_copy.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw_copy.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw_copy.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw_copy.at<uchar>(j + 1, i + 1);

                if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lu == 0 && ru == 0 && ld == 0 && rd == 0)
                {
                    img_draw_copy_copy.at<uchar>(j, i) = 0;
                }
                else
                {
                    img_draw_copy_copy.at<uchar>(j, i) = (int)img_draw_copy.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imshow("canny_filter_filter", image_edge_copy_copy);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter.png", image_edge_copy_copy);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy_copy.png", img_draw_copy_copy);

    cv::Mat img_draw_copy_copy_copy(img_draw.rows, img_draw.cols, CV_8UC1, cv::Scalar::all(0));
    for (int i = 0; i < img_draw_copy_copy.cols; i++)
    {
        for (int j = 0; j < img_draw_copy_copy.rows; j++)
        {
            if (i >= 5 && j >= 5 && i <= (img_draw_copy_copy.cols - 5) && j <= (img_draw_copy_copy.rows - 5))
            {
                float cen = (int)img_draw_copy_copy.at<uchar>(j, i);
                float up = (int)img_draw_copy_copy.at<uchar>(j - 1, i);
                float dw = (int)img_draw_copy_copy.at<uchar>(j + 1, i);
                float lt = (int)img_draw_copy_copy.at<uchar>(j, i - 1);
                float rt = (int)img_draw_copy_copy.at<uchar>(j, i + 1);
                float lu = (int)img_draw_copy_copy.at<uchar>(j - 1, i - 1);
                float ru = (int)img_draw_copy_copy.at<uchar>(j - 1, i + 1);
                float ld = (int)img_draw_copy_copy.at<uchar>(j + 1, i - 1);
                float rd = (int)img_draw_copy_copy.at<uchar>(j + 1, i + 1);

                if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lu == 0 && ru == 0 && ld == 0 && rd == 0)
                {
                    img_draw_copy_copy_copy.at<uchar>(j, i) = 0;
                }
                else
                {
                    img_draw_copy_copy_copy.at<uchar>(j, i) = (int)img_draw_copy_copy.at<uchar>(j, i);
                }
            }
        }
    }
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/img_draw_copy_copy_copy.png", img_draw_copy_copy_copy);

    cv::Mat line_temp = cv::Mat(cv::Size(img.rows, img.cols), CV_8UC1, cv::Scalar::all(0));
    // std::cout<<"linetemp0 type channel " << line_temp.type() << "  " << line_temp.channels() << std::endl;
    // for(int i = 0; i < img_draw.cols; i++)
    // {
    //     for(int j =0; j < img_draw.rows; j++)
    //     {
    //         line_temp.at<uchar>(j, i) = (int)img_draw.at<uchar>(j,i);//255 - (int) edge_distance_image2.at<uchar>(j, i);
    //     }
    // }
    if (config["filter_line_features"].as<bool>())
    {
        line_temp = img_draw_copy_copy_copy.clone();
    }
    else
    {
        line_temp = img_draw.clone();
    }

    // cv::imshow("img_draw_copy_copy_copy", img_draw_copy_copy_copy);
    // cv::imshow("img_draw", img_draw);
    // cv::imshow("line_temp", line_temp);
    // cv::waitKey(0);
    // cv::cvtColor(img_draw, line_temp, cv::COLOR_BGR2GRAY);

    // cv::Mat image_edge_copy_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));

    cv::Mat line_img(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));          // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img2(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));         // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img2_bitwise(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0)); // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img3(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));         // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img_thin(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));     // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);

    cv::bitwise_not(line_temp, line_temp);
    cv::threshold(line_temp, line_temp, 100, 255, cv::THRESH_BINARY);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_temp.png", line_temp);
    // cv::imshow("line_temp", line_temp);
    // cv::imshow("line_img", line_img);
    // std::cout<<"line_temp type " << line_temp.type() << " " << line_temp.channels() << std::endl;
    cv::distanceTransform(line_temp, line_img2, CV_DIST_C, 3); //Method1
    // line_img2.convertTo(line_img2, CV_8UC1, 255);
    // cv::imshow("line_img2", line_img2);
    // cv::distanceTransform(img, line_img, CV_DIST_L2, 5);// Method2
    // cv::imshow("img", img);
    // cv::imshow("line_img", line_img);
    // cv::imshow("line_img2 before", line_img2);
    // cv::bitwise_not(line_img2, line_img2);
    for (int i = 0; i < line_img2.cols; i++)
    {
        for (int j = 0; j < line_img2.rows; j++)
        {
            //         newBImgData[i*step+j] = 255- line_img2[i*step+j];
            line_img2_bitwise.at<uchar>(j, i) = 255 - (int)line_img2.at<uchar>(j, i);
            //         // if(((int)line_img2.at<uchar>(j, i)) > 200) line_img2.at<uchar>(j, i) = 0;
            //         // if(((int)line_img2.at<uchar>(j, i)) < 50) line_img2.at<uchar>(j, i) = 255;
        }
    }
    // line_img2_bitwise = 255 - line_img2;
    // cv::cvtColor(line_img2, line_img2, cv::COLOR_GRAY2BGR);//commont
    // cv::threshold(line_img2, line_img2, 200,255,CV_THRESH_BINARY);
    // cv::bitwise_not(line_img2, line_img2_bitwise);
    cv::normalize(line_img2, line_img3, normalize_config, 0, cv::NORM_INF, 1); // Method1
    // cv::imshow("line_img3", line_img3);
    cv::normalize(line_img2, line_img_thin, normalize_config_thin, 0, cv::NORM_INF, 1); // Method1
    // cv::imshow("line_img_thin", line_img_thin);
    // cv::waitKey(0);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img3before.png", line_img3);
    cv::Mat line_img3_after(line_img3.rows, line_img3.cols, CV_8UC1, cv::Scalar::all(0));     // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    cv::Mat line_img_thin_after(line_img3.rows, line_img3.cols, CV_8UC1, cv::Scalar::all(0)); // = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    for (int i = 0; i < line_img3.cols; i++)
    {
        for (int j = 0; j < line_img3.rows; j++)
        {
            if (line_img3.at<uchar>(j, i) < 0 || line_img3.at<uchar>(j, i) > 255)
            {
                std::cout << "error" << std::endl;
                exit(0);
            }
            line_img3_after.at<uchar>(j, i) = 255 - (int)line_img3.at<uchar>(j, i);
            line_img_thin_after.at<uchar>(j, i) = 255 - (int)line_img_thin.at<uchar>(j, i);
            // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
        }
    }
    // cv::GaussianBlur(line_img, line_img2, cv::Size(3, 3), 1, 1);// 2, 2
    // cv::normalize(line_img, line_img2, 0, 5., NORM_MINMAX); // Method2
    // cv::imshow("line_img2_bitwise", line_img2_bitwise);
    // cv::imshow("line_img3 after", line_img3);
    // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img2.png", line_img2);

    //cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img.png", line_img3_after);
    //cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/line_img_thin.png", line_img_thin_after);
    outimg = line_img3_after.clone();
    outimg_thin = line_img_thin_after.clone();

    //cv::imshow("outimg", outimg);
    //cv::imshow("outimg_thin", outimg_thin);
    //cv::Mat outimg_1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));// = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC1);
    //cv::Mat outimg_thin_1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    //cv::Mat structureElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7), cv::Point(-1, -1));
    //cv::erode(outimg, outimg_1, structureElement);               //调用腐蚀API
    //cv::imshow("侵蚀操作后：", outimg_1);
    //cv::erode(outimg_thin, outimg_thin_1, structureElement);               //调用腐蚀API
    //cv::imshow("侵蚀操作后1：", outimg_thin_1);
    //cv::dilate(outimg_1, outimg, structureElement, cv::Point(-1, -1), 1);               //调用膨胀API
    //cv::imshow("膨胀操作后：", outimg);
    //cv::dilate(outimg_thin_1, outimg_thin, structureElement, cv::Point(-1, -1), 1);               //调用膨胀API
    //cv::imshow("膨胀操作后：", outimg_thin_1);

    // outimg = cv::Mat(cv::Size(img.cols, img.rows),CV_8UC3);
    // cv::cvtColor(line_img3, outimg, cv::COLOR_GRAY2BGR);
    // cv::imshow("outimg", outimg);
    // cv::waitKey(0);

    //#pragma omp parallel for
    // for (int i = 0; i < edge_distance_image2.cols; i++) {
    //     for (int j = 0; j < edge_distance_image2.rows; j++) {
    //         if (edge_distance_image2.at<uchar>(j, i) < 0 || edge_distance_image2.at<uchar>(j, i) > 255) {
    //             std::cout << "error" << std::endl;
    //             exit(0);
    //         }
    //         gray_image.at<uchar>(j, i) = 255 - (int) edge_distance_image2.at<uchar>(j, i);
    //         // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
    //     }
    // }

    // lbd->compute(img, keylines, mLdesc);
    // lbd->compute(image2,keylines2,mLdesc2);
    // cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
    // bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    // std::vector<cv::DMatch> matches;
    // for(size_t i=0;i<lmatches.size();i++)
    // {
    //     const cv::DMatch& bestMatch = lmatches[i][0];
    //     const cv::DMatch& betterMatch = lmatches[i][1];
    //     float  distanceRatio = bestMatch.distance / betterMatch.distance;
    //     if (distanceRatio < 0.7)
    //         matches.push_back(bestMatch);
    // }

    // cv::Mat outImg;
    // std::vector<char> mask( lmatches.size(), 1 );
    // drawLineMatches( img, keylines, image2, keylines2, matches, outImg, cv::Scalar::all( -1 ), cv::Scalar::all( -1 ), mask, cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
    // cv::imshow( "Matches", outImg );
    // cv::waitKey(0);
}

void AutoCalib::project2image(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, cv::Mat raw_image, cv::Mat &output_image, Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{

    Eigen::Matrix<float, 3, 4> T_lidar2cam_top3_local, T_lidar2image_local; //lida2image=T_lidar2cam*(T_cam02cam2)*T_cam2image
    T_lidar2cam_top3_local = RT.topRows(3);
    T_lidar2image_local = camera_param * T_lidar2cam_top3_local;
    if (raw_image.channels() < 3 && raw_image.channels() >= 1)
    {
        // std::cout << "before cvt" << std::endl;
        // cv::cvtColor(raw_image, output_image, cv::COLOR_GRAY2BGR);
        cv::Mat output_image_3channels(raw_image.rows, raw_image.cols, CV_8UC3, cv::Scalar::all(0));
        for (int i = 0; i < raw_image.cols; i++)
        {
            for (int j = 0; j < raw_image.rows; j++)
            {
                output_image_3channels.at<cv::Vec3b>(j, i)[2] = output_image_3channels.at<cv::Vec3b>(j, i)[1] =
                    output_image_3channels.at<cv::Vec3b>(j, i)[0] = (int)raw_image.at<uchar>(j, i);
                //  (int) raw_image.at<uchar>(j, i);
                // output_image_3channels.at<cv::Vec3b>(j,i)[0] = (int) raw_image.at<uchar>(j, i);
            }
        }
        output_image_3channels.copyTo(output_image);
        // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
        // cv::imshow("cvt", output_image);
        // cv::waitKey(0);
        // std::cout << "after cvt" << std::endl;
    }
    else
    {
        raw_image.copyTo(output_image);
    }
    pcl::PointXYZI r;
    Eigen::Vector4f raw_point;
    Eigen::Vector3f trans_point;
    double deep, deep_config; //deep_config: normalize, max deep
    int point_r;
    deep_config = 80;
    point_r = 2;
    //std::cout << "image size; " << raw_image.cols << " * " << raw_image.rows << std::endl;
    for (int i = 0; i < pc->size(); i++)
    {
        r = pc->points[i];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point = T_lidar2image_local * raw_point;
        int x = (int)(trans_point(0, 0) / trans_point(2, 0));
        int y = (int)(trans_point(1, 0) / trans_point(2, 0));

        //cout<<"!!!@@@####"<<x<<" "<<y<<" ";

        if (x < 0 || x > (raw_image.cols - 1) || y < 0 || y > (raw_image.rows - 1))
            continue;
        deep = trans_point(2, 0) / deep_config;
        //deep = r.intensity / deep_config;
        int blue, red, green;
        if (deep <= 0.5)
        {
            green = (int)((0.5 - deep) / 0.5 * 255);
            red = (int)(deep / 0.5 * 255);
            blue = 0;
        }
        else if (deep <= 1)
        {
            green = 0;
            red = (int)((1 - deep) / 0.5 * 255);
            blue = (int)((deep - 0.5) / 0.5 * 255);
        }
        else
        {
            blue = 0;
            green = 0;
            red = 255;
        };

        cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(0, 255, 0), -1);
        // cv::circle(output_image, cv::Point2f(x, y), point_r, cv::Scalar(blue,green,red), -1);
    }
}

void AutoCalib::extractFeature(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs, std::vector<cv::Mat> images, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pc_feature,
                               std::vector<cv::Mat> &distance_image, std::vector<cv::Mat> &distance_image_thin)
{
    std::cout << "Start extract feature" << std::endl;
    int data_num = pcs.size();
    if (data_num != images.size())
    {
        std::cout << "\033[31mExtractFeatur Error: pointcloud num unequal to image num!\033[0m" << std::endl;
        std::exit(0);
    }

    float search_r, search_r2, search_r3;
    float x_max, x_min, y_min, y_max, z_min, z_max;
    int search_num, search_num2, search_num3;
    x_max = config["x_max"].as<float>();
    x_min = config["x_min"].as<float>();
    y_min = config["y_min"].as<float>();
    y_max = config["y_max"].as<float>();
    z_min = config["z_min"].as<float>();
    z_max = config["z_max"].as<float>();
    search_r = config["search_r"].as<float>();
    search_num = config["search_num"].as<int>();
    search_r2 = config["search_r2"].as<float>();
    search_num2 = config["search_num2"].as<int>();
    search_r3 = config["search_r3"].as<float>();
    search_num3 = config["search_num3"].as<int>();
    dis_threshold = config["dis_threshold"].as<float>();
    angle_threshold = config["angle_threshold"].as<float>();
    canny_threshold_mini = config["canny_threshold_mini"].as<int>();
    canny_threshold_max = config["canny_threshold_max"].as<int>();
    normalize_config = config["normalize_config"].as<int>();
    normalize_config_thin = config["normalize_config_thin"].as<int>();
    factor = ((rings - 1) / (upperBound - lowerBound));

    for (int i = 0; i < data_num; i++)
    {

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw = pcs[i];
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr edges(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*raw, *pc);
        pcl::PassThrough<pcl::PointXYZI> pass_filter;
        pass_filter.setInputCloud(pc);
        pass_filter.setFilterFieldName("y");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(y_min, y_max);
        pass_filter.filter(*filtered_y);

        pass_filter.setInputCloud(filtered_y);
        pass_filter.setFilterFieldName("x");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(x_min, x_max);
        pass_filter.filter(*filtered_x);

        pass_filter.setInputCloud(filtered_x);
        pass_filter.setFilterFieldName("z");
        pass_filter.setFilterLimitsNegative(false);
        pass_filter.setFilterLimits(z_min, z_max);
        pass_filter.filter(*filtered);

        filtered_pc.push_back(filtered);

        cv::Mat image1 = images[i];
        cv::Mat image = images[i];
        images_withouthist.push_back(image1);

        if (image1.channels() > 1)
        {
            cv::Mat imageRGB1[3];
            split(image1, imageRGB1);

            // cv::imshow("image_clone0", image_clone1);
            // cv::waitKey(0);

            for (int i = 0; i < 3; i++)
            {
                cv::equalizeHist(imageRGB1[i], imageRGB1[i]);
            }
            cv::merge(imageRGB1, 3, image1);
            //再一次增强
            cv::Mat imageRGB[3];
            split(image1, imageRGB);
            for (int i = 0; i < 3; i++)
            {
                cv::equalizeHist(imageRGB[i], imageRGB[i]);
            }
            cv::merge(imageRGB, 3, image);
        }

        // imshow("直方图均衡化图像增强效果", image);
        // cv::waitKey(0);

        // cv::Mat imageEnhance;
        // cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
        // cv::filter2D(image_clone, imageEnhance, CV_8UC3, kernel);
        // cv::imshow("拉普拉斯算子图像增强效果", imageEnhance);
        // cv::imshow("image_clone1", image_clone);
        // cv::waitKey(0);

        // cv::Mat imageLog(image_clone.size(), CV_32FC3);
        // for (int i = 0; i < image.rows; i++)
        // {
        //     for (int j = 0; j < image.cols; j++)
        //     {
        //         imageLog.at<cv::Vec3f>(i, j)[0] = log(1 + image_clone.at<cv::Vec3b>(i, j)[0]);
        //         imageLog.at<cv::Vec3f>(i, j)[1] = log(1 + image_clone.at<cv::Vec3b>(i, j)[1]);
        //         imageLog.at<cv::Vec3f>(i, j)[2] = log(1 + image_clone.at<cv::Vec3b>(i, j)[2]);
        //     }
        // }
        // //归一化到0~255
        // cv::normalize(imageLog, imageLog, 0, 255, CV_MINMAX);
        // //转换成8bit图像显示
        // cv::convertScaleAbs(imageLog, imageLog);
        // cv::imshow("LOG图像增强效果", imageLog);
        // cv::imshow("image_clone2", image_clone);
        // cv::waitKey(0);

        // cv::Mat imageGamma(image_clone.size(), CV_32FC3);
        // for (int i = 0; i < image.rows; i++)
        // {
        //     for (int j = 0; j < image.cols; j++)
        //     {
        //         imageGamma.at<cv::Vec3f>(i, j)[0] = (image_clone.at<cv::Vec3b>(i, j)[0])*(image_clone.at<cv::Vec3b>(i, j)[0])*(image_clone.at<cv::Vec3b>(i, j)[0]);
        //         imageGamma.at<cv::Vec3f>(i, j)[1] = (image_clone.at<cv::Vec3b>(i, j)[1])*(image_clone.at<cv::Vec3b>(i, j)[1])*(image_clone.at<cv::Vec3b>(i, j)[1]);
        //         imageGamma.at<cv::Vec3f>(i, j)[2] = (image_clone.at<cv::Vec3b>(i, j)[2])*(image_clone.at<cv::Vec3b>(i, j)[2])*(image_clone.at<cv::Vec3b>(i, j)[2]);
        //     }
        // }
        // //归一化到0~255
        // cv::normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);
        // //转换成8bit图像显示
        // cv::convertScaleAbs(imageGamma, imageGamma);
        // cv::imshow("伽马变换图像增强效果", imageGamma);
        // cv::imshow("image_clone3", image_clone);
        // cv::waitKey();

        extract_pc_feature_6(filtered, edges);

        pc_feature.push_back(edges);
        // cv::waitKey(0);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_noground(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered2(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered3(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered4(new pcl::PointCloud<pcl::PointXYZI>);
        //
        //
        //        pcl::PassThrough<pcl::PointXYZI> pass_filter;
        //        pass_filter.setInputCloud(pc);
        //        pass_filter.setFilterFieldName("x");
        //        pass_filter.setFilterLimitsNegative(false);
        //        pass_filter.setFilterLimits(x_min, x_max);
        //        pass_filter.filter(*filtered);
        //
        ////        pcl::RadiusOutlierRemoval<pcl::PointXYZI> r_filter;
        ////        r_filter.setInputCloud(filtered);
        ////        r_filter.setRadiusSearch(search_r);
        ////        r_filter.setMinNeighborsInRadius(search_num);
        ////        r_filter.filter(*filtered2);
        ////
        ////        pass_filter.setFilterLimits(x_max, FLT_MAX);
        ////        pass_filter.filter(*filtered);
        ////        r_filter.setInputCloud(filtered);
        ////        r_filter.setRadiusSearch(search_r2);
        ////        r_filter.setMinNeighborsInRadius(search_num2);
        ////        r_filter.filter(*filtered3);
        ////
        ////        *filtered4 = *filtered2 + *filtered3;
        //
        //        std::vector<int> indices;
        //        pcl::removeNaNFromPointCloud(*filtered, *filtered4, indices);
        //        std::vector<pcl::PointCloud<PointXYZIA> > point_rings;
        //        PointXYZIA point;
        //        point_rings.resize(rings);
        //
        //
        //        for (int i = 0; i < filtered4->size(); i++) {
        //
        //            PointXYZIA point;
        //            int ring_id;
        //            float xiebian, angle;
        //            point.x = filtered4->points[i].x;
        //            point.y = filtered4->points[i].y;
        //            point.z = filtered4->points[i].z;
        //            point.intensity = filtered4->points[i].intensity;
        //            xiebian = std::sqrt(std::pow(point.y, 2) + std::pow(point.x, 2));
        //            point.cosangle = -(point.y / xiebian);
        //            point.distance = std::sqrt(std::pow(point.y, 2) + std::pow(point.x, 2) + std::pow(point.z, 2));
        //            angle = std::atan(point.z / xiebian);
        //            ring_id = int(((angle * 180 / M_PI) - lowerBound) * factor + 0.5);
        //            if (ring_id >= rings || ring_id < 0) {
        //                //std::cout << "\033[33mWarning: one point cannot find a ring!\033[0m" << std::endl;
        //                continue;
        //            }
        //
        //            { point_rings[ring_id].push_back(point); }
        //        }
        //
        //
        ////extract feature
        //        pcl::PointXYZI p;
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr edges(new pcl::PointCloud<pcl::PointXYZI>);
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered5(new pcl::PointCloud<pcl::PointXYZI>);
        //        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> edge_filtered;
        //        edge_filtered.resize(rings);
        //        int edge1 = 0;
        //        int edge2 = 0;
        //
        ////#pragma omp parallel for
        //        for (int i = 0; i < rings; i++) {
        //            if (debug) {
        //                if (omp_in_parallel()) {
        //                    std::cout << "in parallel" << std::endl;
        //                    std::cout << omp_get_num_threads() << " threads" << std::endl;
        //                } else {
        //                    std::cout << "not in parallel" << std::endl;
        //                }
        //            }
        //            pcl::PointCloud<pcl::PointXYZI>::Ptr haha(new pcl::PointCloud<pcl::PointXYZI>);
        //            edge_filtered[i] = haha;
        //            if (point_rings[i].size() >= 3) {
        //                pcl::PointXYZI p;
        //                sort(point_rings[i].points.begin(), point_rings[i].points.end(), pointcmp);
        //

        //                for (int j = 1; j < point_rings[i].size() - 1; j++) {
        //                    if (point_rings[i].points[j - 1].cosangle > point_rings[i].points[j].cosangle) {
        //                        std::cout << "\033[31mcountScore Error: Sort error!\033[0m" << std::endl;
        //                        std::exit(0);
        //                    }
        //                    //intensity = score
        //                    if ((point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance) > dis_threshold ||
        //                        (point_rings[i].points[j + 1].distance - point_rings[i].points[j].distance) > dis_threshold) {
        //                        p.x = point_rings[i].points[j].x;
        //                        p.y = point_rings[i].points[j].y;
        //                        p.z = point_rings[i].points[j].z;
        //                        //p.intensity = point_rings[i].points[j-1].intensity;
        //                        p.intensity = (point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance) >
        //                                      (point_rings[i].points[j + 1].distance - point_rings[i].points[j].distance) ? (
        //                                              point_rings[i].points[j - 1].distance - point_rings[i].points[j].distance)
        //                                                                                                                  : (
        //                                              point_rings[i].points[j + 1].distance -
        //                                              point_rings[i].points[j].distance);
        //                        haha->points.push_back(p);
        //                    } else {
        //                        float aa = std::acos(point_rings[i].points[j - 1].cosangle);
        //                        float bb = std::acos(point_rings[i].points[j].cosangle);
        //                        float cc = std::acos(point_rings[i].points[j + 1].cosangle);
        //                        float dis_angle = (aa - bb) > (bb - cc) ? (aa - bb) * 180 / M_PI : (bb - cc) * 180 / M_PI;
        //                        if (dis_angle > angle_threshold) {
        //                            p.x = point_rings[i].points[j].x;
        //                            p.y = point_rings[i].points[j].y;
        //                            p.z = point_rings[i].points[j].z;
        //                            //p.intensity = point_rings[i].points[j-1].intensity;
        //                            p.intensity = (aa - bb) * 180 / M_PI;
        //                            haha->points.push_back(p);
        //                            //std::cout << dis_angle << std::endl;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //
        //
        //        for (int i = 0; i < rings; i++) {
        //            edge1 += edge_filtered[i]->size();
        //            (*filtered5) += (*edge_filtered[i]);
        //        }
        //        pcl::PointCloud<pcl::PointXYZI>::Ptr edges2(new pcl::PointCloud<pcl::PointXYZI>);
        //        std::cout << edge1 << ", " << edge2 << std::endl;
        ////        r_filter.setInputCloud(filtered5);
        ////        r_filter.setRadiusSearch(search_r3);
        ////        r_filter.setMinNeighborsInRadius(search_num3);
        ////        r_filter.filter(*edges2);
        //        float max_distance = config["max_distance"].as<float>();
        //        pass_filter.setInputCloud(filtered5);
        //        pass_filter.setFilterFieldName("x");
        //        pass_filter.setFilterLimitsNegative(false);
        //        pass_filter.setFilterLimits(0.00, max_distance);
        //        pass_filter.filter(*edges);
        //        pc_feature.push_back(edges);

        //extract image edges
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat raw_image, gray_image, gray_image_filter, hsv_image, edge_distance_image, edge_distance_image2, edge_distance_image3;
        cv::GaussianBlur(image, gray_image, cv::Size(5, 5), 2, 2); // 2, 2

        // cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        if (gray_image.channels() == 1)
        {
            gray_image.copyTo(gray_image);
        }
        else
        {
            cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        }
        cv::Mat image_edge;
        cv::Canny(gray_image, image_edge, canny_threshold_mini, canny_threshold_max);

        // cv::Sobel(gray_image, image_edge, -1, 1, 0);
        // cv::imshow("Canny", image_edge);
        // cv::waitKey(0);
        std::string i_string = std::to_string(i);

        std::string canny_string = "../data" + std::to_string(frame_cnt) + "/result/canny" + i_string + ".png";
        // cv::imwrite(canny_string, image_edge);

        cv::Mat image_edge_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge.cols; i++)
        {
            for (int j = 0; j < image_edge.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge.cols - 5) && j <= (image_edge.rows - 5))
                {
                    float cen = (int)image_edge.at<uchar>(j, i);
                    float up = (int)image_edge.at<uchar>(j - 1, i);
                    float dw = (int)image_edge.at<uchar>(j + 1, i);
                    float lt = (int)image_edge.at<uchar>(j, i - 1);
                    float rt = (int)image_edge.at<uchar>(j, i + 1);
                    float lu = (int)image_edge.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge.at<uchar>(j + 1, i + 1);
                    float uu = (int)image_edge.at<uchar>(j - 2, i);
                    float dd = (int)image_edge.at<uchar>(j + 2, i);
                    float luu = (int)image_edge.at<uchar>(j - 2, i - 1);
                    float ruu = (int)image_edge.at<uchar>(j - 2, i + 1);
                    float ldd = (int)image_edge.at<uchar>(j + 2, i - 1);
                    float rdd = (int)image_edge.at<uchar>(j + 2, i + 1);
                    float lluu = (int)image_edge.at<uchar>(j - 2, i - 2);
                    float rruu = (int)image_edge.at<uchar>(j - 2, i + 2);
                    float llu = (int)image_edge.at<uchar>(j - 1, i - 2);
                    float rru = (int)image_edge.at<uchar>(j - 1, i + 2);
                    float ll = (int)image_edge.at<uchar>(j, i - 2);
                    float rr = (int)image_edge.at<uchar>(j, i + 2);
                    float lld = (int)image_edge.at<uchar>(j + 1, i - 2);
                    float rrd = (int)image_edge.at<uchar>(j + 1, i + 2);
                    float lldd = (int)image_edge.at<uchar>(j + 2, i - 2);
                    float rrdd = (int)image_edge.at<uchar>(j + 2, i + 2);

                    // 竖直特征
                    /*// if((cen==255||lu==255||up==255||ru==255||ld==255||dw==255||rd==255||luu==255||uu==255||ruu==255||
                    //     ldd==255||dd==255||rdd==255)
                    //     // && lluu==0&&llu==0&&ll==0&&lld==0&&lldd==0 && rruu==0&&rru==0&&rr==0&&rrd==0&&rrdd==0
                    //     )
                    // {
                    //     image_edge_copy.at<uchar>(j, i) = (int) image_edge.at<uchar>(j, i);
                    // } */
                    if (cen == 255 && (lu == 255 || up == 255 || ru == 255 || ld == 255 || dw == 255 || rd == 255) && lluu == 0 && llu == 0 && ll == 0 && lld == 0 && lldd == 0 && rruu == 0 && rru == 0 && rr == 0 && rrd == 0 && rrdd == 0)
                    {
                        image_edge_copy.at<uchar>(j, i) = (int)image_edge.at<uchar>(j, i);
                    }

                    // 水平特征
                    // if((cen==255||lu==255||lt==255||ld==255||llu==255||ll==255||lld==255||rt==255||ru==255||rd==255||
                    //     rru==255||rr==255||rrd==255)
                    //     && lluu==0&&luu==0&&uu==0&&ruu==0&&rruu==0 && lldd==0&&ldd==0&&dd==0&&rdd==0&&rrdd==0
                    //     )
                    // {
                    //     image_edge_copy.at<uchar>(j, i) = (int) image_edge.at<uchar>(j, i);
                    // }
                }
            }
        }
        // cv::imshow("canny_filter", image_edge_copy);
        // cv::waitKey(0);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter.png", image_edge_copy);

        cv::Mat image_edge_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge_copy.cols; i++)
        {
            for (int j = 0; j < image_edge_copy.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge_copy.cols - 5) && j <= (image_edge_copy.rows - 5))
                {
                    float cen = (int)image_edge_copy.at<uchar>(j, i);
                    float up = (int)image_edge_copy.at<uchar>(j - 1, i);
                    float dw = (int)image_edge_copy.at<uchar>(j + 1, i);
                    float lt = (int)image_edge_copy.at<uchar>(j, i - 1);
                    float rt = (int)image_edge_copy.at<uchar>(j, i + 1);
                    float lu = (int)image_edge_copy.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge_copy.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge_copy.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge_copy.at<uchar>(j + 1, i + 1);

                    if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lt == 0 && rt == 0 && ld == 0 && rd == 0)
                    {
                        image_edge_copy_copy.at<uchar>(j, i) = 0;
                    }
                    else
                    {
                        image_edge_copy_copy.at<uchar>(j, i) = (int)image_edge_copy.at<uchar>(j, i);
                    }
                }
            }
        }
        // cv::imshow("canny_filter_filter", image_edge_copy_copy);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter.png", image_edge_copy_copy);

        cv::Mat image_edge_copy_copy_copy(image_edge.rows, image_edge.cols, CV_8UC1, cv::Scalar::all(0));
        for (int i = 0; i < image_edge_copy_copy.cols; i++)
        {
            for (int j = 0; j < image_edge_copy_copy.rows; j++)
            {
                if (i >= 5 && j >= 5 && i <= (image_edge_copy_copy.cols - 5) && j <= (image_edge_copy_copy.rows - 5))
                {
                    float cen = (int)image_edge_copy_copy.at<uchar>(j, i);
                    float up = (int)image_edge_copy_copy.at<uchar>(j - 1, i);
                    float dw = (int)image_edge_copy_copy.at<uchar>(j + 1, i);
                    float lt = (int)image_edge_copy_copy.at<uchar>(j, i - 1);
                    float rt = (int)image_edge_copy_copy.at<uchar>(j, i + 1);
                    float lu = (int)image_edge_copy_copy.at<uchar>(j - 1, i - 1);
                    float ru = (int)image_edge_copy_copy.at<uchar>(j - 1, i + 1);
                    float ld = (int)image_edge_copy_copy.at<uchar>(j + 1, i - 1);
                    float rd = (int)image_edge_copy_copy.at<uchar>(j + 1, i + 1);

                    if (cen == 255 && up == 0 && dw == 0 && lt == 0 && rt == 0 && lt == 0 && rt == 0 && ld == 0 && rd == 0)
                    {
                        image_edge_copy_copy_copy.at<uchar>(j, i) = 0;
                    }
                    else
                    {
                        image_edge_copy_copy_copy.at<uchar>(j, i) = (int)image_edge_copy_copy.at<uchar>(j, i);
                    }
                }
            }
        }
        // cv::imshow("canny_filter_filter_filter", image_edge_copy_copy_copy);
        // cv::waitKey(0);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/canny_filter_filter_filter.png", image_edge_copy_copy_copy);

        // cv::Mat grad_x, abs_grad_x;
        // cv::Sobel(image_edge, grad_x, CV_16S, 1, 0, 1, 1, 1, cv::BORDER_DEFAULT);
        // cv::convertScaleAbs(grad_x, abs_grad_x);
        // cv::imshow("sobel", abs_grad_x);
        //  cv::waitKey(0);
        // std::vector<cv::Vec4i> Lines;
        // cv::HoughLinesP(image_edge, Lines, 1, CV_PI/180, 10, 10, 20);
        // for(int i = 0; i < Lines.size(); i++)
        // {
        //     if(abs(Lines[i][0]-Lines[i][2]) > 5) continue;
        //     cv::line(image, cv::Point(Lines[i][0], Lines[i][1]), cv::Point(Lines[i][2], Lines[i][3]), cv::Scalar(0, 0, 255), 2, 8);
        // }
        // cv::imshow("hough", image);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/hough.png", image);
        // cv::waitKey(0);
        // cv::imshow("image_edge_copy_copy_copy", image_edge_copy_copy_copy);
        // cv::waitKey(0);
        if (config["filter_edge_features"].as<bool>())
        {
            cv::bitwise_not(image_edge_copy_copy_copy, image_edge);
        }
        else
        {
            cv::bitwise_not(image_edge, image_edge);
        }

        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/image_edge.png", image_edge);
        // std::cout<<"imgage_edge type " << image_edge.type() << image_edge.channels() << std::endl;
        cv::distanceTransform(image_edge, edge_distance_image, CV_DIST_C, 3); //Method1
        // cv::distanceTransform(image_edge, edge_distance_image, CV_DIST_L2, 5);// Method2
        // cv::imshow("image_edge", image_edge);
        // cv::imshow("edge_distance_image", edge_distance_image);
        cv::normalize(edge_distance_image, edge_distance_image2, normalize_config, 0, cv::NORM_INF, 1); // Method1
                                                                                                        // std::cout << "img " << image_edge_copy_copy_copy.rows << " " << image_edge_copy_copy_copy.cols <<" " << image_edge.rows << " "
                                                                                                        // << image_edge.cols << " " << edge_distance_image.rows << " " << edge_distance_image.cols << " " <<
                                                                                                        // " " << edge_distance_image2.rows << " " << edge_distance_image2.cols << std::endl;
                                                                                                        // std::cout << "channels image_edge edge_distance_image edge_distance_image2 gray_image = " << image_edge.channels() << " " <<
                                                                                                        // edge_distance_image.channels() << " " << edge_distance_image2.channels() << std::endl;

        // cv::GaussianBlur(edge_distance_image, edge_distance_image2, cv::Size(3, 3), 1, 1);// 2, 2
        // cv::normalize(edge_distance_image, edge_distance_image2, 0, 5., NORM_MINMAX); // Method2
        // cv::imshow("edge_distance_image2", edge_distance_image2);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/edge_distance_image.png", edge_distance_image);
        // cv::imwrite("/home/zh/code/useful_tools/auto_calibration/data/result/edge_distance_image2.png", edge_distance_image2);
        // cv::waitKey(0);
        // std::cout<<"edge distance imag2 00 = " << (int) edge_distance_image2.at<uchar>(0, 0) <<std::endl;
        // std::cout<<"edge_distance_image2 = "<<edge_distance_image2<<std::endl;

//        cv::findContours(image_edge, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//        cv::drawContours(image, contours, -1, cv::Scalar(0,0,255), 1, 8);
//      cv::imshow("test", image);
#pragma omp parallel for
        for (int i = 0; i < edge_distance_image2.cols; i++)
        {
            for (int j = 0; j < edge_distance_image2.rows; j++)
            {
                if (edge_distance_image2.at<uchar>(j, i) < 0 || edge_distance_image2.at<uchar>(j, i) > 255)
                {
                    std::cout << "error" << std::endl;
                    exit(0);
                }
                gray_image.at<uchar>(j, i) = 255 - (int)edge_distance_image2.at<uchar>(j, i);
                // std::cout << (int)gray_image.at<uchar>(j, i) << std::endl;
            }
        }
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/gray_image.png", gray_image);
        gray_image_vec.push_back(gray_image);
        // std::cout<<"gray_image 00 = " << (int) gray_image.at<uchar>(0, 0) <<std::endl;
        gray_image_filter = gray_image.clone();
        bool white_block;
        for (int i = 0; i < gray_image.cols; i++)
        {
            for (int j = 0; j < gray_image.rows; j++)
            {
                if (gray_image.at<uchar>(j, i) < 0 || gray_image.at<uchar>(j, i) > 255)
                {
                    std::cout << "error" << std::endl;
                    exit(0);
                }
                white_block = true;
                // std::cout<<"0"<<std::endl;
                if (i >= 10 && j >= 10 && i <= (gray_image.cols - 10) && j <= (gray_image.rows - 10))
                {
                    // std::cout<<"1"<<std::endl;
                    int img_cnt = 0;
                    int gray_pixel = 0;
                    for (int m = -5; m <= 5; m++)
                    {
                        for (int n = -5; n <= 5; n++)
                        {
                            // std::cout<<"2"<<std::endl;
                            img_cnt++;
                            gray_pixel += (int)gray_image.at<uchar>(j + m, i + n);
                            // if(abs((int)gray_image.at<uchar>(j+m, i+n)-255)>5){
                            //     // std::cout<<"3"<<std::endl;
                            //     white_block = false;
                            // }
                        }
                    }
                    gray_pixel = gray_pixel / img_cnt;
                    if (gray_pixel > 200)
                    {
                        // for(int m = -3; m <= 3; m++){
                        // for(int n = -3; n <= 3; n++){
                        // gray_image_filter.at<uchar>(j, i) = 127;
                        // }
                        // }
                    }
                }
                // std::cout<<"4"<<std::endl;
                // if(white_block){
                //     gray_image_filter.at<uchar>(j, i) = 0;
                // }
                // std::cout<<"5"<<std::endl;
                //std::cout << (int)edge_distance_image2.at<uchar>(j, i) << std::endl;
            }
        }
        // cv::imshow("gray_image", gray_image);
        // cv::imshow("gray_image_filter", gray_image_filter);
        // cv::waitKey(0);
        std::ostringstream str;
        str << i;
        // std::string image_gray = result_file[i].substr(0, result_file[i].length()-4);
        // std::string image_gray = "/home/zh/code/useful_tools/auto_calibration/data/result/image_gray" +  str.str() + ".png";
        // image_gray = "edge" + image_gray + ".png";
        // cv::imwrite(image_gray, gray_image_filter);

        // distance_image.push_back(gray_image);

        if (config["edge_features"].as<bool>())
        {
            distance_image.push_back(gray_image_filter);
        }
        else if (config["line_features"].as<bool>())
        {
            cv::Mat outimg, outimg_thin;
            std::vector<cv::line_descriptor::KeyLine> keylines, keylines2;
            extract_image_feature(image, image, keylines, keylines2, outimg, outimg_thin);
            distance_image.push_back(outimg); //outimg是单通道要改为三通道
            distance_image_thin.push_back(outimg_thin);
        }
    }
    std::cout << "End extract feature" << std::endl;
}

float AutoCalib::countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                            Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{
    float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//如果有一个像素点大于150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//周围必须有像素点大于150
        points_num++;

        // if((int) distance_image.at<uchar>(y, x) < 150)continue;//129

        double pt_dis = pow(r.x * r.x + r.y * r.y + r.z * r.z, double(1.0 / 2.0));
        //std::cout << r.x << "  " << r.y << "   " << r.z << "  " << pt_dis << std::endl;
        if (config["add_dis_weight"].as<bool>())
        {
            // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
            if (abs(r.intensity - 0.1) < 0.2)
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2) * 3;
            }
            else
            {
                one_score += (distance_image.at<uchar>(y, x) / pt_dis * 2);
            }
        }
        else
        {
            one_score += distance_image.at<uchar>(y, x);
        }
    }
    // score = one_score;// / (float)data_num;
    score = one_score / 255.0 / points_num;
    if (config["many_or_one"].as<int>() == 2)
    {
        std::cout << "has " << points_num << std::endl;
    }

    return score;
}

float AutoCalib::countConfidence(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param)
{
    float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    float points_whiter_200 = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//如果有一个像素点大于150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//周围必须有像素点大于150
        points_num++;
        if (distance_image.at<uchar>(y, x) > 130)
        {
            points_whiter_200++;
        }
    }

    score = points_whiter_200 / points_num;

    return score;
}

bool AutoCalib::isWhiteEnough(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                              Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, bool fine_result, float &score)
{
    // float score = 0;
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    //count Score
    int edge_size = pc_feature->size();
    float one_score = 0;
    float points_num = 0;
    float points_whiter_200 = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        //one_score += r.intensity * (int)distance_image[i].at<uchar>(y, x);

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//如果有一个像素点大于150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;//周围必须有像素点大于150
        points_num++;
        if (distance_image.at<uchar>(y, x) > 130)
        {
            points_whiter_200++;
        }

        // if((int) distance_image.at<uchar>(y, x) < 150)continue;//129
    }
    // std::cout << "random sample ========= " << points_whiter_200 / points_num << std::endl;
    score = points_whiter_200 / points_num;
    if (fine_result)
    {
        if (score > 0.6)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        if (score > 0.92)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

void AutoCalib::filterUnusedPoiintCloudFeature(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                               Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered)
{
    Eigen::Matrix<float, 3, 4> RT_TOP3, RT_X_CAM; //lida2image=T*(T_cam02cam2)*T_cam2image
    RT_TOP3 = RT.topRows(3);
    RT_X_CAM = camera_param * RT_TOP3;

    int edge_size = pc_feature->size();
    int one_score = 0;
    int points_num = 0;
    for (int j = 0; j < edge_size; j++)
    {
        pcl::PointXYZI r;
        Eigen::Vector4f raw_point;
        Eigen::Vector3f trans_point3;
        r = pc_feature->points[j];
        raw_point(0, 0) = r.x;
        raw_point(1, 0) = r.y;
        raw_point(2, 0) = r.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (distance_image.cols - 1) || y < 0 || y > (distance_image.rows - 1))
            continue;
        if (r.intensity < 0 || distance_image.at<uchar>(y, x) < 0)
        {
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
            exit(0);
        }
        points_num++;

        // bool gray128 = false;
        // for(int m = -5; m <= 5; m++){
        //     for(int n = -5; n <= 5; n++){
        //         if((int) distance_image.at<uchar>(y+m, x+n) > 150){//如果有一个像素点大于150
        //             gray128 = true;
        //         }
        //     }
        // }
        // if(!gray128)
        //     continue;
        pc_feature_filtered->push_back(r);
    }
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> AutoCalib::get_in_pcs()
{
    return in_pcs;
}

std::vector<cv::Mat> AutoCalib::get_in_images()
{
    return in_images;
}
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> AutoCalib::get_in_pcs_feature()
{
    return in_pcs_feature;
}

std::vector<cv::Mat> AutoCalib::get_in_images_feature()
{
    return in_images_feature;
}
Eigen::Matrix4f AutoCalib::get_in_pcs_current_guess()
{
    std::vector<float> ext = config["T_frame2frame0_pcs"]["data"].as<std::vector<float>>();
    Eigen::Matrix4f T_frame2frame0_pcs;
    assert((int)ext.size() == 16);
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            T_frame2frame0_pcs(row, col) = ext[row * 4 + col];
        }
    }
    return T_frame2frame0_pcs;
}
Eigen::Matrix4f AutoCalib::get_in_images_current_guess()
{
    std::vector<float> ext = config["T_frame2frame0_images"]["data"].as<std::vector<float>>();
    Eigen::Matrix4f T_frame2frame0_images;
    assert((int)ext.size() == 16);
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            T_frame2frame0_images(row, col) = ext[row * 4 + col];
        }
    }
    return T_frame2frame0_images;
}
Eigen::Matrix3f AutoCalib::get_in_k()
{
    return T_cam2image;
}
std::vector<Eigen::Matrix4f> AutoCalib::get_in_result_gt_vec()
{
    return result_gt_vec;
}
std::vector<Eigen::Matrix4f> AutoCalib::get_in_calibrated_result_vec()
{
    return calibrated_result_vec;
}
bool AutoCalib::get_in_overlap()
{
    return overlap;
}
bool AutoCalib::get_in_add_dis_weight()
{
    return add_dis_weight;
}
Sophus::SE3d AutoCalib::toSE3d(Eigen::Matrix4f &T)
{
    Eigen::Matrix3d R;
    R(0, 0) = T(0, 0);
    R(0, 1) = T(0, 1);
    R(0, 2) = T(0, 2);
    R(1, 0) = T(1, 0);
    R(1, 1) = T(1, 1);
    R(1, 2) = T(1, 2);
    R(2, 0) = T(2, 0);
    R(2, 1) = T(2, 1);
    R(2, 2) = T(2, 2);
    Eigen::Quaterniond q(R);

    Eigen::Vector3d t(T(0, 3), T(1, 3), T(2, 3));
    Sophus::SE3d result(q, t);
    return result;
}
Eigen::Matrix4f AutoCalib::toMatrix4f(Eigen::Matrix4d s)
{
    Eigen::Matrix4f result;
    result(0, 0) = s(0, 0);
    result(0, 1) = s(0, 1);
    result(0, 2) = s(0, 2);
    result(0, 3) = s(0, 3);
    result(1, 0) = s(1, 0);
    result(1, 1) = s(1, 1);
    result(1, 2) = s(1, 2);
    result(1, 3) = s(1, 3);
    result(2, 0) = s(2, 0);
    result(2, 1) = s(2, 1);
    result(2, 2) = s(2, 2);
    result(2, 3) = s(2, 3);
    result(3, 0) = s(3, 0);
    result(3, 1) = s(3, 1);
    result(3, 2) = s(3, 2);
    result(3, 3) = s(3, 3);
    return result;
}
void AutoCalib::Run()
{
    cv::Mat first_distance_image;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs_3frames;
    std::vector<cv::Mat> images;
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_pcl_edge(new pcl::PointCloud<pcl::PointXYZI>);

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pc_features;
    std::vector<cv::Mat> distance_images, distance_images_thin;

    std::vector<float> oxts_vec;

    int dataNum = getData(config["txtname"].as<std::string>(), config["foldername"].as<std::string>(), pcs, images, oxts_vec);

    in_images = images;
    in_pcs = pcs;

    pcl::PointCloud<pcl::PointXYZI>::Ptr frame12(new pcl::PointCloud<pcl::PointXYZI>);
    const double m_deg = 40075040.0 / 360.0;
    const double m_min = m_deg / 60.0;
    const double m_sec = m_min / 60.0;
    // std::cout << "m_deg " << std::setprecision(8)<< m_deg << " " << m_min << " " << m_sec << std::endl;

    // Transform and Rotation Matrix, imu to velodyne.
    Eigen::Matrix4d T_imu2velo;
    Eigen::Matrix3d R_imu2velo;
    T_imu2velo(0, 0) = 9.999976e-01;
    T_imu2velo(0, 1) = 7.553071e-04;
    T_imu2velo(0, 2) = -2.035826e-03;
    T_imu2velo(0, 3) = -8.086759e-01;
    T_imu2velo(1, 0) = -7.854027e-04;
    T_imu2velo(1, 1) = 9.998898e-01;
    T_imu2velo(1, 2) = -1.482298e-02;
    T_imu2velo(1, 3) = 3.195559e-01;
    T_imu2velo(2, 0) = 2.024406e-03;
    T_imu2velo(2, 1) = 1.482454e-02;
    T_imu2velo(2, 2) = 9.998881e-01;
    T_imu2velo(2, 3) = -7.997231e-01;
    T_imu2velo(3, 0) = 0;
    T_imu2velo(3, 1) = 0;
    T_imu2velo(3, 2) = 0;
    T_imu2velo(3, 3) = 1;

    R_imu2velo(0, 0) = 9.999976e-01;
    R_imu2velo(0, 1) = 7.553071e-04;
    R_imu2velo(0, 2) = -2.035826e-03;
    R_imu2velo(1, 0) = -7.854027e-04;
    R_imu2velo(1, 1) = 9.998898e-01;
    R_imu2velo(1, 2) = -1.482298e-02;
    R_imu2velo(2, 0) = 2.024406e-03;
    R_imu2velo(2, 1) = 1.482454e-02;
    R_imu2velo(2, 2) = 9.998881e-01;

    if (pcs.size() < 2)
    {
        std::cout << "Too few point cloud frames" << std::endl;
    }

    if (config["down_sample"].as<bool>())
    {
        // Down Sample
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
        for (int i = 0; i < pcs.size(); ++i)
        {
            pcl::VoxelGrid<pcl::PointXYZI> pc_ds;
            pc_ds.setInputCloud(pcs[i]);
            pc_ds.setLeafSize(0.2f, 0.2f, 0.2f);
            pc_ds.filter(*pc_downsampled);
            pcl::copyPointCloud(*pc_downsampled, *pcs[i]);
        }
    }
           if (config["merge_frame"].as<bool>())
       {
            #define merged_frames 5
           // Merge 3 frames point cloud
           for (int i = 0; i < pcs.size() - merged_frames - 1; ++i)
           {
               Eigen::Matrix4f T_velo_delt = Eigen::Matrix4f::Identity();
               // for(int j = i+1; j <= i; ++j)
               for (int j = i + 1; j <= i + merged_frames - 1; ++j)
               {
    
                   pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
                   ndt.setMaximumIterations(35);
                   ndt.setTransformationEpsilon(0.1);
                   ndt.setStepSize(0.1);
                   ndt.setResolution(0.5);
                   ndt.setInputSource(pcs[j]);
                   ndt.setInputTarget(pcs[i]);
                   pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_result_point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                   ndt.align(*ndt_result_point_cloud_ptr, T_velo_delt);
                   T_velo_delt = ndt.getFinalTransformation();



                   //pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                   //pcl::transformPointCloud(*pc_2, *output_cloud, T);
    
                   //std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs
                   for (int m = 0; m < pcs[j]->points.size(); ++m)
                   {
                       Eigen::Vector4f point_curr_frame(pcs[j]->points[m].x, pcs[j]->points[m].y, pcs[j]->points[m].z, 1);
                       Eigen::Vector4f point_last_frame = T_velo_delt * point_curr_frame;
                       // std::cout<<"point_curr_frame " << std::endl << std::setprecision(14) << std::fixed << point_curr_frame << std::endl;
                       // std::cout<<"point_last_frame " << std::endl << std::setprecision(14) << std::fixed << point_last_frame << std::endl;
                       pcl::PointXYZI point_temp;
                       point_temp.x = point_last_frame(0);
                       point_temp.y = point_last_frame(1);
                       point_temp.z = point_last_frame(2);
                       pcs[i]->push_back(point_temp);
                   }
               }
           }
    

       }
    //    if (config["merge_frame"].as<bool>())
    //    {
    //#define merged_frames 5
    //        // Merge 3 frames point cloud
    //        for (int i = 0; i < pcs.size() - merged_frames - 1; ++i)
    //        {
    //            Eigen::Matrix4f T_velo_delt = Eigen::Matrix4f::Identity();
    //            // for(int j = i+1; j <= i; ++j)
    //            for (int j = i + 1; j <= i + merged_frames - 1; ++j)
    //            {
    //
    //                pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    //                ndt.setMaximumIterations(30);
    //                ndt.setTransformationEpsilon(1e-3);
    //                ndt.setStepSize(0.01);
    //                ndt.setResolution(1.0);
    //                ndt.setInputSource(pcs[i]);
    //                ndt.setInputTarget(pcs[j]);
    //                pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_result_point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    //                ndt.align(*ndt_result_point_cloud_ptr, T_velo_delt);
    //                T_velo_delt = ndt.getFinalTransformation();
    //
    //
    //                //std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs
    //                for (int m = 0; m < pcs[j]->points.size(); ++m)
    //                {
    //                    Eigen::Vector4d point_curr_frame(pcs[j]->points[m].x, pcs[j]->points[m].y, pcs[j]->points[m].z, 1);
    //                    Eigen::Vector4d point_last_frame = T_velo_delt * point_curr_frame;
    //                    // std::cout<<"point_curr_frame " << std::endl << std::setprecision(14) << std::fixed << point_curr_frame << std::endl;
    //                    // std::cout<<"point_last_frame " << std::endl << std::setprecision(14) << std::fixed << point_last_frame << std::endl;
    //                    pcl::PointXYZI point_temp;
    //                    point_temp.x = point_last_frame(0);
    //                    point_temp.y = point_last_frame(1);
    //                    point_temp.z = point_last_frame(2);
    //                    pcs[i]->push_back(point_temp);
    //                }
    //            }
    //        }
    //
    //        // //显示提取效果
    //        // pcl::visualization::PCLVisualizer viewer("frame12_feature Viewer");
    //        // //设置窗口背景颜色，范围为0-1
    //        // viewer.setBackgroundColor(0, 0, 0);
    //        // //添加坐标轴
    //        // viewer.addCoordinateSystem(1);
    //        // //根据点云里某个字段大小设置颜色
    //        // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pcs[0], "x");
    //        // //往窗口添加点云并设置颜色
    //        // viewer.addPointCloud(pcs[0], fildColor, "cloud");
    //        // //添加点云后，通过点云ID来设置显示大小
    //        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
    //        // //重置相机，将点云显示到窗口
    //        // viewer.resetCamera();
    //        // while (!viewer.wasStopped())
    //        // {
    //        //     viewer.spinOnce();
    //        // }
    //    }

//     if (config["merge_frame"].as<bool>())
//     {
// #define merged_frames 5
//         // Merge 3 frames point cloud
//         for (int i = 0; i < pcs.size() - merged_frames - 1; ++i)
//         {
//             // for(int j = i+1; j <= i; ++j)
//             for (int j = i + 1; j <= i + merged_frames - 1; ++j)
//             {
//                 // Current and last eular angle from oxts file. (yaw pitch roll)
//                 Eigen::Vector3d oxts_eular_curr(oxts_vec[j * 30 + 5], oxts_vec[j * 30 + 4], oxts_vec[j * 30 + 3]);
//                 Eigen::Vector3d oxts_eular_last(oxts_vec[i * 30 + 5], oxts_vec[i * 30 + 4], oxts_vec[i * 30 + 3]);

//                 // // Current and last position from oxts file.
//                 // Eigen::Vector3d oxts_posi_curr(int(oxts_vec[ppp*30])*m_deg + int((oxts_vec[ppp*30]-int(oxts_vec[ppp*30]))*60)*m_min
//                 // + ((oxts_vec[ppp*30]-int(oxts_vec[ppp*30]))*60-int((oxts_vec[ppp*30]-int(oxts_vec[ppp*30]))*60))*60*m_sec,
//                 // int(oxts_vec[ppp*30+1])*m_deg + int((oxts_vec[ppp*30+1]-int(oxts_vec[ppp*30+1]))*60)*m_min
//                 // + ((oxts_vec[ppp*30+1]-int(oxts_vec[ppp*30+1]))*60-int((oxts_vec[ppp*30+1]-int(oxts_vec[ppp*30+1]))*60))*60*m_sec,
//                 // oxts_vec[ppp*30+2]);

//                 // Eigen::Vector3d oxts_posi_last(int(oxts_vec[0])*m_deg + int((oxts_vec[0]-int(oxts_vec[0]))*60)*m_min
//                 // + ((oxts_vec[0]-int(oxts_vec[0]))*60-int((oxts_vec[0]-int(oxts_vec[0]))*60))*60*m_sec,
//                 // int(oxts_vec[1])*m_deg + int((oxts_vec[1]-int(oxts_vec[1]))*60)*m_min
//                 // + ((oxts_vec[1]-int(oxts_vec[1]))*60-int((oxts_vec[1]-int(oxts_vec[1]))*60))*60*m_sec,
//                 // oxts_vec[2]);
//                 // std::cout<<"curr posi 1 = " << std::endl << std::setprecision(14) << std::fixed << oxts_posi_curr << std::endl;
//                 // std::cout<<"last posi 1 = " << std::endl << std::setprecision(14) << std::fixed << oxts_posi_last << std::endl;

//                 Eigen::Vector3d oxts_posi_curr, oxts_posi_last;
//                 double pi = 3.14159265358;
//                 double scale = std::cos(oxts_vec[j * 30] * pi / 180.0);
//                 double er = 6378137.0;
//                 double mx_curr = scale * oxts_vec[j * 30 + 1] * pi * er / 180.0;
//                 double my_curr = scale * er * std::log(std::tan((90.0 + oxts_vec[j * 30]) * pi / 360.0));
//                 double mx_last = scale * oxts_vec[i * 30 + 1] * pi * er / 180.0;
//                 double my_last = scale * er * std::log(std::tan((90.0 + oxts_vec[i * 30 + 0]) * pi / 360.0));

//                 oxts_posi_curr << mx_curr, my_curr, oxts_vec[j * 30 + 2];
//                 oxts_posi_last << mx_last, my_last, oxts_vec[i * 30 + 2];
//                 // std::cout<<"curr posi 2 = " << std::endl << std::setprecision(14) << std::fixed << oxts_posi_curr << std::endl;
//                 // std::cout<<"last posi 2 = " << std::endl << std::setprecision(14) << std::fixed << oxts_posi_last << std::endl;

//                 // Convert rotation from eular form to matrix form
//                 Eigen::Matrix3d R_imu_curr, R_imu_last, R_imu_last_counter, R_imu_curr_counter;
//                 // Eigen::Matrix3d R_imu_curr_x, R_imu_curr_y, R_imu_curr_z, R_imu_last_x, R_imu_last_y, R_imu_last_z;
//                 // double rx_curr = oxts_eular_curr[2], ry_curr = oxts_eular_curr[1], rz_curr = oxts_eular_curr[0];
//                 // double rx_last = oxts_eular_curr[2], ry_last = oxts_eular_curr[1], rz_last = oxts_eular_curr[0];
//                 // R_imu_curr_x << 1, 0, 0, 0, cos(rx_curr), -sin(rx_curr), 0, sin(rx_curr), cos(rx_curr);
//                 // R_imu_curr_y << cos(ry_curr), 0, sin(ry_curr), 0, 1, 0, -sin(ry_curr), 0, cos(ry_curr);
//                 // R_imu_curr_z << cos(rz_curr), -sin(rz_curr), 0, sin(rz_curr), cos(rz_curr), 0, 0, 0, 1;
//                 // R_imu_curr = R_imu_curr_z * R_imu_curr_y * R_imu_curr_x;
//                 // R_imu_last_x << 1, 0, 0, 0, cos(rx_last), -sin(rx_last), 0, sin(rx_last), cos(rx_last);
//                 // R_imu_last_y << cos(ry_last), 0, sin(ry_last), 0, 1, 0, -sin(ry_last), 0, cos(ry_last);
//                 // R_imu_last_z << cos(rz_last), -sin(rz_last), 0, sin(rz_last), cos(rz_last), 0, 0, 0, 1;
//                 // R_imu_last = R_imu_last_z * R_imu_last_y * R_imu_last_x;
//                 // std::cout << "last1 = " << std::endl << R_imu_last << std::endl;
//                 // std::cout << "curr1 = " << std::endl << R_imu_curr << std::endl;

//                 R_imu_last = Eigen::AngleAxisd(oxts_eular_last[2], Eigen::Vector3d::UnitX()) *
//                              Eigen::AngleAxisd(oxts_eular_last[1], Eigen::Vector3d::UnitY()) *
//                              Eigen::AngleAxisd(oxts_eular_last[0], Eigen::Vector3d::UnitZ());
//                 R_imu_curr = Eigen::AngleAxisd(oxts_eular_curr[2], Eigen::Vector3d::UnitX()) *
//                              Eigen::AngleAxisd(oxts_eular_curr[1], Eigen::Vector3d::UnitY()) *
//                              Eigen::AngleAxisd(oxts_eular_curr[0], Eigen::Vector3d::UnitZ());
//                 // std::cout << "last2 = " << std::endl << R_imu_last << std::endl;
//                 // std::cout << "curr2 = " << std::endl << R_imu_curr << std::endl;

//                 // R_imu_last_counter = Eigen::AngleAxisd(oxts_eular_last[2], Eigen::Vector3d::UnitX()) *
//                 //                     Eigen::AngleAxisd(oxts_eular_last[1], Eigen::Vector3d::UnitY()) *
//                 //                     Eigen::AngleAxisd(oxts_eular_last[0], Eigen::Vector3d::UnitZ());
//                 // R_imu_curr_counter = Eigen::AngleAxisd(oxts_eular_curr[2], Eigen::Vector3d::UnitX()) *
//                 //                 Eigen::AngleAxisd(oxts_eular_curr[1], Eigen::Vector3d::UnitY()) *
//                 //                 Eigen::AngleAxisd(oxts_eular_curr[0], Eigen::Vector3d::UnitZ());

//                 // Eigen::Vector3d oxts_posi_corrected_curr = R_imu2velo * oxts_posi_curr;
//                 // Eigen::Vector3d oxts_posi_corrected_last = R_imu2velo * oxts_posi_last;
//                 // std::cout << "delt1 = " << oxts_posi_corrected_curr - oxts_posi_corrected_last << std::endl;
//                 // std::cout << "delt2 = " << oxts_posi_curr - oxts_posi_last << std::endl;

//                 Eigen::Matrix4d T_imu_curr, T_imu_last;

//                 T_imu_curr.block(0, 0, 3, 3) = R_imu_curr;
//                 T_imu_curr.block(0, 3, 3, 1) = oxts_posi_curr;
//                 T_imu_curr.block(3, 0, 1, 4) << 0, 0, 0, 1;

//                 T_imu_last.block(0, 0, 3, 3) = R_imu_last;
//                 T_imu_last.block(0, 3, 3, 1) = oxts_posi_last;
//                 T_imu_last.block(3, 0, 1, 4) << 0, 0, 0, 1;
//                 Eigen::Matrix4d T_imu_delt = T_imu_last.inverse() * T_imu_curr;
//                 // std::cout<<"T_imu_delt = " << std::endl <<T_imu_delt << std::endl;

//                 // // Eigen::Matrix4d T_imu_delt = T_imu_last.inverse() * T_imu_curr;
//                 // // Eigen::Matrix3d R_imu_delt = R_imu_last.inverse() * R_imu_curr; //IMU
//                 // Eigen::Vector3d t_imu_delt = oxts_posi_curr - oxts_posi_last; //IMU
//                 // std::cout<<"test = " << std::endl << R_imu_curr * t_imu_delt << std::endl;
//                 // Eigen::Vector3d rpy = R_imu_curr.eulerAngles(0, 1, 2);
//                 // std::cout << "rpy = " << rpy << std::endl;
//                 // Eigen::Matrix4d T_imu_delt; // IMU new to last delt
//                 // T_imu_delt(0,0) = R_imu_delt(0,0);        T_imu_delt(0,1) = R_imu_delt(0,1);        T_imu_delt(0,2) = R_imu_delt(0,2);
//                 // T_imu_delt(1,0) = R_imu_delt(1,0);        T_imu_delt(1,1) = R_imu_delt(1,1);        T_imu_delt(1,2) = R_imu_delt(1,2);
//                 // T_imu_delt(2,0) = R_imu_delt(2,0);        T_imu_delt(2,1) = R_imu_delt(2,1);        T_imu_delt(2,2) = R_imu_delt(2,2);
//                 // T_imu_delt(0,3) = t_imu_delt(0);          T_imu_delt(1,3) = t_imu_delt(1);          T_imu_delt(2,3) = t_imu_delt(2);    T_imu_delt(3,3) = 1;
//                 // std::cout<<"T_imu_delt1 = " << std::endl <<T_imu_delt << std::endl;

//                 // std::cout << "T_imu_last = " << std::endl << T_imu_last << std::endl << "T_imu_last inverse = " << std::endl << T_imu_last.inverse() << std::endl;
//                 // std::cout << "T_imu_curr = " << std::endl << T_imu_curr << std::endl;

//                 // Eigen::Matrix3d R_velo_delt = (R_imu2velo * R_imu_delt * R_imu2velo.inverse()).inverse();
//                 // Eigen::Vector3d t_vec_velo_delt =
//                 Eigen::Matrix4d T_velo_delt = T_imu2velo * T_imu_last.inverse() * T_imu_curr * T_imu2velo.inverse();
//                 // Eigen::Matrix4d T_velo = T_imu2velo * T_imu_last.inverse() * T_imu_curr * T_imu2velo.inverse();

//                 // std::cout<<"R_imu_curr " << std::endl << R_imu_curr <<std::endl;
//                 // std::cout<<"R_imu_last " << std::endl << R_imu_last <<std::endl;
//                 // std::cout<<"R_imu_last inverse " << std::endl << R_imu_last.inverse() <<std::endl;
//                 // std::cout<<"tran3_delt " << std::endl << tran3_delt <<std::endl;
//                 // std::cout<<"vec_delt " << std::endl << vec_delt <<std::endl;

//                 // std::cout<<"T_imu_curr " << std::endl << T_imu_curr <<std::endl;
//                 // std::cout<<"T_imu_last " << std::endl << T_imu_last <<std::endl;
//                 // std::cout<<"T_imu_last inverse " << std::endl << T_imu_last.inverse() <<std::endl;
//                 // std::cout<<"T_imu_delt " << std::endl << T_imu_delt <<std::endl;
//                 // std::cout<<"T_velo_delt " << std::endl << T_velo_delt <<std::endl;
//                 // std::cout<<"T_velo " << std::endl << T_velo <<std::endl;

//                 //std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs
//                 for (int m = 0; m < pcs[j]->points.size(); ++m)
//                 {
//                     Eigen::Vector4d point_curr_frame(pcs[j]->points[m].x, pcs[j]->points[m].y, pcs[j]->points[m].z, 1);
//                     Eigen::Vector4d point_last_frame = T_velo_delt * point_curr_frame;
//                     // std::cout<<"point_curr_frame " << std::endl << std::setprecision(14) << std::fixed << point_curr_frame << std::endl;
//                     // std::cout<<"point_last_frame " << std::endl << std::setprecision(14) << std::fixed << point_last_frame << std::endl;
//                     pcl::PointXYZI point_temp;
//                     point_temp.x = point_last_frame(0);
//                     point_temp.y = point_last_frame(1);
//                     point_temp.z = point_last_frame(2);
//                     pcs[i]->push_back(point_temp);
//                 }
//             }
//         }

//         // //显示提取效果
//         // pcl::visualization::PCLVisualizer viewer("frame12_feature Viewer");
//         // //设置窗口背景颜色，范围为0-1
//         // viewer.setBackgroundColor(0, 0, 0);
//         // //添加坐标轴
//         // viewer.addCoordinateSystem(1);
//         // //根据点云里某个字段大小设置颜色
//         // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pcs[0], "x");
//         // //往窗口添加点云并设置颜色
//         // viewer.addPointCloud(pcs[0], fildColor, "cloud");
//         // //添加点云后，通过点云ID来设置显示大小
//         // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
//         // //重置相机，将点云显示到窗口
//         // viewer.resetCamera();
//         // while (!viewer.wasStopped())
//         // {
//         //     viewer.spinOnce();
//         // }
//     }

    if (config["save_pointcloud"].as<bool>())
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr frame12_feature(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr frame12_no_ground(new pcl::PointCloud<pcl::PointXYZI>);
        point_cb(frame12, frame12_no_ground);
        extract_pc_feature_6(frame12_no_ground, frame12_feature);
        std::string output_pc_name = "../data" + std::to_string(frame_cnt) + "/pc";
        std::ofstream out(output_pc_name, std::ios::app);
        if (out)
        {
            for (size_t i = 0; i < frame12_no_ground->size(); ++i)
            {
                out << frame12_no_ground->points[i].x << " " << frame12_no_ground->points[i].y << " " << frame12_no_ground->points[i].z << std::endl;
            }
        }
        else
        {
            std::cout << "save error pc" << std::endl;
        }
        out.close();
    }

    extractFeature(pcs, images, pc_features, distance_images, distance_images_thin);
    in_pcs_feature = pc_features;
    in_images_feature = distance_images_thin;
    std::cout << "out extract feature" << std::endl;

    // 显示提取效果
    //pcl::visualization::PCLVisualizer viewer("frame12_feature Viewer");
    //    设置窗口背景颜色，范围为0-1
   //viewer.setBackgroundColor(0, 0, 0);
    //     添加坐标轴
//viewer.addCoordinateSystem(1);
    //     根据点云里某个字段大小设置颜色
    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(pcs[0], "x");
    //     往窗口添加点云并设置颜色
    //viewer.addPointCloud(pcs[0], fildColor, "cloud");
    //     添加点云后，通过点云ID来设置显示大小
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
    //     重置相机，将点云显示到窗口
   //viewer.resetCamera();
     //while (!viewer.wasStopped())
     //{
     //    viewer.spinOnce();
    // }

    float bias_x, bias_y, bias_z;
    bias_x = config["bias_x"].as<float>();
    bias_y = config["bias_y"].as<float>();
    bias_z = config["bias_z"].as<float>();
    Eigen::AngleAxisf r_vx(M_PI * bias_x / 180, Eigen::Vector3f(1, 0, 0)); //手动添加的偏差
    Eigen::AngleAxisf r_vy(M_PI * bias_y / 180, Eigen::Vector3f(0, 1, 0));
    Eigen::AngleAxisf r_vz(M_PI * bias_z / 180, Eigen::Vector3f(0, 0, 1));
    Eigen::Matrix3f R_lidar2cam0_unbias = Eigen::Matrix3f::Identity();
    //R_lidar2cam0_unbias(0, 0) = 0.007533745;
    //R_lidar2cam0_unbias(0, 1) = -0.9999714;
    //R_lidar2cam0_unbias(0, 2) = -0.000616602;
    //R_lidar2cam0_unbias(1, 0) = 0.01480249;
    //R_lidar2cam0_unbias(1, 1) = 0.0007280733;
    //R_lidar2cam0_unbias(1, 2) = -0.9998902;
    //R_lidar2cam0_unbias(2, 0) = 0.9998621;
    //R_lidar2cam0_unbias(2, 1) = 0.00752379;
    //R_lidar2cam0_unbias(2, 2) = 0.01480755;

    std::vector<float> ext = config["R_lidar2cam0_unbias"]["data"].as<std::vector<float>>();
    assert((int)ext.size() == 9);
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            R_lidar2cam0_unbias(row, col) = ext[row * 3 + col];
        }
    }

    // //*****TEST******
    // std::cout << "*****TEST START*****" << std::endl;
    // Eigen::AngleAxisf ang_axis_x( 90.0 * M_PI / 180, Eigen::Vector3f(1, 0, 0));
    // Eigen::AngleAxisf ang_axis_y( -90.0 * M_PI / 180, Eigen::Vector3f(0, 1, 0));
    // Eigen::AngleAxisf ang_axis_z( 90.0 * M_PI / 180, Eigen::Vector3f(0, 0, 1));
    // Eigen::Matrix3f mat = Eigen::Matrix3f::Identity();

    // mat(0,0) = 1.0;
    // mat(0,1) = 0.0;
    // mat(0,2) = 0.0;
    // mat(1,0) = 0.0;
    // mat(1,1) = 1.0;
    // mat(1,2) = 0.0;
    // mat(2,0) = 0.0;
    // mat(2,1) = 0.0;
    // mat(2,2) = 1.0;
    // mat(0,0) = 1.0;
    // mat(0,1) = 0.0;
    // mat(0,2) = 0.0;
    // mat(1,0) = 0.0;
    // mat(1,1) = 0.707;
    // mat(1,2) = -0.707;
    // mat(2,0) = 0.0;
    // mat(2,1) = 0.707;
    // mat(2,2) = 0.707;
    // Eigen::Vector3f euler_ang_before0 = mat.eulerAngles(0, 1, 2);//x y z
    // Eigen::Vector3f euler_ang_before1 = mat.eulerAngles(2, 1, 0);//z y x
    // std::cout << "mat euler_ang_before0_xyz = " << std::endl << euler_ang_before0 << std::endl;
    // std::cout << "mat euler_ang_before0_zyx = " << std::endl << euler_ang_before1 << std::endl;
    // mat = ang_axis_x.matrix() * ang_axis_y.matrix() * ang_axis_z.matrix() * mat;// 以固定坐标轴旋转
    // Eigen::Vector3f euler_ang_after0_xyz = mat.eulerAngles(0, 1, 2);//x y z // 以自身坐标轴旋转
    // Eigen::Vector3f euler_ang_after0_zyx = mat.eulerAngles(2, 1, 0);//z y x
    // std::cout << "mat euler_ang_after0_xyz = " << std::endl << euler_ang_after0_xyz * 180.0 / M_PI << std::endl;
    // std::cout << "mat euler_ang_after0_zyx = " << std::endl << euler_ang_after0_zyx * 180.0 / M_PI << std::endl;

    // mat(0,0) = 0.007533745; // velodyne to camera
    // mat(0,1) = -0.9999714;
    // mat(0,2) = -0.000616602;
    // mat(1,0) = 0.01480249;
    // mat(1,1) = 0.0007280733;
    // mat(1,2) = -0.9998902;
    // mat(2,0) = 0.9998621;
    // mat(2,1) = 0.00752379;
    // mat(2,2) = 0.01480755;
    // Eigen::Vector3f euler_ang_before1 = mat.eulerAngles(0, 1, 2);//x y z
    // // std::cout << "mat euler_ang_before1_xyz = " << std::endl << euler_ang_before1 * 180.0 / M_PI << std::endl;
    // mat = ang_axis_x.matrix() * ang_axis_y.matrix() * ang_axis_z.matrix() * mat;
    // Eigen::Vector3f euler_ang_after1_xyz = mat.eulerAngles(0, 1, 2);//x y z
    // Eigen::Vector3f euler_ang_after1_zyx = mat.eulerAngles(2, 1, 0);//z y x
    // // std::cout << "mat euler_ang_after1_xyz = " << std::endl << euler_ang_after1_xyz  *180.0 / M_PI << std::endl;
    // // std::cout << "mat euler_ang_after1_zyx = " << std::endl << euler_ang_after1_zyx  *180.0 / M_PI << std::endl;
    // std::cout << "*****TEST END*****" << std::endl;

    Eigen::Vector3f eulerAngle1 = R_lidar2cam0_unbias.eulerAngles(0, 1, 2);
    // std::cout<<"eulerAngle before  = "<<(180.0/3.14159*eulerAngle1)<<std::endl;
    Eigen::Quaternionf quaternion_before(R_lidar2cam0_unbias);
    // std::cout<<"quat before = "<<quaternion_before.w()<<" "<<quaternion_before.x()<<" "<<quaternion_before.y()<<" "<<quaternion_before.z()<<std::endl;

    //T_lidar2cam0_unbias lidar2cam0
    T_lidar2cam0_unbias(0, 0) = R_lidar2cam0_unbias(0, 0);
    T_lidar2cam0_unbias(0, 1) = R_lidar2cam0_unbias(0, 1);
    T_lidar2cam0_unbias(0, 2) = R_lidar2cam0_unbias(0, 2);
    T_lidar2cam0_unbias(1, 0) = R_lidar2cam0_unbias(1, 0);
    T_lidar2cam0_unbias(1, 1) = R_lidar2cam0_unbias(1, 1);
    T_lidar2cam0_unbias(1, 2) = R_lidar2cam0_unbias(1, 2);
    T_lidar2cam0_unbias(2, 0) = R_lidar2cam0_unbias(2, 0);
    T_lidar2cam0_unbias(2, 1) = R_lidar2cam0_unbias(2, 1);
    T_lidar2cam0_unbias(2, 2) = R_lidar2cam0_unbias(2, 2);
    T_lidar2cam0_unbias(0, 3) = config["t03"].as<float>();
    T_lidar2cam0_unbias(1, 3) = config["t13"].as<float>();
    T_lidar2cam0_unbias(2, 3) = config["t23"].as<float>();
    T_lidar2cam0_unbias(3, 0) = 0;
    T_lidar2cam0_unbias(3, 1) = 0;
    T_lidar2cam0_unbias(3, 2) = 0;
    T_lidar2cam0_unbias(3, 3) = 1;
    // Eigen::Matrix4f T_lidar2cam0_unbias = T_lidar2cam0_unbias;

    Eigen::Matrix3f R_lidar2cam0_bias = r_vx.matrix() * r_vy.matrix() * r_vz.matrix() * R_lidar2cam0_unbias; //手动添加误差
    Eigen::Vector3f eulerAngle2 = R_lidar2cam0_bias.eulerAngles(0, 1, 2);
    // std::cout<<"eulerAngle after = "<<(180.0/3.14159*eulerAngle2)<<std::endl;
    // std::cout<<"r_vx Angle = "<<(180.0/3.14159*(r_vx.matrix() * r_vy.matrix() * r_vz.matrix()).eulerAngles(0,1,2))<<std::endl;
    Eigen::Quaternionf quaternion_after(R_lidar2cam0_bias);
    // std::cout<<"quat after = "<<quaternion_after.w()<<" "<<quaternion_after.x()<<" "<<quaternion_after.y()<<" "<<quaternion_after.z()<<std::endl;

    T_lidar2cam0_bias(0, 0) = R_lidar2cam0_bias(0, 0);
    T_lidar2cam0_bias(0, 1) = R_lidar2cam0_bias(0, 1);
    T_lidar2cam0_bias(0, 2) = R_lidar2cam0_bias(0, 2);
    T_lidar2cam0_bias(1, 0) = R_lidar2cam0_bias(1, 0);
    T_lidar2cam0_bias(1, 1) = R_lidar2cam0_bias(1, 1);
    T_lidar2cam0_bias(1, 2) = R_lidar2cam0_bias(1, 2);
    T_lidar2cam0_bias(2, 0) = R_lidar2cam0_bias(2, 0);
    T_lidar2cam0_bias(2, 1) = R_lidar2cam0_bias(2, 1);
    T_lidar2cam0_bias(2, 2) = R_lidar2cam0_bias(2, 2);
    T_lidar2cam0_bias(0, 3) = config["t03"].as<float>();
    T_lidar2cam0_bias(1, 3) = config["t13"].as<float>();
    T_lidar2cam0_bias(2, 3) = config["t23"].as<float>();
    T_lidar2cam0_bias(3, 0) = 0;
    T_lidar2cam0_bias(3, 1) = 0;
    T_lidar2cam0_bias(3, 2) = 0;
    T_lidar2cam0_bias(3, 3) = 1;

    //相机0到相机2
    //T_cam02cam2(0, 0) = 0.9999758;
    //T_cam02cam2(0, 1) = -0.005267463;
    //T_cam02cam2(0, 2) = -0.004552439;
    //T_cam02cam2(0, 3) = 0.05956621;
    //T_cam02cam2(1, 0) = 0.005251945;
    //T_cam02cam2(1, 1) = 0.9999804;
    //T_cam02cam2(1, 2) = -0.003413835;
    //T_cam02cam2(1, 3) = 0.0002900141;
    //T_cam02cam2(2, 0) = 0.004570332;
    //T_cam02cam2(2, 1) = 0.003389843;
    //T_cam02cam2(2, 2) = 0.9999838;
    //T_cam02cam2(2, 3) = 0.002577209;
    //T_cam02cam2(3, 0) = 0;
    //T_cam02cam2(3, 1) = 0;
    //T_cam02cam2(3, 2) = 0;
    //T_cam02cam2(3, 3) = 1;

    // T_cam02cam2(0, 0) = 1;
    // T_cam02cam2(0, 1) = 0;
    // T_cam02cam2(0, 2) = 0;
    // T_cam02cam2(0, 3) = 0;
    // T_cam02cam2(1, 0) = 0;
    // T_cam02cam2(1, 1) = 1;
    // T_cam02cam2(1, 2) = 0;
    // T_cam02cam2(1, 3) = 0;
    // T_cam02cam2(2, 0) = 0;
    // T_cam02cam2(2, 1) = 0;
    // T_cam02cam2(2, 2) = 1;
    // T_cam02cam2(2, 3) = 0;
    // T_cam02cam2(3, 0) = 0;
    // T_cam02cam2(3, 1) = 0;
    // T_cam02cam2(3, 2) = 0;
    // T_cam02cam2(3, 3) = 1;
    std::vector<float> cam02cam2 = config["T_cam02cam2"]["data"].as<std::vector<float>>();
    assert((int)cam02cam2.size() == 16);
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            T_cam02cam2(row, col) = cam02cam2[row * 4 + col];
        }
    }

    T_lidar2cam2_bias = T_cam02cam2 * T_lidar2cam0_bias;
    T_lidar2cam2_unbias = T_cam02cam2 * T_lidar2cam0_unbias;

    Eigen::Matrix3f R_lidar2cam2_bias = Eigen::Matrix3f::Identity();
    R_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias(0, 0);
    R_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias(0, 1);
    R_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias(0, 2);
    R_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias(1, 0);
    R_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias(1, 1);
    R_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias(1, 2);
    R_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias(2, 0);
    R_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias(2, 1);
    R_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias(2, 2);

    //T_cam2image cam2image

    // T_cam2image(0, 0) = 1.020379742184453e+03;
    // T_cam2image(0, 1) = 0.0;
    // T_cam2image(0, 2) = 9.320378452339318e+02;
    // T_cam2image(1, 0) = 0.0;
    // T_cam2image(1, 1) = 1.023240590876533e+03;
    // T_cam2image(1, 2) = 5.808559269188706e+02;
    // T_cam2image(2, 0) = 0;
    // T_cam2image(2, 1) = 0;
    // T_cam2image(2, 2) = 1;

    cv::Mat outimg;
    std::vector<cv::line_descriptor::KeyLine> keylines, keylines2;
    //extract_image_feature(images[0],images[1],keylines,keylines2,outimg);

    //ros::init(argc, argv, "auto_calib");
    ros::Time::init();
    // ros::NodeHandle nh;

    int neg = -1;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pc_features_local;
    std::vector<cv::Mat> distance_images_local, distance_images_local_thin;
    std::string calibrated_result_path = "../data" + std::to_string(frame_cnt) + "/result/calibrated_result";
    std::ofstream calibrated_result(calibrated_result_path, std::ios::app);
    Eigen::Matrix3f R_gt = T_lidar2cam2_unbias.block(0, 0, 3, 3);
    Eigen::Vector3f euler_ang_gt_zyx = R_gt.eulerAngles(0, 1, 2);
    std::vector<Eigen::Vector3f> euler_ang_calibrated_zyx_vec, euler_ang_inv_delt_zyx_vec, euler_ang_inv_delt_xyz_vec,
        euler_ang_delt_zyx_vec, euler_ang_delt_zyx_vec_mean, euler_ang_delt_xyz_vec;
    std::vector<int> large_small_step;
    calibrated_result << "cnt=[";
    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {
        calibrated_result << ppp << ",";
    }
    calibrated_result << "]" << std::endl;
    calibrated_result << "x_g=[";
    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {
        calibrated_result << euler_ang_gt_zyx[2] * 180.0 / M_PI << ",";
    }
    calibrated_result << "]" << std::endl;
    calibrated_result << "y_g=[";
    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {
        calibrated_result << euler_ang_gt_zyx[1] * 180.0 / M_PI << ",";
    }
    calibrated_result << "]" << std::endl;
    calibrated_result << "z_g=[";
    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {
        calibrated_result << euler_ang_gt_zyx[0] * 180.0 / M_PI << ",";
    }
    calibrated_result << "]" << std::endl;

    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {
        pc_features_local.push_back(pc_features[ppp]);
        distance_images_local.push_back(distance_images[ppp]);
        distance_images_local_thin.push_back(distance_images_thin[ppp]);
    }

    // std::cout << "pcs.size() = " << pcs.size() << std::endl;

    float iterate_ang_step_big = 0.8; //0.06
    float iterate_tra_step_big = 0;   //0.002; //0.002

    float iterate_ang_step_small = 0.01; //0.01
    float iterate_tra_step_small = 0;    //0.001; //0.001
    bool got_fine_result = false;
    std::vector<float> confidence_vec;
    //Eigen::Matrix4f T_lidar2cam2_bias_temp = T_lidar2cam2_bias;
    //Kitti
    for (int ppp = 0; ppp < pcs.size(); ppp++)
    {

        // cv::Mat image_orig;
        // project2image(pcs[ppp], images[ppp], image_orig, T_lidar2cam2_unbias, T_cam2image);
        cv::Mat image_before_optimize;
        cv::Mat image_unbias;

        // cv::namedWindow("cvt", CV_WINDOW_NORMAL);
        // cv::imshow("cvt", image_orig);
        // cv::waitKey(0);

        if (ppp % 10 == 0 && ppp != 0)
        {
            int rand_num = rand() % 20;
            if (rand_num == 1)
            {
                neg = 1;
            }
            else
            {
                neg = -1;
            }
            float rand_float = (rand_num - 10) / 10.0;

            std::cout << "Add Bias" << std::endl;
            float bias_x_temp, bias_y_temp, bias_z_temp;
            bias_x_temp = rand_float; // config["bias_x"].as<float>()/2.0*neg;
            bias_y_temp = rand_float; //config["bias_y"].as<float>()/2.0*neg;
            std::cout << "bias_y_temp = " << bias_y_temp << std::endl;
            bias_z_temp = rand_float;                                                         //config["bias_z"].as<float>()/2.0*neg;
            Eigen::AngleAxisf rot_x_temp(M_PI * bias_x_temp / 180, Eigen::Vector3f(1, 0, 0)); //手动添加的偏差
            Eigen::AngleAxisf rot_y_temp(M_PI * bias_y_temp / 180, Eigen::Vector3f(0, 1, 0));
            Eigen::AngleAxisf rot_z_temp(M_PI * bias_z_temp / 180, Eigen::Vector3f(0, 0, 1));
            Eigen::Matrix3f rot_temp = Eigen::Matrix3f::Identity();

            // Utlize last result to calibrate
            // rot_temp(0, 0) = T_lidar2cam2_bias(0,0);
            // rot_temp(0, 1) = T_lidar2cam2_bias(0,1);
            // rot_temp(0, 2) = T_lidar2cam2_bias(0,2);
            // rot_temp(1, 0) = T_lidar2cam2_bias(1,0);
            // rot_temp(1, 1) = T_lidar2cam2_bias(1,1);
            // rot_temp(1, 2) = T_lidar2cam2_bias(1,2);
            // rot_temp(2, 0) = T_lidar2cam2_bias(2,0);
            // rot_temp(2, 1) = T_lidar2cam2_bias(2,1);
            // rot_temp(2, 2) = T_lidar2cam2_bias(2,2);
            // Eigen::Matrix3f rot_temp_bias = Eigen::Matrix3f::Identity();
            // rot_temp_bias = rot_x_temp.matrix() * rot_y_temp.matrix() * rot_z_temp.matrix() * rot_temp;
            // T_lidar2cam2_bias(0,0) = rot_temp_bias(0, 0);
            // T_lidar2cam2_bias(0,1) = rot_temp_bias(0, 1);
            // T_lidar2cam2_bias(0,2) = rot_temp_bias(0, 2);
            // T_lidar2cam2_bias(1,0) = rot_temp_bias(1, 0);
            // T_lidar2cam2_bias(1,1) = rot_temp_bias(1, 1);
            // T_lidar2cam2_bias(1,2) = rot_temp_bias(1, 2);
            // T_lidar2cam2_bias(2,0) = rot_temp_bias(2, 0);
            // T_lidar2cam2_bias(2,1) = rot_temp_bias(2, 1);
            // T_lidar2cam2_bias(2,2) = rot_temp_bias(2, 2);

            Eigen::Matrix4f T_lidar2cam2_bias_temp = T_lidar2cam2_unbias;

            rot_temp = T_lidar2cam2_bias_temp.block(0, 0, 3, 3);
            Eigen::Matrix3f rot_temp_bias = Eigen::Matrix3f::Identity();
            rot_temp_bias = rot_x_temp.matrix() * rot_y_temp.matrix() * rot_z_temp.matrix() * rot_temp;
            T_lidar2cam2_bias_temp.block(0, 0, 3, 3) = rot_temp_bias;
            T_lidar2cam2_bias = T_lidar2cam2_bias_temp;
        }

        project2image(pc_features_local[ppp], images[ppp], image_unbias, T_lidar2cam2_unbias, T_cam2image);
        
        cv::Mat image1;
        project2image(pc_features_local[ppp], distance_images_local[ppp], image1, T_lidar2cam2_bias, T_cam2image);
        cv::Mat image_bias;
        project2image(pc_features_local[ppp], images_withouthist[ppp], image_bias, T_lidar2cam2_bias, T_cam2image);

        // #define USE_ORB
#ifdef USE_ORB
        static std::vector<cv::Mat> image_rois, image_recs;
        cv::Mat image_roi(images[ppp].rows, images[ppp].cols, CV_8UC1, cv::Scalar::all(255));
        cv::Mat image_rec(images[ppp].rows, images[ppp].cols, CV_8UC3, cv::Scalar::all(0));
        // cv::imshow("ori", image_roi);

#define ROI_LEFT 0
#define ROI_TOP images[ppp].rows / 2
#define ROI_WIDTH images[ppp].cols
#define ROI_HEIGHT images[ppp].rows / 2

        cv::Rect rect(ROI_LEFT, ROI_TOP, ROI_WIDTH, ROI_HEIGHT);
        // cv::Rect rect(images[ppp].cols/3, images[ppp].rows/2, images[ppp].cols/3, images[ppp].rows/2);
        images[ppp](rect).copyTo(image_roi);

        image_rois.push_back(image_roi);
        image_roi.copyTo(image_rec(rect)); //原始未加偏差图像
        // image_roi.copyTo(image_rec(cv::Rect(images[ppp].cols/3, images[ppp].rows/2, images[ppp].cols/3, images[ppp].rows/2)));//原始未加偏差图像
        // image4.copyTo(dst(cv::Rect(0, image2.rows * 2, image2.cols, image2.rows)));//最终标定结果
        image_recs.push_back(image_rec);
        // cv::imshow("rect",image_rec);
        // cv::waitKey(0);

        Eigen::Matrix3f R_cam_del;
        if (ppp > 0)
        {
            std::vector<cv::KeyPoint> keypoints_first, keypoints_second;
            std::vector<cv::DMatch> matches;

            // find_feature_matches(image_recs[ppp-1], image_recs[ppp], keypoints_first, keypoints_second, matches);
            find_feature_matches(images[0], images[ppp], keypoints_first, keypoints_second, matches);
            std::cout << "一共找到 " << matches.size() << " 组匹配点" << std::endl;
            if (matches.size() < 9)
            {
                std::cout << "匹配点太少!!!!!!!!" << std::endl;
                continue;
            }
            cv::Mat img_goodmatch;
            cv::drawMatches(images[ppp - 1], keypoints_first, images[ppp], keypoints_second, matches, img_goodmatch);
            cv::namedWindow("ROI图片", CV_WINDOW_NORMAL);
            cv::imshow("ROI图片", images[ppp]);
            cv::namedWindow("匹配点对", CV_WINDOW_NORMAL);
            cv::imshow("匹配点对", img_goodmatch);
            cv::waitKey(0);
            cv::imwrite(result_file[ppp], img_goodmatch);

            // std::cout<<std::endl;
            // for(int i = 0; i < oxts_vec.size(); ++i)
            // {
            //     std::cout<<oxts_vec[i]<<" ";
            // }
            // std::cout<<std::endl;
            Eigen::Vector3f vec_oxts_now(oxts_vec[ppp * 30 + 5], oxts_vec[ppp * 30 + 4], oxts_vec[ppp * 30 + 3]);
            Eigen::Vector3f vec_oxts_last(oxts_vec[(ppp - 1) * 30 + 5], oxts_vec[(ppp - 1) * 30 + 4], oxts_vec[(ppp - 1) * 30 + 3]);
            // vec_oxts[0] = 1.5757;
            // vec_oxts[1] = 0;
            // vec_oxts[2] = 0;
            Eigen::Matrix3f rot_oxts_now, rot_oxts_last;
            // std::cout<<"oxts_vec.size = " << oxts_vec.size() <<std::endl;
            // std::cout<<"yaw pitch roll = " << vec_oxts_last[0] << " " << vec_oxts_last[1] << " " << vec_oxts_last[2] << std::endl;
            rot_oxts_last = Eigen::AngleAxisf(vec_oxts_last[0], Eigen::Vector3f::UnitZ()) *
                            Eigen::AngleAxisf(vec_oxts_last[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(vec_oxts_last[2], Eigen::Vector3f::UnitX());
            rot_oxts_now = Eigen::AngleAxisf(vec_oxts_now[0], Eigen::Vector3f::UnitZ()) *
                           Eigen::AngleAxisf(vec_oxts_now[1], Eigen::Vector3f::UnitY()) *
                           Eigen::AngleAxisf(vec_oxts_now[2], Eigen::Vector3f::UnitX());
            // std::cout << "last rotation matrix3 =\n" << rot_oxts_last << std::endl;
            // std::cout << "now rotation matrix3 =\n" << rot_oxts_now << std::endl;
            Eigen::Matrix3f rot_oxts_delt = rot_oxts_last.inverse() * rot_oxts_now;
            // std::cout << "Ground truth -- R is\n" << rot_oxts_delt << std::endl;
            Eigen::Vector3f euler_ang_zyx_oxts_delt = rot_oxts_delt.eulerAngles(2, 1, 0); //z y x
            // std::cout << "euler_ang_xyz_icp = " << std::endl << euler_ang_xyz_icp * 180.0 / M_PI << std::endl;
            // std::cout << "Ground truth -- euler_ang_zyx_oxts_delt is " << std::endl << euler_ang_zyx_oxts_delt * 180.0 / M_PI << std::endl;

            // // 为什么把函数里内容拿出来计算的本质矩阵和旋转矩阵结果不对？？？！！！！！！
            // // cv::Mat K = (cv::Mat_<double> (3,3) << 721.5377, 0, 609.5593, 0, 721.5377, 172.854, 0, 0, 1);
            // cv::Mat K = (cv::Mat_<double> (3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

            // std::vector<cv::Point2f> points1;
            // std::vector<cv::Point2f> points2;
            // for(int i = 0; i < (int)matches.size(); i++)
            // {
            //     points1.push_back(keypoints_first[matches[i].queryIdx].pt);
            //     points2.push_back(keypoints_second[matches[i].queryIdx].pt);
            // }
            // std::cout<<"matches.size() " << matches.size() << " " << points1.size()<< " " << points2.size()<<std::endl;

            // //-- 计算基础矩阵
            // cv::Mat fundamental_matrix;
            // fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
            // std::cout<<"fundamental_matrix is "<<std::endl<< fundamental_matrix<<endl;

            // // 计算本质矩阵
            // // cv::Point2d principal_point(609.5593, 172.854);
            // cv::Point2d principal_point(325.1, 249.7);
            // // int focal_length = 721.5377;
            // double focal_length = 521;
            // cv::Mat essential_matrix;
            // essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
            // std::cout<<"essential_matrix is "<<std::endl<<essential_matrix<<std::endl;

            // // 计算单应矩阵
            // cv::Mat homography_matrix;
            // homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);
            // // std::cout<<"homography matrix is "<<std::endl<<homography_matrix<<std::endl;

            // // 从本质矩阵中恢复旋转和平移信息
            // // cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
            // // cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
            // cv::Mat R,t;
            // // cv::Mat_<float> R = Mat_<float>
            // recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
            // Eigen::Matrix3f eigen_rot;// = Eigen::Matrix3f::Identity();
            // eigen_rot << R.at<double>(0, 0),R.at<double>(0, 1),R.at<double>(0, 2),R.at<double>(1, 0),
            //         R.at<double>(1, 1),R.at<double>(1, 2),R.at<double>(2, 0),
            //         R.at<double>(2, 1),R.at<double>(2, 2);
            // Eigen::Vector3f euler_ang_xyz = eigen_rot.eulerAngles(0, 1, 2);//x y z
            // Eigen::Vector3f euler_ang_zyx = eigen_rot.eulerAngles(2, 1, 0);//z y x
            // std::cout<<"eigen_rot is "<<std::endl<<eigen_rot<<std::endl;
            // std::cout<<"R is "<<std::endl<<R<<std::endl;
            // std::cout << "euler_ang_xyz = " << std::endl << euler_ang_xyz * 180.0 / M_PI << std::endl;
            // std::cout << "euler_ang_zyx = " << std::endl << euler_ang_zyx * 180.0 / M_PI << std::endl;

            cv::Mat R, t;
            pose_estimation_2d2d(keypoints_first, keypoints_second, matches, R, t);

            R_cam_del << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(2, 0),
                R.at<double>(2, 1), R.at<double>(2, 2);

            //pcl::PointCloud<pcl::PointXYZI>::Ptr
            pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);
            src = pcs[ppp - 1];
            tgt = pcs[ppp];
#define LEAF_SIZE 0.3
            down_sample_pc(src, src, LEAF_SIZE);
            down_sample_pc(tgt, tgt, LEAF_SIZE);

            pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setTransformationEpsilon(1e-10);
            icp.setEuclideanFitnessEpsilon(0.001);
            icp.setMaximumIterations(100);

            icp.setInputSource(src);
            icp.setInputTarget(tgt);
            icp.align(*output);

            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            std::cout << "ICP -- transform is " << std::endl
                      << transformation << std::endl;
            rot_icp = transformation.block(0, 0, 3, 3);

            Eigen::Vector3f euler_ang_xyz_icp = rot_icp.eulerAngles(0, 1, 2); //x y z
            Eigen::Vector3f euler_ang_zyx_icp = rot_icp.eulerAngles(2, 1, 0); //z y x
            // std::cout << "euler_ang_xyz_icp = " << std::endl << euler_ang_xyz_icp * 180.0 / M_PI << std::endl;
            // std::cout << "ICP -- euler_ang_zyx is " << std::endl << euler_ang_zyx_icp * 180.0 / M_PI << std::endl;

            // geometry_msgs::TransformStamped tf_trans;
            tf_trans.header.stamp = ros::Time::now();
            tf_trans.header.frame_id = "first";

            tf_trans.child_frame_id = "second";
            Eigen::Matrix3f rot_eigen = transformation.block(0, 0, 3, 3);
            Eigen::Quaternionf tf_quat_eigen(rot_eigen);

            tf_trans.transform.translation.x = transformation(0, 3);
            tf_trans.transform.translation.y = transformation(1, 3);
            tf_trans.transform.translation.z = transformation(2, 3);
            tf_trans.transform.rotation.x = tf_quat_eigen.x();
            tf_trans.transform.rotation.y = tf_quat_eigen.y();
            tf_trans.transform.rotation.z = tf_quat_eigen.z();
            tf_trans.transform.rotation.w = tf_quat_eigen.w();

            // tf::TransformBroadcaster tf_broadcaster;

            // std::thread tf_thread{send_transform_thread};
            // tf_thread.join();
            // // tf_broadcaster.sendTransform(tf_trans);
        }

#endif

        // continue;

        omp_lock_t mylock;
        omp_init_lock(&mylock);
        float starTime = omp_get_wtime();
        float tran_x, tran_y, tran_z;
        float rot_x_adjust, rot_y_adjust, rot_z_adjust;
        tran_x = T_lidar2cam2_bias(0, 3);
        tran_y = T_lidar2cam2_bias(1, 3);
        tran_z = T_lidar2cam2_bias(2, 3);
        rot_x_adjust = 0;
        rot_y_adjust = 0;
        rot_z_adjust = 0;
        float max_score = 0;
        int nn = 0;
        int num_ten_thous = 0;
        if (got_fine_result)
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
        }
        else
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
        }

        if (config["many_or_one"].as<int>() == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < 1000 * 125; i++)
            {
                int dr = i / 1000;       // 0-125
                int drx = dr / 25;       // 0-5
                int dry = (dr / 5) % 5;  // 0-5
                int drz = dr % 5;        // 0-5
                int dt = i % 1000;       // 0-1000
                int dx = (dt / 100);     // 0-10
                int dy = (dt / 10) % 10; // 0-10
                int dz = dt % 10;        // 0-10
                float score = 0;
                Eigen::Matrix4f T_lidar2cam2_bias_copy; // = T_lidar2cam2_bias;
                // std::cout << "(" << dx << ", " << dy << ", " << dz << ")" << std::endl;
                Eigen::AngleAxisf rot_dx(M_PI * (-0.2 + 0.1 * drx) / 180, Eigen::Vector3f(1, 0, 0));
                Eigen::AngleAxisf rot_dy(M_PI * (-0.2 + 0.1 * dry) / 180, Eigen::Vector3f(0, 1, 0));
                Eigen::AngleAxisf rot_dz(M_PI * (-0.2 + 0.1 * drz) / 180, Eigen::Vector3f(0, 0, 1));
                Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);
                ;
                T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);
                T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) - 0.1 + dx * 0.01;
                T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) - 0.1 + dy * 0.01;
                T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) - 0.1 + dz * 0.01;
                score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                omp_set_lock(&mylock);
                {
                    if (nn / 10000 >= num_ten_thous)
                    {
                        num_ten_thous++;
                        std::cout << "n: " << 10000 * (int)(nn / 10000) << std::endl;
                    }
                    nn++;
                    if (score > max_score)
                    {
                        max_score = score;
                        tran_x = T_lidar2cam2_bias_copy(0, 3);
                        tran_y = T_lidar2cam2_bias_copy(1, 3);
                        tran_z = T_lidar2cam2_bias_copy(2, 3);
                        rot_x_adjust = -0.2 + 0.1 * drx;
                        rot_y_adjust = -0.2 + 0.1 * dry;
                        rot_z_adjust = -0.2 + 0.1 * drz;
                    }
                }
                omp_unset_lock(&mylock);
            }
            std::cout << "image " << ppp << " :" << std::endl;
            std::cout << "Max_Score: " << max_score << std::endl;
            std::cout << "(tran_x, tran_y, tran_z) = " << tran_x << ", " << tran_y << ", " << tran_z << std::endl;
            std::cout << "(new_rx, new_ry, new_rz) = " << bias_x + rot_x_adjust << ", " << bias_y + rot_y_adjust << ", " << bias_z + rot_z_adjust << std::endl;
            std::cout << "Time: " << omp_get_wtime() - starTime << std::endl;
        }
        if (config["many_or_one"].as<int>() == 2)
        {
            max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            std::cout << "Max_Score: " << max_score << std::endl;
        }

        std::cout << "Kitti GoundTruth score   =  " << countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_unbias, T_cam2image) << std::endl;
        std::cout << "Before Calibration score =  " << countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) << std::endl;
        bool flag = true;
        if (config["many_or_one"].as<int>() == 3)
        {
            //T_lidar2cam_top3, T_lidar2image is Matrix3*4 //lida2image=T*(T_cam02cam2)*T_cam2image
            T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
            T_lidar2image = T_cam2image * T_lidar2cam_top3;
            cv::Mat image = images[config["show_num"].as<int>()].clone();

            cv::Mat dis_imags_rgb;
            if (distance_images[config["show_num"].as<int>()].channels() > 1)
            {
                cv::cvtColor(distance_images[config["show_num"].as<int>()], dis_imags_rgb, cv::COLOR_GRAY2BGR);
            }
            else
            {
                distance_images[config["show_num"].as<int>()].copyTo(dis_imags_rgb);
            }
            pcl::PointXYZI r;
            Eigen::Vector4f raw_point;
            Eigen::Vector4f trans_point;
            Eigen::Vector3f trans_point2;
            Eigen::Vector3f trans_point3;
            cv::Mat raw_image, gray_image, edge_distance_image, edge_distance_image2;
            double deep, deep_config; //deep_config: normalize, max deep
            int point_r;
            float better_cnt = 0;
            deep_config = config["deep_config"].as<double>();
            point_r = config["point_r"].as<int>();

            // if(ppp<pcs.size()-2 && config["3frame_score"].as<bool>())
            // {
            //     max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image)
            //                     +countScore(pc_features_local[ppp+1], distance_images_local[ppp+1], T_lidar2cam2_bias, T_cam2image)
            //                     +countScore(pc_features_local[ppp+2], distance_images_local[ppp+2], T_lidar2cam2_bias, T_cam2image))/3.0;
            //     std::cout << "3 frame socre" << std::endl;
            // }
            // else
            // {
            //     if(got_fine_result)
            //     {
            //         max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
            //     }
            //     else
            //     {
            //         max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            //     }

            //     std::cout << "1 frame socre" << std::endl;
            // }
            if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
            {
                if (got_fine_result)
                {
                    max_score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                }
                else
                {
                    max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                }
            }
            else
            {
                // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                if (got_fine_result)
                {
                    max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
                }
                else
                {
                    max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                }
            }
            //    int i = 0;

            while (flag)
            {
                flag = false;
                tran_x = T_lidar2cam2_bias(0, 3);
                tran_y = T_lidar2cam2_bias(1, 3);
                tran_z = T_lidar2cam2_bias(2, 3);
#pragma omp parallel for
                //Both adjusting tran and angle
                /*for (int i = 0; i < 27*27; i++) {

                    int dr = i / 27;      // 0-125
                    int drx = dr / 9;     // 0-5
                    int dry = (dr / 3) % 3; // 0-5
                    int drz = dr % 3;       // 0-5
                    int dt = i % 27;      // 0-1000
                    int dx = (dt / 9);    // 0-10
                    int dy = (dt / 3) % 3;// 0-10
                    int dz = dt % 3;       // 0-
                */
                // Only update angle
                for (int i = 0; i < 27; i++)
                {

                    int drx = i / 9;       // 0-5
                    int dry = (i / 3) % 3; // 0-5
                    int drz = i % 3;       // 0-5

                    Eigen::Matrix4f T_lidar2cam2_bias_copy;                                                                                // = T_lidar2cam2_bias;
                    Eigen::AngleAxisf rot_dx(M_PI * (-iterate_ang_step_big + iterate_ang_step_big * drx) / 180, Eigen::Vector3f(1, 0, 0)); //0.06
                    Eigen::AngleAxisf rot_dy(M_PI * (-iterate_ang_step_big + iterate_ang_step_big * dry) / 180, Eigen::Vector3f(0, 1, 0));
                    Eigen::AngleAxisf rot_dz(M_PI * (-iterate_ang_step_big + iterate_ang_step_big * drz) / 180, Eigen::Vector3f(0, 0, 1));
                    Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                    R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);

                    T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                    T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                    T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                    T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                    T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                    T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                    T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                    T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                    T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);
                    //Both adjusting tran and angle
                    // T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) - iterate_tra_step_big + dx * iterate_tra_step_big;//0.002
                    // T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) - iterate_tra_step_big + dy * iterate_tra_step_big;
                    // T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) - iterate_tra_step_big + dz * iterate_tra_step_big;

                    //Only update angle
                    T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3); // - iterate_tra_step_big + dx * iterate_tra_step_big;//0.002
                    T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3); // - iterate_tra_step_big + dy * iterate_tra_step_big;
                    T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3); // - iterate_tra_step_big + dz * iterate_tra_step_big;

                    float score = 0;
                    if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                    {
                        if (got_fine_result)
                        {
                            score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                        }
                        else
                        {
                            score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                        }
                    }
                    else
                    {
                        // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        if (got_fine_result)
                        {
                            score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        }
                        else
                        {
                            score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        }
                    }
                    omp_set_lock(&mylock);
                    {
                        if (score > max_score)
                        {
                            flag = true;
                            // i = 0;
                            max_score = score;
                            T_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias_copy(0, 0);
                            T_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias_copy(0, 1);
                            T_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias_copy(0, 2);
                            T_lidar2cam2_bias(0, 3) = T_lidar2cam2_bias_copy(0, 3);
                            T_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias_copy(1, 0);
                            T_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias_copy(1, 1);
                            T_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias_copy(1, 2);
                            T_lidar2cam2_bias(1, 3) = T_lidar2cam2_bias_copy(1, 3);
                            T_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias_copy(2, 0);
                            T_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias_copy(2, 1);
                            T_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias_copy(2, 2);
                            T_lidar2cam2_bias(2, 3) = T_lidar2cam2_bias_copy(2, 3);
                            T_lidar2cam2_bias(3, 0) = T_lidar2cam2_bias_copy(3, 0);
                            T_lidar2cam2_bias(3, 1) = T_lidar2cam2_bias_copy(3, 1);
                            T_lidar2cam2_bias(3, 2) = T_lidar2cam2_bias_copy(3, 2);
                            T_lidar2cam2_bias(3, 3) = T_lidar2cam2_bias_copy(3, 3);
                            better_cnt++;
                        }
                    }
                    omp_unset_lock(&mylock);
                }
                std::cout << "better_cnt first = " << better_cnt << std::endl;

                T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
                T_lidar2image = T_cam2image * T_lidar2cam_top3;
                cv::Mat dis_imags_rgb;
                // cv::cvtColor(distance_images[ppp], dis_imags_rgb, cv::COLOR_GRAY2BGR);
                if (distance_images[ppp].channels() > 1)
                {
                    cv::cvtColor(distance_images[ppp], dis_imags_rgb, cv::COLOR_GRAY2BGR);
                }
                else
                {
                    distance_images[ppp].copyTo(dis_imags_rgb);
                }
                pcl::PointXYZI r;
                Eigen::Vector4f raw_point;
                Eigen::Vector4f trans_point;
                Eigen::Vector3f trans_point2;
                Eigen::Vector3f trans_point3;
                cv::Mat raw_image, gray_image, edge_distance_image, edge_distance_image2;
                double deep, deep_config; //deep_config: normalize, max deep
                int point_r;
                deep_config = config["deep_config"].as<double>();
                point_r = config["point_r"].as<int>();
            } //while loop end

            // max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            // if(ppp<pcs.size()-2 && config["3frame_score"].as<bool>())
            // {
            //     max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image)
            //                     +countScore(pc_features_local[ppp+1], distance_images_local[ppp+1], T_lidar2cam2_bias, T_cam2image)
            //                     +countScore(pc_features_local[ppp+2], distance_images_local[ppp+2], T_lidar2cam2_bias, T_cam2image))/3.0;
            // }
            // else
            // {
            //     max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            // }
            if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
            {
                if (got_fine_result)
                {
                    max_score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                }
                else
                {
                    max_score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias, T_cam2image)) / 3.0;
                }
            }
            else
            {
                // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                if (got_fine_result)
                {
                    max_score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
                }
                else
                {
                    max_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
                }
            }

            flag = true;
            while (flag)
            {
                flag = false;
                tran_x = T_lidar2cam2_bias(0, 3);
                tran_y = T_lidar2cam2_bias(1, 3);
                tran_z = T_lidar2cam2_bias(2, 3);
#pragma omp parallel for
                //Both adjusting tran and angle
                /*    for (int i = 0; i < 27*27; i++) {


                    int dr = i / 27;      // 0-125
                    int drx = dr / 9 ;     // 0-5
                    int dry = (dr / 3) % 3; // 0-5
                    int drz = dr % 3;       // 0-5
                    int dt = i % 27;      // 0-1000
                    int dx = (dt / 9);    // 0-10
                    int dy = (dt / 3) % 3;// 0-10
                    int dz = dt % 3;       // 0-
    */
                // Only update angle
                for (int i = 0; i < 27; i++)
                {

                    int drx = i / 9;       // 0-5
                    int dry = (i / 3) % 3; // 0-5
                    int drz = i % 3;       // 0-5

                    Eigen::Matrix4f T_lidar2cam2_bias_copy;                                                                                    // = T_lidar2cam2_bias;
                    Eigen::AngleAxisf rot_dx(M_PI * (-iterate_ang_step_small + iterate_ang_step_small * drx) / 180, Eigen::Vector3f(1, 0, 0)); //0.01
                    Eigen::AngleAxisf rot_dy(M_PI * (-iterate_ang_step_small + iterate_ang_step_small * dry) / 180, Eigen::Vector3f(0, 1, 0));
                    Eigen::AngleAxisf rot_dz(M_PI * (-iterate_ang_step_small + iterate_ang_step_small * drz) / 180, Eigen::Vector3f(0, 0, 1));
                    Eigen::Matrix3f R_lidar2cam2_bias_temp = Eigen::Matrix3f::Identity();
                    // R_lidar2cam2_bias_temp(0, 0)=T_lidar2cam2_bias_copy(0,0);
                    // R_lidar2cam2_bias_temp(0, 1)=T_lidar2cam2_bias_copy(0,1);
                    // R_lidar2cam2_bias_temp(0, 2)=T_lidar2cam2_bias_copy(0,2);
                    // R_lidar2cam2_bias_temp(1, 0)=T_lidar2cam2_bias_copy(1,0);
                    // R_lidar2cam2_bias_temp(1, 1)=T_lidar2cam2_bias_copy(1,1);
                    // R_lidar2cam2_bias_temp(1, 2)=T_lidar2cam2_bias_copy(1,2);
                    // R_lidar2cam2_bias_temp(2, 0)=T_lidar2cam2_bias_copy(2,0);
                    // R_lidar2cam2_bias_temp(2, 1)=T_lidar2cam2_bias_copy(2,1);
                    // R_lidar2cam2_bias_temp(2, 2)=T_lidar2cam2_bias_copy(2,2);
                    R_lidar2cam2_bias_temp = rot_dx.matrix() * rot_dy.matrix() * rot_dz.matrix() * T_lidar2cam2_bias.block<3, 3>(0, 0);
                    T_lidar2cam2_bias_copy(0, 0) = R_lidar2cam2_bias_temp(0, 0);
                    T_lidar2cam2_bias_copy(0, 1) = R_lidar2cam2_bias_temp(0, 1);
                    T_lidar2cam2_bias_copy(0, 2) = R_lidar2cam2_bias_temp(0, 2);
                    T_lidar2cam2_bias_copy(1, 0) = R_lidar2cam2_bias_temp(1, 0);
                    T_lidar2cam2_bias_copy(1, 1) = R_lidar2cam2_bias_temp(1, 1);
                    T_lidar2cam2_bias_copy(1, 2) = R_lidar2cam2_bias_temp(1, 2);
                    T_lidar2cam2_bias_copy(2, 0) = R_lidar2cam2_bias_temp(2, 0);
                    T_lidar2cam2_bias_copy(2, 1) = R_lidar2cam2_bias_temp(2, 1);
                    T_lidar2cam2_bias_copy(2, 2) = R_lidar2cam2_bias_temp(2, 2);

                    //Both adjusting tran and angle
                    // T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3) - iterate_tra_step_small + dx * iterate_tra_step_small;//0.002
                    // T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3) - iterate_tra_step_small + dy * iterate_tra_step_small;
                    // T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3) - iterate_tra_step_small + dz * iterate_tra_step_small;

                    //Only update angle
                    T_lidar2cam2_bias_copy(0, 3) = T_lidar2cam2_bias(0, 3); // - iterate_tra_step_small + dx * iterate_tra_step_small;//0.002
                    T_lidar2cam2_bias_copy(1, 3) = T_lidar2cam2_bias(1, 3); // - iterate_tra_step_small + dy * iterate_tra_step_small;
                    T_lidar2cam2_bias_copy(2, 3) = T_lidar2cam2_bias(2, 3); // - iterate_tra_step_small + dz * iterate_tra_step_small;

                    float score = 0;
                    // float score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                    // if(ppp<pcs.size()-2 && config["3frame_score"].as<bool>())
                    // {
                    //     score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image)
                    //             +countScore(pc_features_local[ppp+1], distance_images_local[ppp+1], T_lidar2cam2_bias_copy, T_cam2image)
                    //             +countScore(pc_features_local[ppp+2], distance_images_local[ppp+2], T_lidar2cam2_bias_copy, T_cam2image))/3.0;
                    // }
                    // else
                    // {
                    //     if(got_fine_result)
                    //     {
                    //         score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                    //     }
                    //     else
                    //     {
                    //         score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                    //     }
                    //     // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                    // }
                    if (ppp < pcs.size() - 2 && config["3frame_score"].as<bool>())
                    {
                        if (got_fine_result)
                        {
                            score = (countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local_thin[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local_thin[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                        }
                        else
                        {
                            score = (countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 1], distance_images_local[ppp + 1], T_lidar2cam2_bias_copy, T_cam2image) + countScore(pc_features_local[ppp + 2], distance_images_local[ppp + 2], T_lidar2cam2_bias_copy, T_cam2image)) / 3.0;
                        }
                    }
                    else
                    {
                        // score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        if (got_fine_result)
                        {
                            score = countScore(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        }
                        else
                        {
                            score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias_copy, T_cam2image);
                        }
                    }
                    omp_set_lock(&mylock);
                    {
                        if (score > max_score)
                        {
                            flag = true;
                            // i = 0;
                            max_score = score;
                            T_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias_copy(0, 0);
                            T_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias_copy(0, 1);
                            T_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias_copy(0, 2);
                            T_lidar2cam2_bias(0, 3) = T_lidar2cam2_bias_copy(0, 3);
                            T_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias_copy(1, 0);
                            T_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias_copy(1, 1);
                            T_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias_copy(1, 2);
                            T_lidar2cam2_bias(1, 3) = T_lidar2cam2_bias_copy(1, 3);
                            T_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias_copy(2, 0);
                            T_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias_copy(2, 1);
                            T_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias_copy(2, 2);
                            T_lidar2cam2_bias(2, 3) = T_lidar2cam2_bias_copy(2, 3);
                            T_lidar2cam2_bias(3, 0) = T_lidar2cam2_bias_copy(3, 0);
                            T_lidar2cam2_bias(3, 1) = T_lidar2cam2_bias_copy(3, 1);
                            T_lidar2cam2_bias(3, 2) = T_lidar2cam2_bias_copy(3, 2);
                            T_lidar2cam2_bias(3, 3) = T_lidar2cam2_bias_copy(3, 3);
                            better_cnt++;
                        }
                    }
                    omp_unset_lock(&mylock);
                }
                std::cout << "better_cnt second = " << better_cnt << std::endl;

                T_lidar2cam_top3 = T_lidar2cam2_bias.topRows(3);
                T_lidar2image = T_cam2image * T_lidar2cam_top3;
                cv::Mat image = images[config["show_num"].as<int>()].clone();
                cv::Mat dis_imags_rgb;
                // cv::cvtColor(distance_images[config["show_num"].as<int>()], dis_imags_rgb, cv::COLOR_GRAY2BGR);
                if (distance_images[config["show_num"].as<int>()].channels() > 1)
                {
                    cv::cvtColor(distance_images[config["show_num"].as<int>()], dis_imags_rgb, cv::COLOR_GRAY2BGR);
                }
                else
                {
                    distance_images[config["show_num"].as<int>()].copyTo(dis_imags_rgb);
                }
                pcl::PointXYZI r;
                Eigen::Vector4f raw_point;
                Eigen::Vector4f trans_point;
                Eigen::Vector3f trans_point2;
                Eigen::Vector3f trans_point3;
                cv::Mat raw_image, gray_image, edge_distance_image, edge_distance_image2;
                double deep, deep_config; //deep_config: normalize, max deep
                int point_r;
                deep_config = config["deep_config"].as<double>();
                point_r = config["point_r"].as<int>();
            } //while loop end

            T_lidar2cam2_bias_vec.push_back(T_lidar2cam2_bias);
            if (T_lidar2cam2_bias_vec.size() >= 3)
            {
                Eigen::Matrix3f R_lidar2cam2_bias_delt1 = T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 2].block(0, 0, 3, 3).inverse() * T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 1].block(0, 0, 3, 3);
                Eigen::Matrix3f R_lidar2cam2_bias_delt2 = T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 3].block(0, 0, 3, 3).inverse() * T_lidar2cam2_bias_vec[T_lidar2cam2_bias_vec.size() - 1].block(0, 0, 3, 3);
                Eigen::Vector3f R_euler_xyz_delt1 = R_lidar2cam2_bias_delt1.eulerAngles(0, 1, 2); //x y z
                Eigen::Vector3f R_euler_xyz_delt2 = R_lidar2cam2_bias_delt2.eulerAngles(0, 1, 2); //x y z
                for (int r = 0; r < 3; ++r)
                {
                    if ((R_euler_xyz_delt1[r] > 0) && (abs(R_euler_xyz_delt1[r] - 3) < 0.2))
                    {
                        R_euler_xyz_delt1[r] = (R_euler_xyz_delt1[r] - M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt1[r] < 0) && (abs(R_euler_xyz_delt1[r] + 3) < 0.2))
                    {
                        R_euler_xyz_delt1[r] = (R_euler_xyz_delt1[r] + M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt2[r] > 0) && (abs(R_euler_xyz_delt2[r] - 3) < 0.2))
                    {
                        R_euler_xyz_delt2[r] = (R_euler_xyz_delt2[r] - M_PI) * 180.0 / M_PI;
                    }
                    if ((R_euler_xyz_delt2[r] < 0) && (abs(R_euler_xyz_delt2[r] + 3) < 0.2))
                    {
                        R_euler_xyz_delt2[r] = (R_euler_xyz_delt2[r] + M_PI) * 180.0 / M_PI;
                    }
                }
                std::cout << "R euler xyz delt 1 " << R_euler_xyz_delt1 * 180.0 / M_PI << std::endl;
                std::cout << "R euler xyz delt 2 " << R_euler_xyz_delt2 * 180.0 / M_PI << std::endl;
                std::cout << "R euler delt = " << abs(R_euler_xyz_delt1[0] - R_euler_xyz_delt2[0]) * 180.0 / M_PI << " " << abs(R_euler_xyz_delt1[1] - R_euler_xyz_delt2[1]) * 180.0 / M_PI << " " << abs(R_euler_xyz_delt1[2] - R_euler_xyz_delt2[2]) * 180.0 / M_PI << std::endl;
                // if((abs(R_euler_xyz_delt1[0]-R_euler_xyz_delt2[0]) * 180.0 / M_PI  < 1) && (abs(R_euler_xyz_delt1[1]-R_euler_xyz_delt2[1]) * 180.0 / M_PI  < 1) &&
                // (abs(R_euler_xyz_delt1[2]-R_euler_xyz_delt2[2]) * 180.0 / M_PI < 1) &&
                // isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image))
                bool big_step = false;
                float score;
                if (got_fine_result)
                {
                    big_step = isWhiteEnough(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, score);
                }
                else
                {
                    big_step = isWhiteEnough(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, got_fine_result, score);
                }

                float confidence_step = countConfidence(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);

                if (big_step)
                // if(confidence_step>0.7)
                {
                    got_fine_result = true;
                    large_small_step.push_back(1);
                    iterate_ang_step_big = 0.18;
                    iterate_tra_step_big = 0; //0.001;
                    iterate_ang_step_small = 0.03;
                    iterate_tra_step_small = 0; //0.001;
                    std::cout << "!!!!!!!!Enter!!!!!!! " << R_euler_xyz_delt1[0] - R_euler_xyz_delt2[0] << std::endl;
                }
                else //if(ppp<=21)
                {
                    got_fine_result = false;
                    large_small_step.push_back(0);
                    iterate_ang_step_big = 1;
                    iterate_tra_step_big = 0; //0.002;
                    iterate_ang_step_small = 0.12;
                    iterate_tra_step_small = 0; //0.001;
                }
            }
            project2image(pc_features_local[ppp], images_withouthist[ppp], image_before_optimize, T_lidar2cam2_bias, T_cam2image);

            //设定g2o
            typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
            typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
            auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            g2o::SparseOptimizer optimizer; // 图模型
            optimizer.setAlgorithm(solver); // 设置求解器
            optimizer.setVerbose(true);     // 打开调试输出

            VertexPose *v = new VertexPose();
            v->setId(0);
            v->setEstimate(AutoCalib::toSE3d(T_lidar2cam2_bias));
            optimizer.addVertex(v);

            for (int i = 0; i < pc_features_local[ppp]->size(); i++)
            {
                EdgeProjectionPoseOnly *e = new EdgeProjectionPoseOnly(pc_features_local[ppp]->points[i], distance_images_local_thin[ppp], T_cam2image, config["add_dis_weight"].as<bool>());

                e->setId(i);
                e->setVertex(0, v);
                e->setMeasurement(0);
                e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
                optimizer.addEdge(e);
            }

            cout << "optimizing ..." << endl;
            optimizer.initializeOptimization();
            optimizer.optimize(30);

            cout << "End optimization" << endl;

            T_lidar2cam2_bias = AutoCalib::toMatrix4f(v->estimate().matrix());
            std::cout<<T_lidar2cam2_bias<<std::endl;

            float confidence = countConfidence(pc_features_local[ppp], distance_images_local_thin[ppp], T_lidar2cam2_bias, T_cam2image);
            confidence_vec.push_back(confidence);

            std::cout << "Better rate = " << better_cnt / (27 * 27) / 2 << std::endl;
            std::cout << "Picture: " << ppp << std::endl;
            std::cout << "Time: " << omp_get_wtime() - starTime << std::endl;
            float aft_score = countScore(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image);
            std::cout << "After Calibration score  =  " << aft_score << std::endl;
            std::cout << std::endl;
            // rectangle(distance_images_local[ppp],Point(boxes[i].x,boxes[i].y),
            //         Point((boxes[i].x+boxes[i].width),(boxes[i].y+boxes[i].height)),
            //         Scalar(blue,green,red),2,8,0);                                //draw boxes

            std::ostringstream str;
            str << aft_score;
            cv::String label = cv::String(str.str());
            int baseLine = 0;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            putText(distance_images_local[ppp], label, cv::Point(0, 0), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 1), 2);
        }

        R_lidar2cam2_bias(0, 0) = T_lidar2cam2_bias(0, 0);
        R_lidar2cam2_bias(0, 1) = T_lidar2cam2_bias(0, 1);
        R_lidar2cam2_bias(0, 2) = T_lidar2cam2_bias(0, 2);
        R_lidar2cam2_bias(1, 0) = T_lidar2cam2_bias(1, 0);
        R_lidar2cam2_bias(1, 1) = T_lidar2cam2_bias(1, 1);
        R_lidar2cam2_bias(1, 2) = T_lidar2cam2_bias(1, 2);
        R_lidar2cam2_bias(2, 0) = T_lidar2cam2_bias(2, 0);
        R_lidar2cam2_bias(2, 1) = T_lidar2cam2_bias(2, 1);
        R_lidar2cam2_bias(2, 2) = T_lidar2cam2_bias(2, 2);

        Eigen::Matrix3f R_calibrated_delt = R_gt.inverse() * R_lidar2cam2_bias;
        Eigen::Vector3f euler_ang_delt_zyx = R_calibrated_delt.eulerAngles(2, 1, 0); //zyx
        Eigen::Vector3f euler_ang_delt_xyz = R_calibrated_delt.eulerAngles(0, 1, 2); //xyz

        Eigen::Matrix3f R_calibrated_inv_delt = R_lidar2cam2_bias.inverse() * R_gt;
        Eigen::Vector3f euler_ang_inv_delt_zyx = R_calibrated_inv_delt.eulerAngles(2, 1, 0); //zyx
        Eigen::Vector3f euler_ang_inv_delt_xyz = R_calibrated_inv_delt.eulerAngles(0, 1, 2); //xyz

        euler_ang_delt_zyx_vec.push_back(euler_ang_delt_zyx);
        euler_ang_delt_xyz_vec.push_back(euler_ang_delt_xyz);
        euler_ang_inv_delt_zyx_vec.push_back(euler_ang_inv_delt_zyx);
        euler_ang_inv_delt_xyz_vec.push_back(euler_ang_inv_delt_xyz);
        std::cout << "1== " << euler_ang_delt_zyx[0] * 180.0 / M_PI << " " << euler_ang_delt_zyx[1] * 180.0 / M_PI << " " << euler_ang_delt_zyx[2] * 180.0 / M_PI << std::endl;
        std::cout << "2== " << euler_ang_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_delt_xyz[1] * 180.0 / M_PI << " " << euler_ang_delt_xyz[2] * 180.0 / M_PI << std::endl;
        std::cout << "3== " << euler_ang_inv_delt_zyx[0] * 180.0 / M_PI << " " << euler_ang_inv_delt_zyx[1] * 180.0 / M_PI << " " << euler_ang_inv_delt_zyx[2] * 180.0 / M_PI << std::endl;
        std::cout << "4== " << euler_ang_inv_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_inv_delt_xyz[1] * 180.0 / M_PI << " " << euler_ang_inv_delt_xyz[2] * 180.0 / M_PI << std::endl;

        Eigen::Vector3f euler_ang_calibrated_zyx = R_lidar2cam2_bias.eulerAngles(0, 1, 2); //x y z
        euler_ang_calibrated_zyx_vec.push_back(euler_ang_calibrated_zyx);

        std::cout << "euler angle calibrated = " << std::endl
                  << euler_ang_calibrated_zyx * 180.0 / M_PI << std::endl
                  << "euler angle gt = " << std::endl
                  << euler_ang_gt_zyx * 180.0 / M_PI << std::endl;

        std::cout << "euler angle delt = " << std::endl
                  << euler_ang_delt_xyz[0] * 180.0 / M_PI << " " << euler_ang_delt_xyz[1] * 180.0 / M_PI
                  << " " << euler_ang_delt_xyz[2] * 180.0 / M_PI << std::endl;
        std::cout << std::endl;
#ifdef USE_ORB
        if (ppp > 0)
        {
            std::cout << "R_cam_Del = " << std::endl
                      << R_cam_del << std::endl;
            Eigen::Matrix3f R_lidar2cam2_bias_last = T_lidar2cam2_bias_last.block(0, 0, 3, 3);
            Eigen::Matrix3f R_cam22lidar_bias = R_lidar2cam2_bias.inverse();
            Eigen::Matrix3f R_lidar_cal = R_cam22lidar_bias * R_cam_del.inverse() * R_lidar2cam2_bias_last;
            Eigen::Matrix3f R_lidar2cam2_cal = R_cam_del * R_lidar2cam2_bias * rot_icp.inverse();
            Eigen::Matrix3f R_lidar2cam2_unbias = T_lidar2cam2_unbias.block(0, 0, 3, 3);
            Eigen::Matrix3f R_lidar2cam2_bias = T_lidar2cam2_bias.block(0, 0, 3, 3);
            // Eigen::Matrix3f R_lidar_cal = R_cam_del * R_cam22lidar_bias;
            // std::cout << "R_lidar2cam2_bias = " << std::endl << R_lidar2cam2_bias << std::endl;
            // std::cout << "R_cam22lidar_bias = " << std::endl << R_cam22lidar_bias << std::endl;
            // std::cout << "R_lidar_cal = " << std::endl << R_lidar_cal << std::endl;
            std::cout << "====Compare Result====" << std::endl;
            std::cout << "T_lidar2cam2_unbias = " << std::endl
                      << T_lidar2cam2_unbias.block(0, 0, 3, 3) << std::endl;
            std::cout << "T_lidar2cam2_bias = " << std::endl
                      << T_lidar2cam2_bias.block(0, 0, 3, 3) << std::endl;
            std::cout << "R_lidar2cam2_cal = " << std::endl
                      << R_lidar2cam2_cal << std::endl;
            Eigen::Vector3f euler_ang_zyx_R_lidar2cam2_unbias = R_lidar2cam2_unbias.eulerAngles(2, 1, 0); //z y x
            std::cout << "Ground truth -- euler_ang_zyx_R_lidar2cam2_unbias is " << std::endl
                      << euler_ang_zyx_R_lidar2cam2_unbias * 180.0 / M_PI << std::endl;
            Eigen::Vector3f euler_ang_zyx_R_lidar2cam2_bias = R_lidar2cam2_bias.eulerAngles(2, 1, 0); //z y x
            std::cout << "Method Line -- euler_ang_zyx_R_lidar2cam2_bias is " << std::endl
                      << euler_ang_zyx_R_lidar2cam2_bias * 180.0 / M_PI << std::endl;
            Eigen::Vector3f euler_ang_zyx_R_lidar2cam2_cal = R_lidar2cam2_cal.eulerAngles(2, 1, 0); //z y x
            std::cout << "Method ORB -- euler_ang_zyx_R_lidar2cam2_cal is " << std::endl
                      << euler_ang_zyx_R_lidar2cam2_cal * 180.0 / M_PI << std::endl;
        }

#endif

        T_lidar2cam2_bias_last = T_lidar2cam2_bias; //lidar2cam last

        cv::Mat image2;
        if (got_fine_result)
        {
            project2image(pc_features_local[ppp], distance_images_local_thin[ppp], image2, T_lidar2cam2_bias, T_cam2image);
        }
        else
        {
            project2image(pc_features_local[ppp], distance_images_local[ppp], image2, T_lidar2cam2_bias, T_cam2image);
        }

        cv::Mat dst;
        dst.create(image1.rows * 4, image1.cols, CV_8UC3);

        pcl::PointCloud<pcl::PointXYZI>::Ptr line_features(new pcl::PointCloud<pcl::PointXYZI>);
        // 分隔符
        const char *SEPARATOR = " ";
        // 读取文本数据
        std::ifstream inFile("../data" + std::to_string(frame_cnt) + "/pc_lines");
        std::string lineStr; // 文件中的一行数据

        if (inFile) // 有该文件
        {
            int i = 0;                       // 循环下标
            while (getline(inFile, lineStr)) // line中不包括每行的换行符
            {
                // string转char *
                char *lineCharArray;
                const int len = lineStr.length();
                lineCharArray = new char[len + 1];
                strcpy(lineCharArray, lineStr.c_str());

                char *p;                        // 分隔后的字符串
                p = strtok(lineCharArray, " "); // 按照spaceChar分隔
                std::vector<double> data_temp;
                // 将数据加入vector中
                while (p)
                {
                    data_temp.push_back(atof(p));
                    // int a = strlen(p);
                    p = strtok(NULL, " ");
                }
                pcl::PointXYZI pc_temp;
                pc_temp.x = data_temp[0];
                pc_temp.y = data_temp[1];
                pc_temp.z = data_temp[2];
                line_features->push_back(pc_temp);
                data_temp.clear();
            }
        }
        cv::Mat image_line_feature;
        project2image(line_features, gray_image_vec[0], image_line_feature, T_lidar2cam2_unbias, T_cam2image);
        // cv::imshow("line_feature",image_line_feature);
        // cv::waitKey(0);

        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        cv::Mat image4, image_result;
        filterUnusedPoiintCloudFeature(pc_features_local[ppp], distance_images_local[ppp], T_lidar2cam2_bias, T_cam2image, pc_feature_filtered);
        project2image(pc_feature_filtered, images[ppp], image4, T_lidar2cam2_bias, T_cam2image);
        project2image(filtered_pc[ppp], images[ppp], image_result, T_lidar2cam2_unbias, T_cam2image);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/project_result_" + std::to_string(ppp) + ".png", image_result);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/original_bias_" + std::to_string(ppp) + ".png", image2);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/unbias_" + std::to_string(ppp) + ".png", image_unbias);
        cv::imwrite("../data" + std::to_string(frame_cnt) + "/result/calibrated_result_" + std::to_string(ppp) + ".png", image4);
        // project2image(pc_features_local[ppp], images[ppp], image4, T_lidar2cam2_bias, T_cam2image);
        //cv::imwrite("../data0/result/project_result_" + std::to_string(ppp) + ".png", image_result);
        //cv::imwrite("../data0/result/original_bias_" + std::to_string(ppp) + ".png", image2);
        // cv::imwrite("../data0/result/unbias_" + std::to_string(ppp) + ".png", image_unbias);
        //cv::imwrite("../data0/result/calibrated_result_" + std::to_string(ppp) + ".png", image4);
        // image_bias.copyTo(dst(cv::Rect(0, 0, images[ppp].cols, images[ppp].rows)));
        image_unbias.copyTo(dst(cv::Rect(0, 0, images[ppp].cols, images[ppp].rows))); //groundtruth
        // image2.copyTo(dst(cv::Rect(0, image1.rows, image1.cols, image1.rows)));
        image_bias.copyTo(dst(cv::Rect(0, image1.rows, image1.cols, image1.rows))); //原始加偏差图像
        // image1.copyTo(dst(cv::Rect(0, image1.rows, image1.cols, image1.rows)));//distance_images_local图像
        image_before_optimize.copyTo(dst(cv::Rect(0, image1.rows * 2, image1.cols, image1.rows)));
        image4.copyTo(dst(cv::Rect(0, image2.rows * 3, image2.cols, image2.rows))); //最终标定结果

        //    cv::imwrite("/home/mkjhnb/gazebo_ws/test/pic_3" + std::to_string(ppp) + ".png", dst);
        // cv::namedWindow(result_file[ppp], cv::WINDOW_NORMAL);
        // cv::imshow(result_file[ppp], dst);
        // std::ostringstream str;
        //   str << min_dist;
        //   marker.text = str.str();
        cv::imwrite(result_file[ppp], dst);
        // cv::waitKey(0);
        result_gt_vec.push_back(T_lidar2cam2_unbias);
        calibrated_result_vec.push_back(T_lidar2cam2_bias);
    }
    if (config["save_calibrated_result"].as<bool>())
    {
        if (calibrated_result)
        {
            calibrated_result << "x_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][2] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
            calibrated_result << "y_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][1] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
            calibrated_result << "z_c=[";
            for (int i = 0; i < euler_ang_calibrated_zyx_vec.size(); ++i)
            {
                calibrated_result << euler_ang_calibrated_zyx_vec[i][0] * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;
        }
        else
        {
            std::cout << "save error" << std::endl;
        }

        if (calibrated_result)
        {
            calibrated_result << "x2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][2] = euler_ang_delt_zyx_vec[i][2] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][2] = euler_ang_delt_zyx_vec[i][2] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][2] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "y2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][1] = euler_ang_delt_zyx_vec[i][1] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][1] = euler_ang_delt_zyx_vec[i][1] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][1] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "z2_c=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                if (abs(abs(euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI > 0)
                {
                    euler_ang_delt_zyx_vec[i][0] = euler_ang_delt_zyx_vec[i][0] - M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
                else if (abs(abs(euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI < 0)
                {
                    euler_ang_delt_zyx_vec[i][0] = euler_ang_delt_zyx_vec[i][0] + M_PI;
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
                else
                {
                    calibrated_result << euler_ang_delt_zyx_vec[i][0] * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl
                              << std::endl;

            float mean_x = 0, mean_y = 0, mean_z = 0;
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                mean_x = mean_x + euler_ang_delt_zyx_vec[i][2];
                mean_y = mean_y + euler_ang_delt_zyx_vec[i][1];
                mean_z = mean_z + euler_ang_delt_zyx_vec[i][0];
                std::cout << "mean x = " << mean_x << std::endl;
            }
            mean_x = mean_x / euler_ang_delt_zyx_vec.size();
            mean_y = mean_y / euler_ang_delt_zyx_vec.size();
            mean_z = mean_z / euler_ang_delt_zyx_vec.size();
            std::cout << "mean x = " << mean_x << std::endl;

            calibrated_result << "roll=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][2] - mean_x) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "pitch=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][1] - mean_y) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "yaw=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size(); ++i)
            {
                calibrated_result << (euler_ang_delt_zyx_vec[i][0] - mean_z) * 180.0 / M_PI << ",";
            }
            calibrated_result << "]" << std::endl
                              << std::endl;

            calibrated_result << "mean = " << mean_x << " " << mean_y << " " << mean_z << std::endl;
            float mean_x_new = 0, mean_y_new = 0, mean_z_new = 0;
            for (int i = euler_ang_delt_zyx_vec.size() / 4; i < euler_ang_delt_zyx_vec.size() / 4 * 3; ++i)
            {
                mean_x_new += (euler_ang_delt_zyx_vec[i][2] - mean_x);
                mean_y_new += (euler_ang_delt_zyx_vec[i][1] - mean_y);
                mean_z_new += (euler_ang_delt_zyx_vec[i][0] - mean_z);
            }
            mean_x_new = mean_x_new / euler_ang_delt_zyx_vec.size();
            mean_y_new = mean_y_new / euler_ang_delt_zyx_vec.size();
            mean_z_new = mean_z_new / euler_ang_delt_zyx_vec.size();
            calibrated_result << "mean new= " << mean_x_new << " " << mean_y_new << " " << mean_z_new << std::endl;

            calibrated_result << std::endl;
            calibrated_result << "large_small_step=[0,0,";
            for (int i = 0; i < large_small_step.size(); ++i)
            {
                calibrated_result << large_small_step[i] << ",";
            }
            calibrated_result << "]" << std::endl
                              << std::endl;

            float confidence_mean = 0;
            calibrated_result << "confidence=[";
            for (int i = 0; i < confidence_vec.size(); ++i)
            {
                calibrated_result << confidence_vec[i] << ",";
                confidence_mean += confidence_vec[i];
            }
            calibrated_result << "]" << std::endl
                              << std::endl;
            confidence_mean /= confidence_vec.size();

            calibrated_result << "roll_cfd_mean=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            {
                if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
                {
                    calibrated_result << (euler_ang_delt_zyx_vec[i][2] - mean_x) * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "pitch_cfd_mean=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            {
                if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
                {
                    calibrated_result << (euler_ang_delt_zyx_vec[i][1] - mean_y) * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl;

            calibrated_result << "yaw_cfd_mean=[";
            for (int i = 0; i < euler_ang_delt_zyx_vec.size() - 1; ++i)
            {
                if (large_small_step[i] == 1 && large_small_step[i + 1] == 1)
                {
                    calibrated_result << (euler_ang_delt_zyx_vec[i][0] - mean_z) * 180.0 / M_PI << ",";
                }
            }
            calibrated_result << "]" << std::endl
                              << std::endl;
        }
        else
        {
            std::cout << "save error" << std::endl;
        }
    }
    calibrated_result.close();

    /****************************************************************/

    //    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //
    //    viewer->initCameraParameters ();
    //    int v1(0);
    //
    //    viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    //
    //    viewer->setBackgroundColor (0, 0, 0, v1);
    //
    //    viewer->addText ("Radius: 0.01", 10, 10, "v1 text", v1);
    //
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(raw);
    //
    //    viewer->addPointCloud<pcl::PointXYZI> (cloud, rgb, "sample cloud1", v1);
    //    int v2(0);
    //
    //    viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    //
    //    viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
    //
    //    viewer->addText ("Radius: 0.1", 10, 10, "v2 text", v2);
    //
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color (cloud, 0, 255, 0);
    //
    //    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", v2);

    //    pcl::visualization::PCLVisualizer viewer;
    //    int a, b, c, d;
    //    viewer.createViewPort (0.0, 0.0, 0.5, 0.5, a); //(Xmin,Ymin,Xmax,Ymax)设置窗口坐标
    //    viewer.createViewPort (0.5, 0.0, 1.0, 0.5, b);
    //    viewer.createViewPort (0, 0.5, 0.5, 1.0, c);
    //    viewer.createViewPort (0.5, 0.5, 1.0, 1.0, d);
    //    viewer.addPointCloud<pcl::PointXYZI>(raw, "cloud1", a);
    //    viewer.addPointCloud<pcl::PointXYZI>(filtered4, "cloud2", b);
    //    viewer.addPointCloud<pcl::PointXYZI>(filtered_noground, "cloud3",c);
    //    viewer.addPointCloud<pcl::PointXYZI>(edges, "cloud4",d);
    //    std::cout << "Showing" << std::endl;
    //
    //    viewer.spin();
    //    pcl::visualization::CloudViewer viewer1("vision");
    //    pcl::visualization::CloudViewer viewer2("vision2");
    //    viewer.showCloud(raw);
    //    viewer2.showCloud(filtered2);
}

void AutoCalib::PerformNdt(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_parent_cloud_vec,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_child_cloud_vec,
                           int calib_frame_cnt, Eigen::Matrix4f &pcs_current_guess,
                           std::vector<Eigen::Matrix4f> &pcs_calib)
{

    int cnt = std::min(in_parent_cloud_vec.size(), in_child_cloud_vec.size());
    std::cout << "transformation from lidar " << calib_frame_cnt + 1 << " to lidar 1" << std::endl;

    for (int i = 0; i < cnt; i++)
    {
        pcl::console::TicToc time; //申明时间记录
        time.tic();                //time.tic开始  time.toc结束时间

        std::cout << "PointCloud Frame" << i << endl;
        //Initializing Normal Distributions Transform (NDT).
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize(0.05, 0.05, 0.05);
        approximate_voxel_filter.setInputCloud(in_child_cloud_vec[i]);
        approximate_voxel_filter.filter(*filtered_cloud);

        pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

        ndt.setTransformationEpsilon(0.1);
        ndt.setStepSize(0.1);
        ndt.setResolution(0.5);

        ndt.setMaximumIterations(35);

        ndt.setInputSource(filtered_cloud);
        ndt.setInputTarget(in_parent_cloud_vec[i]);

        pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        ndt.align(*output_cloud, pcs_current_guess);
        // pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        // // Set the input source and target
        // icp.setInputCloud(filtered_cloud);
        // icp.setInputTarget(in_parent_cloud_vec[i]);

        // icp.setMaximumIterations(35);
        // pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // icp.align(*output_cloud);

        std::cout << "Normal Distributions Transform converged:" << ndt.hasConverged()
                  << " score: " << ndt.getFitnessScore() << " Probability " << ndt.getTransformationProbability() << std::endl;

        // Transforming unfiltered, input cloud using found transform.
        //pcl::transformPointCloud(*in_child_cloud_, *output_cloud, ndt.getFinalTransformation());

        Eigen::Matrix4f T_pcs_current = ndt.getFinalTransformation();

        //Eigen::Matrix3f rotation_matrix = T_pcs_current.block(0, 0, 3, 3);
        //Eigen::Vector3f translation_vector = T_pcs_current.block(0, 3, 3, 1);

        //std::cout << "This transformation can be replicated using:" << std::endl;
        //std::cout << "rosrun tf static_transform_publisher " << translation_vector.transpose()
        //    << " " << rotation_matrix.eulerAngles(2, 1, 0).transpose() << " /" << parent_frame_
        //    << " /" << child_frame_ << " 10" << std::endl;

        std::cout << "Corresponding transformation matrix:" << std::endl
                  << std::endl
                  << T_pcs_current << std::endl
                  << std::endl;

        pcs_calib.push_back(T_pcs_current);
        std::cout << time.toc() << " ms" << std::endl;
    }
}
// void AutoCalib::PerformICP(std::vector<cv::Mat> & in_parent_images_vec, std::vector<cv::Mat>& in_child_images_vec,
//                            int calib_frame_cnt,Eigen::Matrix4f & images_current_guess,std::vector<Eigen::Matrix4f > & images_calib)
// {
//     int cnt = std::min(in_parent_images_vec.size(), in_child_images_vec.size());
//     std::cout << "transformation from camera " << calib_frame_cnt + 1 << " to camera 1" << std::endl;

//     for (int i = 0; i < cnt; i++)
//     {
//         //cv::Matx44d  pose;

//         double re,pose[16];
//         cv::ppf_match_3d::ICP calculate(100, 0.005f, 2.5f, 8);
//         calculate.registerModelToScene(in_parent_images_vec[i], in_child_images_vec[i],re,pose);

//         images_calib.push_back();
//     }
// }
void AutoCalib::GlobalOptimize(std::vector<std::vector<Eigen::Matrix4f>> &pcs_calib_vec, std::vector<std::vector<Eigen::Matrix4f>> &Calibrated_Result_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_vec,
                               std::vector<std::vector<cv::Mat>> &images_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_feature_vec,
                               std::vector<std::vector<cv::Mat>> &images_feature_vec, bool &add_dis_weight,
                               std::vector<bool> &overlap_vec,
                               int &calib_frame_num, std::vector<Eigen::Matrix3f> &k_vec, std::vector<std::vector<Eigen::Matrix4f>> &Result_gt_vec)
{
    std::vector<std::vector<Eigen::Matrix4f>> Result_Optimize_vec;
    std::vector<std::vector<Eigen::Matrix4f>> Result_lidar_vec;
    std::vector<std::vector<Eigen::Matrix4f>> Result_cam_vec;
    std::vector<Eigen::Matrix4f> vc;
    for (int i = 0; i < calib_frame_num; i++)
    {
        Result_Optimize_vec.push_back(vc);
    }
    for (int i = 0; i < calib_frame_num-1; i++)
    {
        Result_lidar_vec.push_back(vc);
        Result_cam_vec.push_back(vc);
    }
    int cnt = Calibrated_Result_vec[calib_frame_num - 1].size();
    for (int i = 0; i < pcs_calib_vec.size(); i++)
    {
        int a = pcs_calib_vec[i].size();
        int b = Calibrated_Result_vec[i].size();
        cnt = std::min(cnt, a);
        cnt = std::min(cnt, b);
    }
    for (int i = 0; i < cnt; i++)
    {
        // 设定g2o
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; // 图模型
        optimizer.setAlgorithm(solver); // 设置求解器
        optimizer.setVerbose(true);     // 打开调试输出
        for (int j = 0; j < calib_frame_num; j++)
        {
            VertexPose *v = new VertexPose();
            v->setId(j * 2);
            v->setEstimate(AutoCalib::toSE3d(Calibrated_Result_vec[j][i]));
            optimizer.addVertex(v);
        }
        for (int j = 0; j < calib_frame_num - 1; j++)
        {
            VertexPose *v = new VertexPose();
            v->setId(j * 2 + 1);
            v->setEstimate(AutoCalib::toSE3d(pcs_calib_vec[j + 1][i]));
            optimizer.addVertex(v);
        }
        for (int j = 0; j < calib_frame_num - 1; j++)
        {
            std::vector<cv::KeyPoint> keypoints_first, keypoints_second;
            std::vector<cv::DMatch> matches;
            if (overlap_vec[j + 1] = true)
            {

                AutoCalib::find_feature_matches(images_vec[0][i], images_vec[j + 1][i], keypoints_first, keypoints_second, matches);
            }

            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize(0.05, 0.05, 0.05);
            approximate_voxel_filter.setInputCloud(pcs_vec[j + 1][i]);
            approximate_voxel_filter.filter(*filtered_cloud);

            EdgePRScale *e = new EdgePRScale(pcs_feature_vec[0][i], pcs_feature_vec[j + 1][i],
                                             images_feature_vec[0][i], images_feature_vec[j + 1][i],
                                             pcs_vec[0][i], pcs_vec[j + 1][i],
                                             images_vec[0][i], images_vec[j + 1][i],
                                             k_vec[0], k_vec[j + 1], add_dis_weight, overlap_vec[j + 1], calib_frame_num,
                                             keypoints_first, keypoints_second, matches, filtered_cloud);

            e->setId(j);
            e->setVertex(0, optimizer.vertices()[0]);
            e->setVertex(1, optimizer.vertices()[j * 2 + 1]);
            e->setVertex(2, optimizer.vertices()[(j + 1) * 2]);
            e->setMeasurement(0);
            e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            optimizer.addEdge(e);
        }

        cout << "optimizing ..." << endl;
        optimizer.initializeOptimization();
        optimizer.optimize(20);
        cout << "end optimizing" << endl;

        cout << "saving optimization results ..." << endl;
        std::vector<Eigen::Matrix4f> vc;
        for (int j = 0; j < calib_frame_num; j++)
        {

            Eigen::Matrix4f T_after_global_optimze = AutoCalib::toMatrix4f(dynamic_cast<VertexPose *>(optimizer.vertex(j * 2))->estimate().matrix());

            Result_Optimize_vec[j].push_back(T_after_global_optimze);
            cv::Mat image_after_global_potimze;
            AutoCalib::project2image(pcs_feature_vec[j][i], images_vec[j][i], image_after_global_potimze, T_after_global_optimze, k_vec[j]);
            cv::imwrite("../data" + std::to_string(j) + "/result/after_global_optimze_result_" + std::to_string(i) + ".png", image_after_global_potimze);
            // std::string calibrated_result_path = "../data" + std::to_string(j) + "/result/calibrated_result";
            // std::ofstream calibrated_result(calibrated_result_path, std::ios::app);
            // calibrated_result << "after optimize" << std::endl
            //                   << "cnt=[" << i << "]" << T_after_global_optimze << std::endl;

            // calibrated_result.close();
        }
        for (int j = 0; j < calib_frame_num - 1; j++)
        {
            Eigen::Matrix4f T_lidar = AutoCalib::toMatrix4f(dynamic_cast<VertexPose *>(optimizer.vertex(j * 2+1))->estimate().matrix());

            Result_lidar_vec[j].push_back(T_lidar);
            //Result_cam_vec[j].push_back(T_lidar);
            std::string calibrated_result_path = "../data" + std::to_string(j+1) + "/result/calibrated_result";
            std::ofstream calibrated_result(calibrated_result_path, std::ios::app);
            calibrated_result << "lidar " <<std::to_string(j+1) << " to lidar0"<< std::endl
                              << "cnt=[" << i << "]" << T_lidar << std::endl;

            calibrated_result.close();
        }
    }
    std::vector<std::vector<Eigen::Vector3f>> euler_ang_delt_xyz_after_global_optimze_vec;
    std::vector<Eigen::Vector3f> vec;
    std::vector<std::vector<Eigen::Vector3f>> euler_ang_xyz_after_global_optimze_vec;
    std::vector<Eigen::Vector3f> euler_ang_vec;
    std::vector<std::vector<float>> confidence_after_global_optimze_vec;
    std::vector<float> con_vec;
    for (int j = 0; j < calib_frame_num; j++)
    {
        euler_ang_delt_xyz_after_global_optimze_vec.push_back(vec);
        euler_ang_xyz_after_global_optimze_vec.push_back(euler_ang_vec);
        confidence_after_global_optimze_vec.push_back(con_vec);
        for (int i = 0; i < Result_Optimize_vec[0].size(); i++)
        {
            Eigen::Matrix3f R_delt_xyz_after_global_optimze = Result_gt_vec[j][i].block(0, 0, 3, 3).inverse() * Result_Optimize_vec[j][i].block(0, 0, 3, 3);
            Eigen::Vector3f euler_ang_delt_xyz_after_global_optimze = R_delt_xyz_after_global_optimze.eulerAngles(0, 1, 2);
            Eigen::Matrix3f R_Result = Result_Optimize_vec[j][i].block(0, 0, 3, 3);
            Eigen::Vector3f euler_ang_xyz_after_global_optimze = R_Result.eulerAngles(0, 1, 2);
            euler_ang_delt_xyz_after_global_optimze_vec[j].push_back(euler_ang_delt_xyz_after_global_optimze);
            euler_ang_xyz_after_global_optimze_vec[j].push_back(euler_ang_xyz_after_global_optimze);
            confidence_after_global_optimze_vec[j].push_back(AutoCalib::countConfidence(pcs_feature_vec[j][i], images_feature_vec[j][i], Result_Optimize_vec[j][i], k_vec[j]));
        }
    }
    // for (int j = 0; j < calib_frame_num; j++)
    // {

    //     for (int i = 0; i < euler_ang_delt_xyz_after_global_optimze_vec[0].size(); i++)
    //     {
    //         for (int r = 0; r < 3; ++r)
    //         {
    //             if ((euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] > 0) && (abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] - 3) < 0.2))
    //             {
    //                 euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] = (euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] - M_PI) * 180.0 / M_PI;
    //             }
    //             if ((euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] < 0) && (abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] + 3) < 0.2))
    //             {
    //                 euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] = (euler_ang_delt_xyz_after_global_optimze_vec[j][i][r] + M_PI) * 180.0 / M_PI;
    //             }
    //         }
    //     }
    // }
    for (int j = 0; j < calib_frame_num; j++)
    {

        std::string calibrated_result_path = "../data" + std::to_string(j) + "/result/calibrated_result";
        std::ofstream calibrated_result(calibrated_result_path, std::ios::app);
        calibrated_result << "after optimize" << std::endl;
        calibrated_result << "after_optimize_x_c=[";
        for (int i = 0; i < euler_ang_xyz_after_global_optimze_vec[0].size(); i++)
        {
            calibrated_result << euler_ang_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI << ",";
        }
        calibrated_result << "]" << std::endl;
        calibrated_result << "after_optimize_y_c=[";
        for (int i = 0; i < euler_ang_xyz_after_global_optimze_vec[0].size(); i++)
        {
            calibrated_result << euler_ang_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI << ",";
        }
        calibrated_result << "]" << std::endl;
        calibrated_result << "after_optimize_z_c=[";
        for (int i = 0; i < euler_ang_xyz_after_global_optimze_vec[0].size(); i++)
        {
            calibrated_result << euler_ang_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI << ",";
        }
        calibrated_result << "]" << std::endl;

        calibrated_result << "after_optimize_delt_x_c=[";
        for (int i = 0; i < euler_ang_delt_xyz_after_global_optimze_vec[0].size(); ++i)
        {
            if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI > 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] - M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI << ",";
            }
            else if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI < 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] + M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI << ",";
            }
            else
            {
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][2] * 180.0 / M_PI << ",";
            }
        }
        calibrated_result << "]" << std::endl;

        calibrated_result << "after_optimize_delt_y_c=[";
        for (int i = 0; i < euler_ang_delt_xyz_after_global_optimze_vec[0].size(); ++i)
        {
            if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI > 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] - M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI << ",";
            }
            else if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI < 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] + M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI << ",";
            }
            else
            {
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][1] * 180.0 / M_PI << ",";
            }
        }
        calibrated_result << "]" << std::endl;

        calibrated_result << "after_optimize_delt_z_c=[";
        for (int i = 0; i < euler_ang_delt_xyz_after_global_optimze_vec[0].size(); ++i)
        {
            if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI > 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] - M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI << ",";
            }
            else if (abs(abs(euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI) - 180) < 20 && euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI < 0)
            {
                euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] = euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] + M_PI;
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI << ",";
            }
            else
            {
                calibrated_result << euler_ang_delt_xyz_after_global_optimze_vec[j][i][0] * 180.0 / M_PI << ",";
            }
        }
        calibrated_result << "]" << std::endl
                          << std::endl;

        float confidence_mean = 0;
        calibrated_result << "after_optimize_confidence=[";
        for (int i = 0; i < confidence_after_global_optimze_vec[0].size(); ++i)
        {
            calibrated_result << confidence_after_global_optimze_vec[j][i] << ",";
            confidence_mean += confidence_after_global_optimze_vec[j][i];
        }
        calibrated_result << "]" << std::endl
                          << std::endl;
        confidence_mean /= confidence_after_global_optimze_vec[j].size();
        calibrated_result.close();
    }
}
AutoCalib::AutoCalib(const std::string &ConfigFile, int frame_num)
{
    config = YAML::LoadFile(ConfigFile);
    debug = config["debug_OnOff"].as<bool>();
    frame_cnt = frame_num;
    overlap = config["overlap"].as<bool>();
    add_dis_weight = config["add_dis_weight"].as<bool>();
    rings = config["rings"].as<int>();
    lowerBound = config["lowerBound"].as<float>();
    upperBound = config["upperBound"].as<float>();
    fx = config["fx"].as<float>();
    fy = config["fy"].as<float>();
    cx = config["cx"].as<float>();
    cy = config["cy"].as<float>();
    k1 = config["k1"].as<float>();
    k2 = config["k2"].as<float>();
    p1 = config["p1"].as<float>();
    p2 = config["p2"].as<float>();
    //cv::Mat now_img = cv::Mat(cv::Size(512, 480), CV_8UC3);
    //cv::Mat now_img2 = cv::Mat(cv::Size(512, 480), CV_8UC4);
    Eigen::Matrix4f T_lidar2cam0_unbias = Eigen::Matrix4f::Identity();    //lidar2cam
    Eigen::Matrix4f T_lidar2cam0_bias = Eigen::Matrix4f::Identity();      //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias = Eigen::Matrix4f::Identity();      //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias_last = Eigen::Matrix4f::Identity(); //lidar2cam last
    Eigen::Matrix4f T_lidar2cam2_unbias = Eigen::Matrix4f::Identity();    //lidar2cam
    Eigen::Matrix4f T_cam02cam2 = Eigen::Matrix4f::Identity();            //cam2cam
    T_cam2image << fx, 0.f, cx, 0.f, fy, cy, 0, 0, 1;
    Eigen::Matrix3f rot_icp = Eigen::Matrix3f::Identity();
}
