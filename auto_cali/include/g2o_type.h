

#ifndef G2O_TYPES_H
#define G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <sophus/se3.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
//#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/console/time.h>
//#include <pcl/registration/icp.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <thread>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
//#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/surface_matching/icp.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

static cv::Point2d pixel2Cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
static double FitnessScore(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_1,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_2, Eigen::Matrix4d &T)
{
    //pcl::console::TicToc time; //申明时间记录
    //time.tic();                //time.tic开始  time.toc结束时间
    double fitness_score = 0.0;
    double max_range = std::numeric_limits<double>::max();
    // Transform the input dataset using the final transformation
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*pc_2, *output_cloud, T);
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;
    tree_.setInputCloud(pc_1);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    // For each point in the source dataset
    int nr = 0;
    for (size_t i = 0; i < output_cloud->points.size(); ++i)
    {
        // Find its nearest neighbor in the target
        tree_.nearestKSearch(output_cloud->points[i], 1, nn_indices, nn_dists);

        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= max_range)
        {
            // Add to the fitness score
            fitness_score += nn_dists[0];
            nr++;
        }
    }
    //std::cout << time.toc() << " ms" << std::endl;

    if (nr > 0)
        return (fitness_score / nr);
    else
        return (std::numeric_limits<double>::max());
}

// static void find_feature_matche(const cv::Mat &img_1, const cv::Mat &img_2,
//                                 std::vector<cv::KeyPoint> &keypoints_1,
//                                 std::vector<cv::KeyPoint> &keypoints_2,
//                                 std::vector<cv::DMatch> &matches)
// {
//     //pcl::console::TicToc time; //申明时间记录
//     //time.tic();                //time.tic开始  time.toc结束时间
//     //-- 初始化
//     cv::Mat descriptors_1, descriptors_2;
//     // used in OpenCV3
//     cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
//     cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
//     // use this if you are in OpenCV2
//     // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//     // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
//     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
//     //-- 第一步:检测 Oriented FAST 角点位置
//     detector->detect(img_1, keypoints_1);
//     detector->detect(img_2, keypoints_2);

//     // #define ROI_LEFT 0
//     // #define ROI_TOP images[ppp].rows/4
//     // #define ROI_WIDTH images[ppp].cols
//     // #define ROI_HEIGHT images[ppp].rows/2

//     // std::vector<cv::KeyPoint> keypoints_1_temp, keypoints_2_temp;

//     // for(int i = 0; i < keypoints_1.size(); ++i)
//     // {
//     //     if(keypoints_1[i].pt.x > 0+20 && keypoints_1[i].pt.x < 0+img_1.cols-20
//     //        && keypoints_1[i].pt.y > img_1.rows/2+20 && keypoints_1[i].pt.y < img_1.rows/2+img_1.rows/2-20)
//     //     {
//     //         keypoints_1_temp.push_back(keypoints_1[i]);
//     //     }
//     // }
//     // for(int i = 0; i < keypoints_2.size(); ++i)
//     // {
//     //     if(keypoints_2[i].pt.x > 0+20 && keypoints_2[i].pt.x < 0+img_2.cols-20
//     //        && keypoints_2[i].pt.y > img_2.rows/2+20 && keypoints_2[i].pt.y < img_2.rows/2+img_2.rows/2-20)
//     //     {
//     //         keypoints_2_temp.push_back(keypoints_2[i]);
//     //     }
//     // }
//     // keypoints_1.clear();
//     // keypoints_2.clear();
//     // keypoints_1 = keypoints_1_temp;
//     // keypoints_2 = keypoints_2_temp;

//     //-- 第二步:根据角点位置计算 BRIEF 描述子
//     descriptor->compute(img_1, keypoints_1, descriptors_1);
//     descriptor->compute(img_2, keypoints_2, descriptors_2);

//     //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//     std::vector<cv::DMatch> match;
//     //BFMatcher matcher ( NORM_HAMMING );
//     matcher->match(descriptors_1, descriptors_2, match);

//     //-- 第四步:匹配点对筛选
//     double min_dist = 10000, max_dist = 0;

//     //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//     for (int i = 0; i < descriptors_1.rows; i++)
//     {
//         double dist = match[i].distance;
//         if (dist < min_dist)
//             min_dist = dist;
//         if (dist > max_dist)
//             max_dist = dist;
//     }

//     //printf("-- Max dist : %f \n", max_dist);
//     //printf("-- Min dist : %f \n", min_dist);

//     //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//     for (int i = 0; i < descriptors_1.rows; i++)
//     {
//         if (match[i].distance <= std::max(2 * min_dist, 30.0))
//         {
//             matches.push_back(match[i]);
//         }
//     }
//     //std::cout << time.toc() << " ms" << std::endl;
// }

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4],
            update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<1, double, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectionPoseOnly(const pcl::PointXYZI &pc_feature,
                           const cv::Mat &distance_image, const Eigen::Matrix3f &K, bool add_dis_weight)
        : _pc_feature(pc_feature), _distance_image(distance_image), _K(K), _add_dis_weight(add_dis_weight) {}

    virtual void computeError() override
    {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Matrix3d _K_tmp;
        _K_tmp(0, 0) = _K(0, 0);
        _K_tmp(0, 1) = _K(0, 1);
        _K_tmp(0, 2) = _K(0, 2);

        _K_tmp(1, 0) = _K(1, 0);
        _K_tmp(1, 1) = _K(1, 1);
        _K_tmp(1, 2) = _K(1, 2);

        _K_tmp(2, 0) = _K(2, 0);
        _K_tmp(2, 1) = _K(2, 1);
        _K_tmp(2, 2) = _K(2, 2);

        double score = 0;
        //std::cout<<"test"<<std::endl;
        Eigen::Matrix<double, 3, 4> RT_TOP3, RT_X_CAM;

        RT_TOP3 = T.matrix().topRows(3);
        RT_X_CAM = _K_tmp * RT_TOP3;

        //count Score

        double one_score = 0;
        int points_num = 0;

        Eigen::Vector4d raw_point;
        Eigen::Vector3d trans_point3;

        raw_point(0, 0) = _pc_feature.x;
        raw_point(1, 0) = _pc_feature.y;
        raw_point(2, 0) = _pc_feature.z;
        raw_point(3, 0) = 1;
        trans_point3 = RT_X_CAM * raw_point;
        int x = (int)(trans_point3(0, 0) / trans_point3(2, 0));
        int y = (int)(trans_point3(1, 0) / trans_point3(2, 0));
        if (x < 0 || x > (_distance_image.cols - 1) || y < 0 || y > (_distance_image.rows - 1))
            one_score = 0;
        else if (_pc_feature.intensity < 0 || _distance_image.at<uchar>(y, x) < 0)
        {
            one_score = 0;
            std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
        }
        else
        {
            double pt_dis = pow(_pc_feature.x * _pc_feature.x + _pc_feature.y * _pc_feature.y + _pc_feature.z * _pc_feature.z, double(1.0 / 2.0));
            if (_add_dis_weight)
            {
                // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
                if (abs(_pc_feature.intensity - 0.1) < 0.2)
                {
                    one_score = (_distance_image.at<uchar>(y, x) / pt_dis * 2) * 3 / 255.0;
                }
                else
                {
                    one_score = (_distance_image.at<uchar>(y, x) / pt_dis * 2) / 255.0;
                }
            }
            else
            {
                one_score = _distance_image.at<uchar>(y, x) / 255.0;
            }
        }

        _error(0, 0) = _measurement - one_score;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override { return true; }

private:
    pcl::PointXYZI _pc_feature;
    cv::Mat _distance_image;
    Eigen::Matrix3f _K;
    bool _add_dis_weight;
};

class EdgePRScale : public g2o::BaseMultiEdge<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgePRScale(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature_parent, const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature_child,
                const cv::Mat &distance_image_parent, const cv::Mat &distance_image_child,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_parent, const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_child,
                const cv::Mat &image_parent, const cv::Mat &image_child,
                const Eigen::Matrix3f &K1, const Eigen::Matrix3f &K2, bool add_dis_weight, bool overlap, int calib_frame,
                const std::vector<cv::KeyPoint> &keypoints_first, const std::vector<cv::KeyPoint> &keypoints_second,
                const std::vector<cv::DMatch> &matches,pcl::PointCloud<pcl::PointXYZI>::Ptr &filtered_cloud)
        : _pc_feature_parent(pc_feature_parent), _pc_feature_child(pc_feature_child),
          _distance_image_parent(distance_image_parent), _distance_image_child(distance_image_child),
          _pc_parent(pc_parent), _pc_child(pc_child),
          _image_parent(image_parent), _image_child(image_child),
          _K1(K1), _K2(K2), _add_dis_weight(add_dis_weight), _overlap(overlap), _calib_frame(calib_frame),
        _keypoints_first(keypoints_first), _keypoints_second(keypoints_second), _matches(matches),_filtered_cloud(filtered_cloud) { resize(3); }

    virtual void computeError() override
    {
        const VertexPose *v1 = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T1 = v1->estimate();
        Eigen::Matrix3d _K1_tmp;
        _K1_tmp(0, 0) = _K1(0, 0);
        _K1_tmp(0, 1) = _K1(0, 1);
        _K1_tmp(0, 2) = _K1(0, 2);

        _K1_tmp(1, 0) = _K1(1, 0);
        _K1_tmp(1, 1) = _K1(1, 1);
        _K1_tmp(1, 2) = _K1(1, 2);

        _K1_tmp(2, 0) = _K1(2, 0);
        _K1_tmp(2, 1) = _K1(2, 1);
        _K1_tmp(2, 2) = _K1(2, 2);

        double score1 = 0;
        Eigen::Matrix<double, 3, 4> RT1_TOP3, RT1_X_CAM;
        RT1_TOP3 = T1.matrix().topRows(3);
        RT1_X_CAM = _K1_tmp * RT1_TOP3;
        int points_num = 0;
        //count Score

        double one_score1 = 0;
        for (int i = 0; i < _pc_feature_parent->size(); i++)
        {
            Eigen::Vector4d raw1_point;
            Eigen::Vector3d trans1_point3;
            pcl::PointXYZI r;
            r = _pc_feature_parent->points[i];

            raw1_point(0, 0) = r.x;
            raw1_point(1, 0) = r.y;
            raw1_point(2, 0) = r.z;
            raw1_point(3, 0) = 1;
            trans1_point3 = RT1_X_CAM * raw1_point;
            int x1 = (int)(trans1_point3(0, 0) / trans1_point3(2, 0));
            int y1 = (int)(trans1_point3(1, 0) / trans1_point3(2, 0));
            if (x1 < 0 || x1 > (_distance_image_parent.cols - 1) || y1 < 0 || y1 > (_distance_image_parent.rows - 1))
                continue;
            if (r.intensity < 0 || _distance_image_parent.at<uchar>(y1, x1) < 0)
            {

                std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
                break;
            }
            points_num++;
            double pt_dis1 = pow(r.x * r.x + r.y * r.y + r.z * r.z, double(1.0 / 2.0));
            if (_add_dis_weight)
            {
                // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
                if (abs(r.intensity - 0.1) < 0.2)
                {
                    one_score1 += (_distance_image_parent.at<uchar>(y1, x1) / pt_dis1 * 2) * 3;
                }
                else
                {
                    one_score1 += (_distance_image_parent.at<uchar>(y1, x1) / pt_dis1 * 2);
                }
            }
            else
            {
                one_score1 += _distance_image_parent.at<uchar>(y1, x1);
            }
        }

        one_score1 = one_score1 / (_calib_frame - 1);
        score1 = one_score1 / 255.0 / points_num;
        points_num = 0;

        const VertexPose *v2 = static_cast<VertexPose *>(_vertices[2]);
        Sophus::SE3d T2 = v2->estimate();
        Eigen::Matrix3d _K2_tmp;
        _K2_tmp(0, 0) = _K2(0, 0);
        _K2_tmp(0, 1) = _K2(0, 1);
        _K2_tmp(0, 2) = _K2(0, 2);

        _K2_tmp(1, 0) = _K2(1, 0);
        _K2_tmp(1, 1) = _K2(1, 1);
        _K2_tmp(1, 2) = _K2(1, 2);

        _K2_tmp(2, 0) = _K2(2, 0);
        _K2_tmp(2, 1) = _K2(2, 1);
        _K2_tmp(2, 2) = _K2(2, 2);

        double score2 = 0;
        Eigen::Matrix<double, 3, 4> RT2_TOP3, RT2_X_CAM;
        RT2_TOP3 = T2.matrix().topRows(3);
        RT2_X_CAM = _K2_tmp * RT2_TOP3;

        //count Score

        double one_score2 = 0;
        for (int i = 0; i < _pc_feature_child->size(); i++)
        {
            Eigen::Vector4d raw2_point;
            Eigen::Vector3d trans2_point3;
            pcl::PointXYZI r;
            r = _pc_feature_child->points[i];
            raw2_point(0, 0) = r.x;
            raw2_point(1, 0) = r.y;
            raw2_point(2, 0) = r.z;
            raw2_point(3, 0) = 1;
            trans2_point3 = RT2_X_CAM * raw2_point;
            int x2 = (int)(trans2_point3(0, 0) / trans2_point3(2, 0));
            int y2 = (int)(trans2_point3(1, 0) / trans2_point3(2, 0));
            if (x2 < 0 || x2 > (_distance_image_child.cols - 1) || y2 < 0 || y2 > (_distance_image_child.rows - 1))
                continue;
            if (r.intensity < 0 || _distance_image_parent.at<uchar>(y2, x2) < 0)
            {

                std::cout << "\033[33mError: has intensity<0\033[0m" << std::endl;
                break;
            }
            points_num++;
            double pt_dis2 = pow(r.x * r.x + r.y * r.y + r.z * r.z, double(1.0 / 2.0));
            if (_add_dis_weight)
            {
                // one_score +=  (distance_image.at<uchar>(y, x) * sqrt(pc_feature->points[j].intensity));
                if (abs(r.intensity - 0.1) < 0.2)
                {
                    one_score2 += (_distance_image_child.at<uchar>(y2, x2) / pt_dis2 * 2) * 3;
                }
                else
                {
                    one_score2 += (_distance_image_child.at<uchar>(y2, x2) / pt_dis2 * 2);
                }
            }
            else
            {
                one_score2 += _distance_image_child.at<uchar>(y2, x2);
            }
        }

        score2 = one_score2 / 255.0 / points_num;

        double score3 = 0;
        const VertexPose *v3 = static_cast<VertexPose *>(_vertices[1]);
        Sophus::SE3d T3 = v3->estimate();
        Eigen::Matrix4d T3_matrix = T3.matrix();
        // pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
        // approximate_voxel_filter.setLeafSize(0.05, 0.05, 0.05);
        // approximate_voxel_filter.setInputCloud(_pc_child);
        // approximate_voxel_filter.filter(*filtered_cloud);

        // pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

        // ndt.setTransformationEpsilon(0.1);
        // ndt.setStepSize(0.1);
        // ndt.setResolution(0.5);

        // ndt.setMaximumIterations(0);

        // ndt.setInputSource(filtered_cloud);
        // ndt.setInputTarget(_pc_parent);

        // pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        // Eigen::Matrix4f E;
        // E(0, 0) = T3_matrix(0, 0);
        // E(0, 1) = T3_matrix(0, 1);
        // E(0, 2) = T3_matrix(0, 2);
        // E(0, 3) = T3_matrix(0, 3);
        // E(1, 0) = T3_matrix(1, 0);
        // E(1, 1) = T3_matrix(1, 1);
        // E(1, 2) = T3_matrix(1, 2);
        // E(1, 3) = T3_matrix(1, 3);
        // E(2, 0) = T3_matrix(2, 0);
        // E(2, 1) = T3_matrix(2, 1);
        // E(2, 2) = T3_matrix(2, 2);
        // E(2, 3) = T3_matrix(2, 3);
        // E(3, 0) = T3_matrix(3, 0);
        // E(3, 1) = T3_matrix(3, 1);
        // E(3, 2) = T3_matrix(3, 2);
        // E(3, 3) = T3_matrix(3, 3);
        // ndt.align(*output_cloud, E);
        // score3 = ndt.getFitnessScore();
        // std::cout << "score3"<<score3 << std::endl;

        score3 = FitnessScore(_pc_feature_parent, _pc_feature_child, T3_matrix);

        Eigen::Matrix4d E4 = T2.matrix() * T3.matrix().inverse() * T1.matrix().inverse();

        double score4 = 0;
        if (_overlap = true)
        {

            // std::vector<cv::KeyPoint> keypoints_first, keypoints_second;
            // std::vector<cv::DMatch> matches;
            // find_feature_matche(_image_parent, _image_child, keypoints_first, keypoints_second, matches);
            //std::cout << "一共找到 " << matches.size() << " 组匹配点" << std::endl;
            if (_matches.size() < 9)
            {
                std::cout << "匹配点太少!!!!!!!!" << std::endl;
            }
            else
            {
                cv::Mat Mat_K2 = (cv::Mat_<double>(3, 3) << _K2_tmp(0, 0), _K2_tmp(0, 1), _K2_tmp(0, 2), _K2_tmp(1, 0), _K2_tmp(1, 1), _K2_tmp(1, 2), _K2_tmp(2, 0), _K2_tmp(2, 1), _K2_tmp(2, 2));
                cv::Mat Mat_K1 = (cv::Mat_<double>(3, 3) << _K1_tmp(0, 0), _K1_tmp(0, 1), _K1_tmp(0, 2), _K1_tmp(1, 0), _K1_tmp(1, 1), _K1_tmp(1, 2), _K1_tmp(2, 0), _K1_tmp(2, 1), _K1_tmp(2, 2));
                cv::Mat R = (cv::Mat_<double>(3, 3) << E4(0, 0), E4(0, 1), E4(0, 2), E4(1, 0), E4(1, 1), E4(1, 2), E4(2, 0), E4(2, 1), E4(2, 2));
                cv::Mat t = (cv::Mat_<double>(3, 1) << E4(0, 3), E4(1, 3), E4(2, 3));
                cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                               t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                               -t.at<double>(1, 0), t.at<double>(0, 0), 0);
                for (cv::DMatch m : _matches)
                {
                    cv::Point2d pt1 = pixel2Cam(_keypoints_first[m.queryIdx].pt, Mat_K1);
                    cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
                    cv::Point2d pt2 = pixel2Cam(_keypoints_second[m.trainIdx].pt, Mat_K2);
                    cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
                    cv::Mat d = y2.t() * Mat_K2.inv().t() * t_x * R * Mat_K1.inv() * y1;
                    score4 += pow(d.at<double>(0, 0)*d.at<double>(0, 0),double(1.0 / 2.0));                    
                }
            }
        }

        _error(0, 0) = _measurement - score1 - score2  + score3 + score4;
        //std::cout<<score1<<" "<<score2<<" "<<score3<<" "<<score4<<" "<<_error(0, 0)<<std::endl;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override { return true; }

private:
    pcl::PointCloud<pcl::PointXYZI>::Ptr _pc_feature_parent, _pc_feature_child;

    cv::Mat _distance_image_parent, _distance_image_child;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _pc_parent, _pc_child, _filtered_cloud;

    std::vector<cv::KeyPoint> _keypoints_first, _keypoints_second;
    std::vector<cv::DMatch> _matches;

    cv::Mat _image_parent, _image_child;
    Eigen::Matrix3f _K1, _K2;
    bool _add_dis_weight, _overlap;
    int _calib_frame;
};
#endif // G2O_TYPES_H
