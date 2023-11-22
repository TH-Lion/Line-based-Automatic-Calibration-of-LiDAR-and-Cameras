#ifndef PROJECT_AUTO_CALIB_H
#define PROJECT_AUTO_CALIB_H

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
#include <pcl/registration/icp.h>
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
#include <pcl/registration/icp.h>

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
#include <pcl/registration/ndt.h>

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
#include "g2o_type.h"

class AutoCalib
{
private:
    YAML::Node config;
    bool debug;
    bool overlap, add_dis_weight;

    int frame_cnt;
    int rings;
    float lowerBound; //< the vertical angle of the first scan ring
    float upperBound; //< the vertical angle of the last scan ring
    float factor;

    float dis_threshold;
    float angle_threshold;
    std::vector<cv::Mat> images_withouthist;

    int canny_threshold_mini, canny_threshold_max, normalize_config, normalize_config_thin;

    cv::Mat now_img;
    cv::Mat now_img2;
    float cx, cy, fx, fy,k1,k2,p1,p2;
    Eigen::Matrix4f T_lidar2cam0_unbias;                        //lidar2cam
    Eigen::Matrix4f T_lidar2cam0_bias;                          //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias;                          //lidar2cam
    Eigen::Matrix4f T_lidar2cam2_bias_last;                     //lidar2cam last
    Eigen::Matrix4f T_lidar2cam2_unbias;                        //lidar2cam
    Eigen::Matrix4f T_cam02cam2;                                //cam2cam
    Eigen::Matrix3f T_cam2image;                                //cam2image
    Eigen::Matrix<float, 3, 4> T_lidar2cam_top3, T_lidar2image; //lida2image=T*(T_cam02cam2)*T_cam2image
    Eigen::Matrix3f rot_icp;
    std::vector<cv::Mat> gray_image_vec;
    std::vector<Eigen::Matrix4f> T_lidar2cam2_bias_vec;
    std::vector<Eigen::Matrix4f> result_gt_vec;
    std::vector<Eigen::Matrix4f> calibrated_result_vec;

    std::vector<cv::Mat> in_images;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> in_pcs;
    std::vector<cv::Mat> in_images_feature;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> in_pcs_feature;

    std::vector<std::string> result_file;
    geometry_msgs::TransformStamped tf_trans;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> filtered_pc;
    static pcl::PointCloud<pcl::PointXYZI> map;
    struct PointXYZIRL
    {
        PCL_ADD_POINT4D; // quad-word XYZ
        float intensity;
        uint16_t label;                 ///< point label
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    } EIGEN_ALIGN16;

    struct PointXYZIA
    {
        PCL_ADD_POINT4D; // quad-word XYZ
        float intensity;
        float cosangle;
        float distance;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    } EIGEN_ALIGN16;

    struct Detected_Obj
    {
        jsk_recognition_msgs::BoundingBox bounding_box_;

        pcl::PointXYZ min_point_;
        pcl::PointXYZ max_point_;
        pcl::PointXYZ centroid_;
    };
    bool getPointcloud(std::string filename, pcl::PointCloud<pcl::PointXYZI>::Ptr ptcloud);
    void getOxtsData(std::string filename, std::vector<float> &oxts_vec);

    static bool point_cmp(pcl::PointXYZI a, pcl::PointXYZI b);
    void point_cb(pcl::PointCloud<pcl::PointXYZI>::Ptr data, pcl::PointCloud<pcl::PointXYZI>::Ptr final_no_ground);

    int getData(std::string txtName, std::string folderName, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pointclouds,
                std::vector<cv::Mat> &images, std::vector<float> &oxts_vec);

    int recoverPose(cv::InputArray &E, cv::InputArray &_points1, cv::InputArray &_points2, cv::OutputArray &_R,
                    cv::OutputArray &_t, double &focal, cv::Point2d &pp, cv::InputOutputArray &_mask);

    void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                              std::vector<cv::KeyPoint> keypoints_2,
                              std::vector<cv::DMatch> matches,
                              cv::Mat &R, cv::Mat &t);

    void send_transform_thread();

    void down_sample_pc(pcl::PointCloud<pcl::PointXYZI>::Ptr &in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out, double leaf_size);

    bool pointcmp(PointXYZIA a, PointXYZIA b);

    void extract_pc_feature_6(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_feature);

    float dist2Point(int x1, int y1, int x2, int y2);

    void extract_image_feature(cv::Mat &img, cv::Mat &image2, std::vector<cv::line_descriptor::KeyLine> &keylines, std::vector<cv::line_descriptor::KeyLine> &keylines2,
                               cv::Mat &outimg, cv::Mat &outimg_thin);

    void extractFeature(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs, std::vector<cv::Mat> images, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &pc_feature,
                        std::vector<cv::Mat> &distance_image, std::vector<cv::Mat> &distance_image_thin);

    float countScore(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                     Eigen::Matrix4f RT, Eigen::Matrix3f camera_param);

    bool isWhiteEnough(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                       Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, bool fine_result, float &score);

    void filterUnusedPoiintCloudFeature(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                        Eigen::Matrix4f RT, Eigen::Matrix3f camera_param, pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature_filtered);

public:
    void Run();
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> get_in_pcs();
    std::vector<cv::Mat> get_in_images();
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> get_in_pcs_feature();
    std::vector<cv::Mat> get_in_images_feature();
    Eigen::Matrix4f get_in_pcs_current_guess();
    Eigen::Matrix4f get_in_images_current_guess();
    Eigen::Matrix3f get_in_k();
    bool get_in_overlap();
    bool get_in_add_dis_weight();
    std::vector<Eigen::Matrix4f> get_in_result_gt_vec();
    std::vector<Eigen::Matrix4f> get_in_calibrated_result_vec();
    static Sophus::SE3d toSE3d(Eigen::Matrix4f &T);
    static Eigen::Matrix4f toMatrix4f(Eigen::Matrix4d s);
    static void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                                     std::vector<cv::KeyPoint> &keypoints_1,
                                     std::vector<cv::KeyPoint> &keypoints_2,
                                     std::vector<cv::DMatch> &matches);

    static float countConfidence(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_feature, const cv::Mat distance_image,
                                 Eigen::Matrix4f RT, Eigen::Matrix3f camera_param);

    static void project2image(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, cv::Mat raw_image, cv::Mat &output_image, Eigen::Matrix4f RT, Eigen::Matrix3f camera_param);

    static void PerformNdt(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_parent_cloud_vec,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &in_child_cloud_vec, int calib_frame_cnt,
                           Eigen::Matrix4f &pcs_current_guess, std::vector<Eigen::Matrix4f> &pcs_calib);
    //static void PerformICP(std::vector<cv::Mat> & in_parent_images_vec, std::vector<cv::Mat>& in_child_images_vec, int calib_frame_cnt,Eigen::Matrix4f & images_current_guess,std::vector<Eigen::Matrix4f > & images_calib);
    static void GlobalOptimize(std::vector<std::vector<Eigen::Matrix4f>> &pcs_calib_vec, std::vector<std::vector<Eigen::Matrix4f>> &Calibrated_Result_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_vec,
                               std::vector<std::vector<cv::Mat>> &images_vec,
                               std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> &pcs_feature_vec,
                               std::vector<std::vector<cv::Mat>> &images_feature_vec, bool &add_dis_weight,
                               std::vector<bool> &overlap_vec,
                               int &calib_frame_num, std::vector<Eigen::Matrix3f> &k_vec, std::vector<std::vector<Eigen::Matrix4f>> &Result_gt_vec);
    AutoCalib(const std::string &ConfigFile, int frame_num);
};

#endif //PROJECT_PROJECT_AUTO_CALIB_H
