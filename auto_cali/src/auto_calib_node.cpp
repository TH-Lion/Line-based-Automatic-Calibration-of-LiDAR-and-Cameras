#include "auto_calib.h"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "auto_calib");
	int calib_frame_num;

	cout << "please input calib_frame_num：    ";
	std::cin >> calib_frame_num;

	std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> pcs_vec;
	std::vector<std::vector<cv::Mat>> images_vec;
	std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>> pcs_feature_vec;
	std::vector<std::vector<cv::Mat>> images_feature_vec;
	std::vector<Eigen::Matrix4f> pcs_current_guess_vec;
	std::vector<Eigen::Matrix4f> images_current_guess_vec;
	std::vector<std::vector<Eigen::Matrix4f>> pcs_calib_vec;

	std::vector<std::vector<Eigen::Matrix4f>> images_calib_vec;
	std::vector<Eigen::Matrix3f> k_vec;
	std::vector<std::vector<Eigen::Matrix4f>> Result_gt_vec;
	std::vector<std::vector<Eigen::Matrix4f>> Calibrated_Result_vec;
	std::vector<bool> overlap_vec;
	bool add_dis_weight;
	for (int calib_frame_cnt = 0; calib_frame_cnt < calib_frame_num; calib_frame_cnt++)
	{

		AutoCalib app("../config" + std::to_string(calib_frame_cnt) + ".yaml", calib_frame_cnt);
		app.Run();
		pcs_vec.push_back(app.get_in_pcs());
		images_vec.push_back(app.get_in_images());
		pcs_feature_vec.push_back(app.get_in_pcs_feature());
		images_feature_vec.push_back(app.get_in_images_feature());
		pcs_current_guess_vec.push_back(app.get_in_pcs_current_guess());
		//images_current_guess_vec.push_back(app.get_in_images_current_guess());
		k_vec.push_back(app.get_in_k());
		Result_gt_vec.push_back(app.get_in_result_gt_vec());
		Calibrated_Result_vec.push_back(app.get_in_calibrated_result_vec());
		overlap_vec.push_back(app.get_in_overlap());
		add_dis_weight = app.get_in_add_dis_weight();
	}


	std::vector<Eigen::Matrix4f> vc;
	pcs_calib_vec.push_back(vc);
	for (int i = 0; i < Calibrated_Result_vec[0].size(); i++)
	{
		pcs_calib_vec[0].push_back(Eigen::Matrix4f::Identity());
	}

	if (calib_frame_num > 1 )
	{

		for (int calib_frame_cnt = 1; calib_frame_cnt < calib_frame_num; calib_frame_cnt++)
		{

			std::vector<Eigen::Matrix4f> vc;
			pcs_calib_vec.push_back(vc);

			// AutoCalib::PerformNdt(pcs_vec[0], pcs_vec[calib_frame_cnt], calib_frame_cnt,
			// 					  pcs_current_guess_vec[calib_frame_cnt], pcs_calib_vec[calib_frame_cnt]);
			AutoCalib::PerformNdt(pcs_vec[0], pcs_vec[calib_frame_cnt], calib_frame_cnt,
								  pcs_current_guess_vec[calib_frame_cnt], pcs_calib_vec[calib_frame_cnt]);
			//AutoCalib::PerformICP(images_vec[0],images_vec[calib_frame_cnt], calib_frame_cnt,images_current_guess_vec[calib_frame_cnt],images_calib_vec[calib_frame_cnt]);
		}

		AutoCalib::GlobalOptimize(pcs_calib_vec, Calibrated_Result_vec, pcs_vec, images_vec,
								  pcs_feature_vec, images_feature_vec, add_dis_weight, overlap_vec,
								  calib_frame_num, k_vec,Result_gt_vec);
	}
	return 0;
}
