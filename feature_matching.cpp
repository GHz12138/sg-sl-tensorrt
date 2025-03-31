#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>

void VisualizeMatching(const cv::Mat &image0, const std::vector<cv::KeyPoint> &keypoints0,
                       const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints1,
                       const std::vector<cv::DMatch> &matches, cv::Mat &output_image,
                       const std::string &algorithm_name, double cost_time = -1) {
    if (image0.size != image1.size) return; // 确保输入图像尺寸一致

    // 画匹配点
    cv::drawMatches(image0, keypoints0, image1, keypoints1, matches, output_image, 
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));

    // 自适应缩放因子
    double sc = std::max(std::min(image0.rows / 640.0, 2.0), 0.5); // 确保最小缩放不低于 0.5
    double font_scale = std::max(0.5, 1.2 * sc); // 确保字体不会太小
    int ht = std::max(20, static_cast<int>(42 * sc)); // 确保行间距不会过小
    int thickness = std::max(1, static_cast<int>(2 * sc)); // 线条厚度适配

    // 标题信息
    cv::putText(output_image, algorithm_name, cv::Point(int(8 * sc), ht), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 0, 0), thickness * 2, cv::LINE_AA);
    cv::putText(output_image, algorithm_name, cv::Point(int(8 * sc), ht), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    // 关键点信息
    std::string feature_points_str = "Keypoints: " + std::to_string(keypoints0.size()) + ":" + std::to_string(keypoints1.size());
    cv::putText(output_image, feature_points_str, cv::Point(int(8 * sc), ht * 2), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 0, 0), thickness * 2, cv::LINE_AA);
    cv::putText(output_image, feature_points_str, cv::Point(int(8 * sc), ht * 2), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    // 匹配点信息
    std::string match_points_str = "Matches: " + std::to_string(matches.size());
    cv::putText(output_image, match_points_str, cv::Point(int(8 * sc), ht * 3), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 0, 0), thickness * 2, cv::LINE_AA);
    cv::putText(output_image, match_points_str, cv::Point(int(8 * sc), ht * 3), 
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    // FPS 信息
    if (cost_time != -1) {
        std::string time_str = "FPS: " + std::to_string(1000.0 / cost_time);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), 
                    cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 0, 0), thickness * 2, cv::LINE_AA);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), 
                    cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./feature_matching image_folder output_folder" << std::endl;
        return 0;
    }

    std::string image_path = argv[1];
    std::string output_path = argv[2];
    
    std::vector<std::string> image_names;
    if (std::filesystem::is_regular_file(image_path)) {
        image_names.push_back(image_path);
    } else {
        cv::glob(image_path, image_names);
    }

    if (image_names.empty()) {
        std::cerr << "No images found in the provided folder." << std::endl;
        return 0;
    }

    std::filesystem::create_directories(output_path);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500); // 限制最多 1000 个 ORB 关键点
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(500); // 限制最多 1000 个 SIFT 关键点

    for (int index = 1; index < image_names.size(); ++index) {
        cv::Mat image0 = cv::imread(image_names[index - 1], cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_GRAYSCALE);
        
        if (image0.empty() || image1.empty()) {
            std::cerr << "Error reading images at index " << index - 1 << " and " << index << std::endl;
            continue;
        }
        
        cv::resize(image0, image0, cv::Size(320, 240));
        cv::resize(image1, image1, cv::Size(320, 240));
        
        std::vector<cv::KeyPoint> keypoints0, keypoints1;
        cv::Mat descriptors0, descriptors1;
        auto start_time = std::chrono::high_resolution_clock::now();
        orb->detectAndCompute(image0, cv::Mat(), keypoints0, descriptors0);
        orb->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
        auto end_time = std::chrono::high_resolution_clock::now();
        double cost_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors0, descriptors1, matches);

        cv::Mat match_image;
        VisualizeMatching(image0, keypoints0, image1, keypoints1, matches, match_image, "ORB Feature Matching", -1);
        cv::imwrite(output_path + "/ORB_" + std::to_string(index) + ".png", match_image);

        start_time = std::chrono::high_resolution_clock::now();
        sift->detectAndCompute(image0, cv::Mat(), keypoints0, descriptors0);
        sift->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
        end_time = std::chrono::high_resolution_clock::now();
        cost_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        cv::FlannBasedMatcher sift_matcher;
        sift_matcher.match(descriptors0, descriptors1, matches);

        VisualizeMatching(image0, keypoints0, image1, keypoints1, matches, match_image, "SIFT Feature Matching", -1);
        cv::imwrite(output_path + "/SIFT_" + std::to_string(index) + ".png", match_image);
    }

    return 0;
}