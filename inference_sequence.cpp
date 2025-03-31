#include <memory> 
#include <chrono> 
#include "utils.h" 
#include "super_glue.h" 
#include "super_point.h" 

int main(int argc, char** argv){ 
    if (argc != 5) { 
        std::cerr << "./superpointglue_sequence config_path model_dir image_folder_absolutely_path output_folder_path" << std::endl; 
        return 0; 
    } 

    std::string config_path = argv[1]; 
    std::string model_dir = argv[2]; 
    std::string image_path = argv[3]; 
    std::string output_path = argv[4]; 
    std::vector<std::string> image_names; 
    GetFileNames(image_path, image_names); 
    Configs configs(config_path, model_dir); 
    int width = configs.superglue_config.image_width; 
    int height = configs.superglue_config.image_height; 

    std::cout << "Building inference engine......" << std::endl; 
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config); 
    if (!superpoint->build()) { 
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl; 
        return 0; 
    } 
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config); 
    if (!superglue->build()) { 
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl; 
        return 0; 
    } 
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl; 

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0; 
    cv::Mat image0 = cv::imread(image_names[0], cv::IMREAD_GRAYSCALE); 
    if(image0.empty()) { 
        std::cerr << "First image in the image folder is empty." << std::endl; 
        return 0; 
    } 
    cv::resize(image0, image0, cv::Size(width, height)); 
    std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl; 
    if (!superpoint->infer(image0, feature_points0)) { 
        std::cerr << "Failed when extracting features from first image." << std::endl; 
        return 0; 
    } 
    std::vector<cv::DMatch> init_matches; 
    superglue->matching_points(feature_points0, feature_points0, init_matches); 
    std::string mkdir_cmd = "mkdir -p " + output_path; 
    system(mkdir_cmd.c_str()); 

    // Begin processing the image sequence
    for (int index = 1; index < image_names.size(); ++index) { 
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1; 
        std::vector<cv::DMatch> superglue_matches; 
        cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_GRAYSCALE); 
        if(image1.empty()) continue; 
        cv::resize(image1, image1, cv::Size(width, height)); 

        std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl; 

        // Time the feature extraction for the second image
        auto start_feature_extraction = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image1, feature_points1)) { 
            std::cerr << "Failed when extracting features from second image." << std::endl; 
            return 0; 
        }
        auto end_feature_extraction = std::chrono::high_resolution_clock::now();
        auto feature_extraction_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_feature_extraction - start_feature_extraction);

        // Time the feature matching
        auto start_matching = std::chrono::high_resolution_clock::now();
        superglue->matching_points(feature_points0, feature_points1, superglue_matches); 
        auto end_matching = std::chrono::high_resolution_clock::now();
        auto matching_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_matching - start_matching);

        std::cout << "Feature extraction time for image " << index << ": " << feature_extraction_duration.count() << " ms" << std::endl;
        std::cout << "Feature matching time for image " << index << ": " << matching_duration.count() << " ms" << std::endl;

        // Prepare keypoints for visualization
        std::vector<cv::KeyPoint> keypoints0, keypoints1; 
        for (size_t i = 0; i < feature_points0.cols(); ++i) { 
            double score = feature_points0(0, i); 
            double x = feature_points0(1, i); 
            double y = feature_points0(2, i); 
            keypoints0.emplace_back(x, y, 8, -1, score); 
        } 
        for (size_t i = 0; i < feature_points1.cols(); ++i) { 
            double score = feature_points1(0, i); 
            double x = feature_points1(1, i); 
            double y = feature_points1(2, i); 
            keypoints1.emplace_back(x, y, 8, -1, score); 
        } 

        // Visualize the matching points
        cv::Mat match_image;
        VisualizeMatching(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
        cv::imwrite(output_path + "/" + std::to_string(index) + ".png", match_image); 
    }

    return 0; 
}

// #include <memory>
// #include <chrono>
// #include "utils.h"
// #include "super_glue.h"
// #include "super_point.h"

// int main(int argc, char** argv){
//     if (argc != 5) {
//         std::cerr << "./superpointglue_sequence config_path model_dir image_folder_absolutely_path output_folder_path" << std::endl;
//         return 0;
//     }

//     std::string config_path = argv[1];
//     std::string model_dir = argv[2];
//     std::string image_path = argv[3];
//     std::string output_path = argv[4];
//     std::vector<std::string> image_names;
//     GetFileNames(image_path, image_names);
//     Configs configs(config_path, model_dir);
//     int width = configs.superglue_config.image_width;
//     int height = configs.superglue_config.image_height;

//     std::cout << "Building inference engine......" << std::endl;
//     auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
//     if (!superpoint->build()) {
//         std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
//         return 0;
//     }
//     auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
//     if (!superglue->build()) {
//         std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
//         return 0;
//     }
//     std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

//     Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;
//     cv::Mat image0 = cv::imread(image_names[0], cv::IMREAD_COLOR);  // 读取为彩色图像
//     if(image0.empty()) {
//         std::cerr << "First image in the image folder is empty." << std::endl;
//         return 0;
//     }
//     cv::resize(image0, image0, cv::Size(width, height));
//     std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
//     if (!superpoint->infer(image0, feature_points0)) {
//         std::cerr << "Failed when extracting features from first image." << std::endl;
//         return 0;
//     }
//     std::vector<cv::DMatch> init_matches;
//     superglue->matching_points(feature_points0, feature_points0, init_matches);
//     std::string mkdir_cmd = "mkdir -p " + output_path;
//     system(mkdir_cmd.c_str());

//     // Begin processing the image sequence
//     for (int index = 1; index < image_names.size(); ++index) {
//         Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1;
//         std::vector<cv::DMatch> superglue_matches;
//         cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_COLOR);  // 读取为彩色图像
//         if(image1.empty()) continue;
//         cv::resize(image1, image1, cv::Size(width, height));

//         std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;

//         // Time the feature extraction for the second image
//         auto start_feature_extraction = std::chrono::high_resolution_clock::now();
//         if (!superpoint->infer(image1, feature_points1)) {
//             std::cerr << "Failed when extracting features from second image." << std::endl;
//             return 0;
//         }
//         auto end_feature_extraction = std::chrono::high_resolution_clock::now();
//         auto feature_extraction_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_feature_extraction - start_feature_extraction);

//         // Time the feature matching
//         auto start_matching = std::chrono::high_resolution_clock::now();
//         superglue->matching_points(feature_points0, feature_points1, superglue_matches);
//         auto end_matching = std::chrono::high_resolution_clock::now();
//         auto matching_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_matching - start_matching);

//         std::cout << "Feature extraction time for image " << index << ": " << feature_extraction_duration.count() << " ms" << std::endl;
//         std::cout << "Feature matching time for image " << index << ": " << matching_duration.count() << " ms" << std::endl;

//         // Prepare keypoints for visualization
//         std::vector<cv::KeyPoint> keypoints0, keypoints1;
//         for (size_t i = 0; i < feature_points0.cols(); ++i) {
//             double score = feature_points0(0, i);
//             double x = feature_points0(1, i);
//             double y = feature_points0(2, i);
//             keypoints0.emplace_back(x, y, 8, -1, score);
//         }
//         for (size_t i = 0; i < feature_points1.cols(); ++i) {
//             double score = feature_points1(0, i);
//             double x = feature_points1(1, i);
//             double y = feature_points1(2, i);
//             keypoints1.emplace_back(x, y, 8, -1, score);
//         }

//         // Visualize the matching points (colored visualization)
//         cv::Mat match_image;
//         // No need to convert to color, images are already in color
//         // Draw the matches with color lines
//         cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image,
//                         cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//         // Save the matched image
//         cv::imwrite(output_path + "/" + std::to_string(index) + "_colored.png", match_image);
//     }

//     return 0;
// }
