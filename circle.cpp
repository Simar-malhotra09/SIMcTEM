#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

struct Circle {
    int id;
    int x, y;
    int radius;
    int intensity;
    cv::Scalar color;
    bool overlapping;
};

struct DatasetConfig {
    int num_samples;
    int img_width;
    int img_height;
    int n_circles_min;
    int n_circles_max;
    int radius_min;
    int radius_max;
    int del_count_min;
    int del_count_max;
    std::string output_dir;
    float train_split;
    float val_split;
    bool add_realistic_texture;
    bool add_background_noise;
};

void create_directory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

void create_directory_structure(const std::string& base_dir) {
    create_directory(base_dir);
    create_directory(base_dir + "/train");
    create_directory(base_dir + "/train/images");
    create_directory(base_dir + "/train/masks");
    create_directory(base_dir + "/train/annotations");
    create_directory(base_dir + "/val");
    create_directory(base_dir + "/val/images");
    create_directory(base_dir + "/val/masks");
    create_directory(base_dir + "/val/annotations");
    create_directory(base_dir + "/test");
    create_directory(base_dir + "/test/images");
    create_directory(base_dir + "/test/masks");
    create_directory(base_dir + "/test/annotations");
}

std::vector<Circle> generate_random_circles(int n, int img_width, int img_height, 
                                           int radius_min, int radius_max) {
    std::vector<Circle> circles;
    circles.reserve(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> radius_dist(radius_min, radius_max);
    std::uniform_int_distribution<int> intensity_dist(30, 150);  
    
    for (int i = 0; i < n; i++) {
        int radius = radius_dist(gen);
        int intensity = intensity_dist(gen);
        
        std::uniform_int_distribution<int> x_dist(radius, img_width - radius);
        std::uniform_int_distribution<int> y_dist(radius, img_height - radius);
        
        Circle c;
        c.id = i; 
        c.x = x_dist(gen);
        c.y = y_dist(gen);
        c.radius = radius;
        c.intensity = intensity;
        c.color = cv::Scalar(intensity, intensity, intensity);
        c.overlapping = false; 
        
        circles.push_back(c);
    }
    
    return circles;
}

// Generate instance mask: Each circle keeps its ID even when overlapping
cv::Mat generate_instance_mask(const std::vector<Circle>& circles,
                               int img_width, int img_height) {
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, CV_16UC1);
    
    // Draw circles in order - each keeps its unique ID
    for (const auto& circle : circles) {
        cv::circle(mask, 
                  cv::Point(circle.x, circle.y), 
                  circle.radius, 
                  cv::Scalar(circle.id + 1),
                  -1);
    }
    
    return mask;
}

void save_instance_info(const std::vector<Circle>& circles, 
                       const std::string& filepath) {
    std::ofstream file(filepath);
    file << "{\n";
    file << "  \"num_instances\": " << circles.size() << ",\n";
    file << "  \"circles\": [\n";
    
    for (size_t i = 0; i < circles.size(); i++) {
        const auto& c = circles[i];
        file << "    {\n";
        file << "      \"id\": " << c.id << ",\n";
        file << "      \"x\": " << c.x << ",\n";
        file << "      \"y\": " << c.y << ",\n";
        file << "      \"radius\": " << c.radius << ",\n";
        file << "      \"intensity\": " << c.intensity << "\n";
        file << "    }";
        if (i < circles.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file.close();
}

// Draw each circle separately with realistic gradient
cv::Mat draw_single_circle_with_gradient(int width, int height, const Circle& circle) {
    cv::Mat circle_img = cv::Mat::zeros(height, width, CV_32FC1);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> intensity_variation(0, 5);
    
    for (int y = std::max(0, circle.y - circle.radius); 
         y <= std::min(height - 1, circle.y + circle.radius); y++) {
        for (int x = std::max(0, circle.x - circle.radius); 
             x <= std::min(width - 1, circle.x + circle.radius); x++) {
            
            int dx = x - circle.x;
            int dy = y - circle.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist <= circle.radius) {
                // Radial gradient: darker at center, lighter at edges
                float normalized_dist = dist / circle.radius;
                float gradient_factor = 0.7f + 0.3f * normalized_dist;
                
                // Add texture variation
                float texture_noise = intensity_variation(gen);
                
                float pixel_intensity = circle.intensity * gradient_factor + texture_noise;
                pixel_intensity = std::max(0.0f, std::min(255.0f, pixel_intensity));
                
                circle_img.at<float>(y, x) = pixel_intensity;
            }
        }
    }
    
    return circle_img;
}

// NEW: Improved rendering that maintains boundaries
cv::Mat draw_realistic_circles(const std::vector<Circle>& circles, 
                               int img_width, int img_height,
                               bool add_texture) {
    
    // Start with noisy background (like CTEM)
    cv::Mat image = cv::Mat::zeros(img_height, img_width, CV_32FC1);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(180, 15);  // Background intensity
    
    // Add background noise
    for (int y = 0; y < img_height; y++) {
        for (int x = 0; x < img_width; x++) {
            image.at<float>(y, x) = noise(gen);
        }
    }
    
    // Draw each circle
    for (const auto& circle : circles) {
        cv::Mat circle_layer = draw_single_circle_with_gradient(img_width, img_height, circle);
        
        // Composite circles (min operation to keep darker values)
        for (int y = 0; y < img_height; y++) {
            for (int x = 0; x < img_width; x++) {
                if (circle_layer.at<float>(y, x) > 0) {
                    image.at<float>(y, x) = std::min(
                        image.at<float>(y, x), 
                        circle_layer.at<float>(y, x)
                    );
                }
            }
        }
    }
    
    // Convert to 8-bit and apply slight blur
    cv::Mat image_8u;
    image.convertTo(image_8u, CV_8UC1);
    cv::GaussianBlur(image_8u, image_8u, cv::Size(3, 3), 0.5);
    
    return image_8u;
}

std::string get_split_name(int idx, int total, float train_split, float val_split) {
    float ratio = static_cast<float>(idx) / total;
    if (ratio < train_split) return "train";
    if (ratio < train_split + val_split) return "val";
    return "test";
}

void generate_dataset(const DatasetConfig& config) {
    create_directory_structure(config.output_dir);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> n_circles_dist(config.n_circles_min, 
                                                       config.n_circles_max);
    
    for (int i = 0; i < config.num_samples; i++) {
        int n_circles = n_circles_dist(gen);
        
        auto circles = generate_random_circles(n_circles, 
                                               config.img_width, 
                                               config.img_height,
                                               config.radius_min, 
                                               config.radius_max);
        
        // Generate realistic image
        cv::Mat image = draw_realistic_circles(circles, config.img_width, config.img_height, true);
        
        // Generate INSTANCE mask (each circle keeps unique ID)
        cv::Mat instance_mask = generate_instance_mask(circles, config.img_width, config.img_height);
        
        // Determine split
        std::string split = get_split_name(i, config.num_samples, 
                                           config.train_split, config.val_split);
        
        // Create filename
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        std::string filename = ss.str();
        
        // Save image
        std::string img_path = config.output_dir + "/" + split + "/images/image_" + filename + ".png";
        cv::imwrite(img_path, image);
        
        // Save instance mask (16-bit PNG)
        std::string mask_path = config.output_dir + "/" + split + "/masks/mask_" + filename + ".png";
        cv::imwrite(mask_path, instance_mask);
        
        // Save annotations as JSON
        std::string json_path = config.output_dir + "/" + split + "/annotations/anno_" + filename + ".json";
        save_instance_info(circles, json_path);
        
        if ((i + 1) % 100 == 0) {
            std::cout << "Generated " << i + 1 << "/" << config.num_samples << " samples\n";
        }
    }
    
    std::cout << "Dataset generation complete!\n";
}

int main() {
    DatasetConfig config;
    config.num_samples = 5000;
    config.img_width = 512;   
    config.img_height = 512;
    config.n_circles_min = 10;
    config.n_circles_max = 30;
    config.radius_min = 15;
    config.radius_max = 40;
    config.del_count_min = 0;  
    config.del_count_max = 0;
    config.output_dir = "ctem_instance_dataset";
    config.train_split = 0.7;
    config.val_split = 0.15;
    config.add_realistic_texture = true;
    config.add_background_noise = true;
    
    generate_dataset(config);
    
    return 0;
}
