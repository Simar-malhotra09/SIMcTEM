
#include <opencv2/opencv.hpp>

struct Circle {
    int id; // just the iter i out of n for now. 
    int x, y;      
    int radius;
    int intensity;
    cv::Scalar color; // what we want with color is some form of addition of intensities; shared space of overlapping objects may be darker than idv themselves
    bool overlapping; // true if it shares space with another object? 
};

struct CollectionCircle{
    std::unordered_map<int, Circle> circles; // key is id of circle 
}; 

// add random randomd discoloration to object by 
// generating a 'blob' object with diff intensity
// relative to parent obj
std::vector<Circle> add_random_discolorations(
    std::vector<Circle> circles
);

// generate random blob for an object 
// https://stackoverflow.com/questions/54828017/c-create-random-shaped-blob-objects
 

std::vector<Circle> generate_random_circles(int n, int img_width, int img_height, 
                                           int radius_min, int radius_max);

// draw plain old circles 
cv::Mat draw_circles_to_image(const std::vector<Circle> circles, 
                              int img_width, int img_height,
                              cv::Scalar background = cv::Scalar(255, 255, 255));

// circles with deformations 
cv::Mat draw_circles_to_image_with_deletions(
    const std::vector<Circle>& circles,// circles with deformations >& circles, 
    int img_width, int img_height,
    int del_count_per_object_min,
    int del_count_per_object_max,
    cv::Scalar background = cv::Scalar(255, 255, 255));
 
