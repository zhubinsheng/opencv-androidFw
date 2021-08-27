#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace dnn { namespace locai {
    typedef std::vector<cv::Point> Vertices;
    typedef std::vector<cv::Point2f> VerticesF;

    typedef std::vector<cv::KeyPoint> Keypoints;

    struct Object
    {
        int lbl;
        Vertices polygon;
    };

    struct Template
    {
        Keypoints keypoints;
        cv::Mat descriptors;
        Vertices pane;
        std::vector<Object> objects;
    };

    class Detector
    {
    public:
        static float good_ratio;
        static int fast_threshold;
        static int patch_size;
        static bool verbose;

        static float akaze_threshold;
        static float ransac_threshold;
        static int min_matches;
        static float reject_ratio;
        
        Detector(bool use_akaze = true);

        void buildTemplate(const cv::Mat& img, Template & tpl);
        void findObjects(const cv::Mat& img, const Template& tpl, std::vector<Object>& output);

    private:
        cv::Ptr<cv::Feature2D> _detector;
        cv::Ptr<cv::DescriptorMatcher> _matcher;

    public:
        /* debug */
        float _inlier_ratio;
        int _matches;
    };
    
}}
