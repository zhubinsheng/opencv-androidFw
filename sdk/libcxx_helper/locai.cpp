#include <jni.h>
#include <android/log.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "locai.h"

#define JNIREG_CLASS "org/opencv/orb/OrbBridge"
#define JNIObj_CLASS "org/opencv/orb/Template"


namespace dnn { namespace locai {
    float Detector::good_ratio = 0.8f;
    int Detector::fast_threshold = 15;
    int Detector::patch_size = 19;
    bool Detector::verbose = false;
    float Detector::akaze_threshold = 3e-4f;
    float Detector::ransac_threshold = 3.f;
    int Detector::min_matches = 10;
    float Detector::reject_ratio = 0.7f;

    static VerticesF
    _cvt(const Vertices& vpts)
    {
        VerticesF ret;
        for(auto pt : vpts) {
            ret.push_back(cv::Point2f(pt.x, pt.y));
        }
        return ret;
    }

    static Vertices
    _cvt(const VerticesF& vpts)
    {
        Vertices ret;
        for(auto pt : vpts) {
            ret.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }
        return ret;
    }

    Detector::Detector(bool use_akaze)
    {
        if(use_akaze) {
            auto akaze = cv::AKAZE::create();
            akaze->setThreshold(akaze_threshold);
            _detector = akaze;
        }
        else
            _detector = cv::ORB::create(500, 1.2f, 8, patch_size, 0, 2, cv::ORB::HARRIS_SCORE, patch_size, fast_threshold);
        
        _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }

    void
    Detector::buildTemplate(const cv::Mat& img, Template& tpl)
    {
        if (!tpl.pane.empty()) {
            cv::Mat mask;
            cv::fillPoly(mask, tpl.pane, cv::Scalar::all(255));
            _detector->detectAndCompute(img, mask, tpl.keypoints, tpl.descriptors);
        } else
            _detector->detectAndCompute(img, cv::noArray(), tpl.keypoints, tpl.descriptors);
    }

    void
    Detector::findObjects(const cv::Mat& img, const Template& tp, std::vector<Object>& output)
    {
        output.clear();

        Keypoints target_keypoints;
        cv::Mat target_descriptors;
        _detector->detectAndCompute(img, cv::noArray(), target_keypoints, target_descriptors);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        _matcher->knnMatch(tp.descriptors, target_descriptors, knn_matches, 2);
//        __android_log_print(ANDROID_LOG_INFO, "native", "knn_matches %lu", knn_matches.size());

        std::vector<cv::DMatch> good_matches;
        for(size_t i = 0; i < knn_matches.size(); i++) {
            if(knn_matches[i][0].distance < good_ratio * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        _matches = good_matches.size();
//        __android_log_print(ANDROID_LOG_INFO, "native", "good_matches %lu", _matches);

        if(good_matches.size() < min_matches)
            return;

        /** Localize the plane **/
        VerticesF obj;
        VerticesF scene;
        for(size_t i = 0; i < good_matches.size(); i++) {
            //-- Get the keypoints from the good matches
            obj.push_back(tp.keypoints[good_matches[i].queryIdx].pt);
            scene.push_back(target_keypoints[good_matches[i].trainIdx].pt);
        }

        cv::Mat inliers;
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC,ransac_threshold,inliers);
        _inlier_ratio = cv::sum(inliers)[0] / good_matches.size();
//        __android_log_print(ANDROID_LOG_INFO, "native", "_inlier_ratio %f", _inlier_ratio);

        if(_inlier_ratio < reject_ratio)
            return;

        Object oo;
        for (auto& obj : tp.objects) {
            oo.lbl = obj.lbl;
            
            VerticesF target_polygon;
            cv::perspectiveTransform(_cvt(obj.polygon), target_polygon, H);
            oo.polygon = _cvt(target_polygon);

            output.push_back(oo);
        }

    }
}}

static dnn::locai::Detector dev_detector;
static dnn::locai::Template tpl;


void logTime(const char string[]) {
    struct timeval xTime{};
    gettimeofday(&xTime, nullptr);
    long long xFactor = 1;
    auto now = (long long)(( xFactor * xTime.tv_sec * 1000) + (xTime.tv_usec / 1000));
    __android_log_print(ANDROID_LOG_INFO, string, "now_ld = %lld", now);
}

jlong getNativeObj(JNIEnv *env, jclass jcInfo, jobject jobj, const char string[], const char string1[]) {
    jfieldID jfi = env->GetFieldID(jcInfo, string, string1);
    jlong jl = env->GetLongField(jobj, jfi);
    return jl;
}

jstring stringFromJNI(JNIEnv *env, jobject thiz, jstring prompt){
    const char* str;
    str = env->GetStringUTFChars(prompt, (jboolean*)false);
    if(str == NULL) {
        return NULL; /* OutOfMemoryError already thrown */
    }
    std::string hello = "Hello fr`om C++";
    hello.append(str);
    env->ReleaseStringUTFChars(prompt, str);
    return env->NewStringUTF(hello.c_str());
}

void buildTemplate(JNIEnv *env, jobject thiz, jlong i, jlong j){
    __android_log_print(ANDROID_LOG_INFO, "native", "buildTemplate %ld", i+j );
//    dev_detector.buildTemplate();
}
void testTemplate(JNIEnv *env, jobject thiz, jobject jobj, jdouble x, jdouble y, jdouble width, jdouble height);

void findObjects(JNIEnv *env, jobject thiz, jobject jobj){

    jclass jcInfo = env->FindClass(JNIObj_CLASS);
    jlong jl = getNativeObj(env, jcInfo,  jobj, "MatNativeObj", "J");

    cv::Mat& mRgb = *(cv::Mat*)jl;
    std::vector<dnn::locai::Object> output;
//    logTime("findObjects start");
    dev_detector.findObjects(mRgb, tpl, output);
//    logTime("findObjects end");
//    __android_log_print(ANDROID_LOG_INFO, "native", "output result : %ld", output.size() );

    if(!output.empty()) {
//        __android_log_print(ANDROID_LOG_INFO, "native", "output not empty" );
    }

    jclass clazz = env->FindClass(JNIREG_CLASS);
    jmethodID myid = env->GetMethodID(clazz, "showOrb", "(DDDD)V");
    if (myid == nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "native", "nullptr" );
    }

    for (auto arr: output){
        __android_log_print(ANDROID_LOG_INFO, "native", "oo.lbl %d", arr.lbl);

        jdouble x = arr.polygon[0].x;
        jdouble y = arr.polygon[0].y;
        jdouble width = arr.polygon[1].x - arr.polygon[0].x;
        jdouble height= arr.polygon[3].y - arr.polygon[0].y;

        __android_log_print(ANDROID_LOG_INFO, "native", "jdouble x %f", x );
        __android_log_print(ANDROID_LOG_INFO, "native", "jdouble y %f", y );
        __android_log_print(ANDROID_LOG_INFO, "native", "jdouble width %f", width );
        __android_log_print(ANDROID_LOG_INFO, "native", "jdouble height %f", height );

        env->CallVoidMethod(thiz, myid, x, y, width, height);

    }

//    cv::polylines(img_object, tpl.objects[0].polygon, true, cv::Scalar(0, 255, 0));
}


std::vector<cv::Point>
cvtPolyCurve(std::vector<cv::Point2f>& vpts)
{
    std::vector<cv::Point> ret;
    for(auto pt : vpts) {
        ret.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    }
    return ret;
}

cv::Point2f
getPolyPoint(int x, int y, float scale)
{
    cv::Point2f ret;
    ret.x = x * scale;
    ret.y = y * scale;
    return ret;
}

void testTemplate(JNIEnv *env, jobject thiz, jobject jobj, jdouble x, jdouble y, jdouble width, jdouble height){

    jclass jcInfo = env->FindClass(JNIObj_CLASS);

    jfieldID jfi = env->GetFieldID(jcInfo, "descriptorsNativeObj", "J");
    jlong jl = env->GetLongField(jobj, jfi);
//    __android_log_print(ANDROID_LOG_INFO, "native", "testTemplate %ld", jl );

    jl = getNativeObj(env, jcInfo,  jobj, "keypointsNativeObj", "J");

//    __android_log_print(ANDROID_LOG_INFO, "native", "testTemplate %ld", jl );


    jl = getNativeObj(env, jcInfo,  jobj, "MatNativeObj", "J");
//    __android_log_print(ANDROID_LOG_INFO, "native", "testTemplate %ld", jl );
    tpl.objects.clear();

    dnn::locai::VerticesF vpts;
    vpts.clear();

    __android_log_print(ANDROID_LOG_INFO, "native", "const jdouble x %f", x );
    __android_log_print(ANDROID_LOG_INFO, "native", "const jdouble y %f", y );
    __android_log_print(ANDROID_LOG_INFO, "native", "const jdouble width %f", width );
    __android_log_print(ANDROID_LOG_INFO, "native", "const jdouble height %f", height );

//    vpts.push_back(getPolyPoint(200, 300,  1));
//    vpts.push_back(getPolyPoint(400, 300,  1));
//    vpts.push_back(getPolyPoint(400, 500, 1));
//    vpts.push_back(getPolyPoint(200, 500,  1));
    vpts.push_back(getPolyPoint(x, y,  1));
    vpts.push_back(getPolyPoint(x+width, y,  1));
    vpts.push_back(getPolyPoint(x+width, y+height, 1));
    vpts.push_back(getPolyPoint(x, y+height,  1));

    cv::Mat& mRgb = *(cv::Mat*)jl;
//    logTime("buildTemplate start");
    dev_detector.buildTemplate(mRgb,tpl);


    dnn::locai::Object oo;
    oo.lbl = 1;
    oo.polygon = cvtPolyCurve(vpts);

    tpl.objects.push_back(oo);


//    logTime("buildTemplate end");

//    drawKeypoints(rgb, points, result, Scalar.all(255.0), DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

//    outImage = cv::Mat();

    jl = getNativeObj(env, jcInfo,  jobj, "outImageMatNativeObj", "J");
//    __android_log_print(ANDROID_LOG_INFO, "native", "testTemplate outImageMatNativeObj %ld", jl );

    cv::Mat& outImage = *(cv::Mat*)jl;

    cv::drawKeypoints(mRgb, tpl.keypoints, outImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    jclass clazz = env->FindClass(JNIREG_CLASS);

    jmethodID myid = env->GetMethodID(clazz, "showrrr", "(J)V");
    if (myid == nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "native", "nullptr" );
    }
    env->CallVoidMethod(thiz, myid, (jlong) &outImage);

//    tpl.descriptors
//    dnn::locai::Object oo;
//    oo.lbl = 1;
//    oo.polygon = cvtPolyCurve(vpts_object);
//
//    tpl.objects.push_back(oo);
}


static const JNINativeMethod gMethods[] = {
        {"stringFromJNI", "(Ljava/lang/String;)Ljava/lang/String;", (jstring*)stringFromJNI},
        {"findObjects", "(Lorg/opencv/orb/Template;)V", (void *)findObjects},
        {"buildTemplate", "(JJ)V", (void *)buildTemplate},
        {"testTemplate", "(Lorg/opencv/orb/Template;DDDD)V", (void *)testTemplate}
};

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
    __android_log_print(ANDROID_LOG_INFO, "native", "Jni_OnLoad");
    JNIEnv* env = NULL;
    if(vm->GetEnv((void**)&env, JNI_VERSION_1_4) != JNI_OK) //从JavaVM获取JNIEnv，一般使用1.4的版本
        return -1;
    jclass clazz = env->FindClass(JNIREG_CLASS);
    if (!clazz){
    __android_log_print(ANDROID_LOG_INFO, "native", "cannot get class: com/example/efan/jni_learn2/MainActivity");
        return -1;
    }
    if(env->RegisterNatives(clazz, gMethods, sizeof(gMethods)/sizeof(gMethods[0]))){
    __android_log_print(ANDROID_LOG_INFO, "native", "register native method failed!\n");
        return -1;
    }
    dev_detector  = new dnn::locai::Detector(true);
//    dev_detector.findObjects();
    return JNI_VERSION_1_4;
}

