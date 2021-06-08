cmake_minimum_required(VERSION 3.6)
set(CMAKE_VERBOSE_MAKEFILE on)
# dummy target to bring libc++_shared.so into packages
#add_library(opencv_jni_shared STATIC dummy.cpp)
#要将 ADD_EXECUTABLE的声明放到设置库文件的TARGET_LINK_LIBRARIES之前
add_library(libopencv_java4 SHARED IMPORTED)
set_target_properties(libopencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${PROJECT_SOURCE_DIR}/../native/libs/arm64-v8a/libopencv_java4.so)

include_directories(${PROJECT_SOURCE_DIR}/../native/jni/include)