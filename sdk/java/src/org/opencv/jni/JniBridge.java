package org.opencv.jni;

import android.app.Activity;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;

import static android.content.ContentValues.TAG;

public class JniBridge {
    static {
        System.loadLibrary("opencv_java4");
    }

    public static CascadeClassifier init(Activity activity, BaseLoaderCallback mLoaderCallback, File cascadeFile){
//        OpenCVLoader.initAsync(
//                OpenCVLoader.OPENCV_VERSION,
//                activity,
//                mLoaderCallback);
        if (!OpenCVLoader.initDebug()){
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        return new CascadeClassifier(cascadeFile.getAbsolutePath());


    }
}
