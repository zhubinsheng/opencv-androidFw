package org.opencv.orb;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect2d;

public class OrbBridge {
    private static final String TAG = OrbBridge.class.getSimpleName();
    private OrbResultCallback orbResultCallback;
    private Mat mat = new Mat();
    private Template template = new Template();
    private MatOfKeyPoint keyPointA = new MatOfKeyPoint();
    static {
        System.loadLibrary("locai"); //libliblocai.so
    }
    public void findObjects(Mat mRgba,OrbResultCallback orbResultCb) {
        this.orbResultCallback = orbResultCb;
        template.setMatNativeObj(mRgba.nativeObj);
        findObjects(template);

    }

    public void buildTemplate(Mat mRgba, Rect2d rect2d, OrbResultCallback orbResultCb) {
//        Log.e(TAG, "Thread:"+ Thread.currentThread().getId());
        this.orbResultCallback = orbResultCb;

        template.setMatNativeObj(mRgba.nativeObj);
//        findObjects(template);

        template.setKeypointsNativeObj(keyPointA.nativeObj);
//        template.setDescriptorsNativeObj(12345678);
        template.setOutImageMatNativeObj(mat.getNativeObjAddr());

        testTemplate(template, rect2d.x , rect2d.y, rect2d.width, rect2d.height);
//        Log.e(TAG, "Thread:"+ Thread.currentThread().getId());

//        findObjects(1,2);
//        buildTemplate(231,232);
//        Log.e(TAG, "buildTemplate:"+ buildTemplate(3,34));
//        Log.e(TAG, "buildTemplate:"+ stringFromJNI("1,2)"));
    }

    public void showrrr(long matNativeObj){
//        Log.e(TAG, "Thread:"+ Thread.currentThread().getId());

//        mat.release();
//        mat.release();
        orbResultCallback.rrr(mat);
    }

    public void showOrb(double x, double y, double w, double h){
        Log.d(TAG, "Thread showOrb:"+ Thread.currentThread().getId());
        orbResultCallback.showOrb(x,y,w,h);
    }

    private native void testTemplate(Template template, double x, double y, double width, double height);

    private native String stringFromJNI(String str);

//    private static native void detectAndCompute_0(long nativeObj, long image_nativeObj, long mask_nativeObj, long keypoints_mat_nativeObj, long descriptors_nativeObj, boolean useProvidedKeypoints);
//

    private native void buildTemplate(long i, long j);

    private native void findObjects(Template template);

}
