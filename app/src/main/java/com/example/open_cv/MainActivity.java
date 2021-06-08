package com.example.open_cv;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.jni.JniBridge;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfDouble;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends CameraActivity implements
        CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener{
    private final String TAG = getClass().getName();
    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera;
    // 手动装载openCV库文件，以保证手机无需安装OpenCV Manager   不加这里将导致无法初始化 级联分类器（开机闪退）

    private Net net;
    private ImageView imageView;

    private ExecutorService singleThreadExecutor =  Executors.newFixedThreadPool(10);;

    String sdPath = "/sdcard/opencv";
//        String image_file = sdPath + File.separator + "dog.jpg";
//        System.out.println("Reading File"+image_file);

    //        Mat im = Imgcodecs.imread(image_file, Imgcodecs.IMREAD_COLOR);
    // get a new Frame
    // Mat im = inputFrame.rgba();
    String[] names = new String[]{
            "aeroplane","bicycle","bird","boat","bottle",
            "bus","car","cat","chair","cow",
            "diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor"
    };
    private List<String> classNamesVec = new ArrayList<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, 1);
        }
        verifyStoragePermissions(this);
        initWindowSettings();
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this); // 设置相机监听
        initClassifier();
        cameraView.enableView();
        Button switchCamera = (Button) findViewById(R.id.switch_camera);
        switchCamera.setOnClickListener(this); // 切换相机镜头，默认后置

//        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test1);
//        Mat source = new Mat();
//        Utils.bitmapToMat(bitmap, source);
//
//        String path = getCacheDir().getAbsolutePath() + File.separator + "lena.jpg" ;
//        Log.w(TAG,path);

//        try {
//            File file = new File(path);
//            FileOutputStream out = new FileOutputStream(file);
//            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
//            out.flush();
//            out.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//
//        Mat bgr = Imgcodecs.imread(path, Imgcodecs.IMREAD_UNCHANGED );
//        Bitmap bitmap2 = Bitmap.createBitmap(bgr.width(), bgr.height(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(bgr, bitmap2);

//        Mat mBgr;
//        try {
//            mBgr = Utils.loadResource(this, R.drawable.test1);
//            showGray(mBgr);
//
//            Mat mRgb = new Mat();
//            Imgproc.cvtColor(mBgr, mRgb, Imgproc.COLOR_BGR2RGB);
//            buildGaussian(mBgr);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        String cfg_path= sdPath + File.separator + "yolov4-tiny.cfg";
        String model_path= sdPath + File.separator + "yolov4-tiny.weights";
        net = Dnn.readNetFromDarknet(cfg_path, model_path);
        if ( net.empty() ) {
            System.out.println("Reading Net error");
        }
        System.out.println("Reading Net success");

//        if( im.empty() ) {
//            System.out.println("Reading Image error");
//        }

        try {   //读取names
            File file = new File(sdPath + File.separator +"coco.names");
            InputStream instream = new FileInputStream(file);
            InputStreamReader inputreader = new InputStreamReader(instream);
            BufferedReader buffreader = new BufferedReader(inputreader);
            String line;
            while ((line = buffreader.readLine()) != null) {
                classNamesVec.add(line);
            }
            instream.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void buildGaussian(Mat mRgb) {
        Mat noise = new Mat(mRgb.size(), mRgb.type());
        Mat result = new Mat();
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble dev = new MatOfDouble();

        Core.meanStdDev(mRgb, mean, dev);

        Core.randn(noise, mean.get(0,0)[0], dev.get(0,0)[0]);
        showMat( noise);
        Core.add(mRgb, noise, result);
        showMat( result);

        noise.release();
        result.release();
    }

    private void showGray(Mat mBgr) {
        Mat gray = new Mat();
        Imgproc.cvtColor(mBgr, gray, Imgproc.COLOR_BGR2GRAY);
        showMat(gray);
        gray.release();
    }

    private void showMat(Mat source) {
        Bitmap bitmap = Bitmap.createBitmap(source.width(), source.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(source, bitmap);
        imageView.setImageBitmap(bitmap);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                String displayString =
                        "OpenCV loaded successfully via initAsync" ;
                Log.w(TAG,displayString);
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.switch_camera:
                cameraView.disableView();
                if (isFrontCamera) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isFrontCamera = false;
                } else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isFrontCamera = true;
                }
                cameraView.enableView();
                break;
            default:
        }
    }

    // 初始化窗口设置, 包括全屏、横屏、常亮
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }

    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = JniBridge.init(this, mLoaderCallback,  cascadeFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }
    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        List<CameraBridgeViewBase> list = new ArrayList<>();
        list.add(cameraView);
        return list;
    }

    int i  =0;
    @Override
    // 这里执行人脸检测的逻辑, 根据OpenCV提供的例子实现(face-detection)
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // 翻转矩阵以适配前后置摄像头
        if (isFrontCamera) {
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        } else {
            //如果发现后摄出现了镜像  把下面的注释打开即可  魅族不需要
//            Core.flip(mRgba, mRgba, -1);
//            Core.flip(mGray, mGray, -1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();
        if (classifier != null) {
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        for (Rect faceRect : facesArray) {
            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
        }
        Mat im = new Mat();
        Imgproc.cvtColor(mRgba, im, Imgproc.COLOR_RGBA2RGB);

        Mat frame = new Mat();
        Size sz1 = new Size(im.cols(),im.rows());
        Imgproc.resize(im, frame, sz1);
        Mat resized = new Mat();
        Size sz = new Size(320,320);
        Imgproc.resize(im, resized, sz);
        float scale = 1.0F / 255.0F;
        Mat inputBlob = Dnn.blobFromImage(im, scale, sz, new Scalar(0), false, false);

        net.setInput(inputBlob, "data");//

        singleThreadExecutor.execute((Runnable) () -> {
            Mat detectionMat = net.forward();
            if( detectionMat.empty() ) {
                System.out.println("No result");
            }
            System.out.println("No result"+detectionMat.rows());

            for (int i = 0; i < detectionMat.rows(); i++)
            {
                int probability_index = 5;
                int size = (int) (detectionMat.cols() * detectionMat.channels());
                float[] data = new float[size];
                detectionMat.get(i, 0, data);
                float confidence = -1;
                int objectClass = -1;
                for (int j=0; j < detectionMat.cols();j++)
                {
                    if (j>=probability_index && confidence<data[j])
                    {
                        confidence = data[j];
                        objectClass = j-probability_index;
                    }
                }

                if (confidence > 0.1)
                {
                    System.out.println("Result Object: "+i);
                    for (int j=0; j < detectionMat.cols();j++) {
                        System.out.print(" "+j+":"+ data[j]);
                    }
                    System.out.println("");
                    float x = data[0];
                    float y = data[1];
                    float width = data[2];
                    float height = data[3];
                    float xLeftBottom = (x - width / 2) * frame.cols();
                    float yLeftBottom = (y - height / 2) * frame.rows();
                    float xRightTop = (x + width / 2) * frame.cols();
                    float yRightTop = (y + height / 2) * frame.rows();
                    System.out.println("Class: "+ classNamesVec.get(objectClass));
                    System.out.println("Confidence: "+confidence);
                    System.out.println("ROI: "+xLeftBottom+" "+yLeftBottom+" "+xRightTop+" "+yRightTop+"\n");

                    Imgproc.rectangle(frame, new Point(xLeftBottom, yLeftBottom),
                            new Point(xRightTop,yRightTop),new Scalar(0, 255, 0),3);
                    // Write class name and confidence.
                    String label = classNamesVec.get(objectClass) + ": " + confidence;
                    Imgproc.putText(frame, label, new Point(xLeftBottom, xLeftBottom),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
                    System.out.println("i  i  i: "+i);

//                    if (i < 30){
//                        i++;
//                        File file =
//                                new File(sdPath+ File.separator +"/result"+ File.separator + i +"qwert.jpg");
//                        System.out.println("file where: "+file.getPath());
//
//                        if (!file.exists()) {
//                            try {
//                                System.out.println("No file");
//                                file.createNewFile();
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        Imgcodecs.imwrite(file.getPath(), frame);
//                        System.out.println("imwrite success");
//
//                    }
                }
            }
        });

        return frame;
    }
    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0 &&
                        grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                } else {
                    Toast.makeText(this, "权限拒绝", Toast.LENGTH_SHORT).show();
                }
        }
    }

    private final int REQUEST_EXTERNAL_STORAGE = 1;

    private String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };

    private void verifyStoragePermissions(Activity activity) {
        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}