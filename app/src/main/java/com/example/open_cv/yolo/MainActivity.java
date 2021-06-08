package com.example.open_cv.yolo;
import android.os.Bundle;
import android.os.Environment;

import androidx.appcompat.app.AppCompatActivity;

import com.example.open_cv.R;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("opencv_java4");
    }
    private String sdPath = Environment.getExternalStorageDirectory().getAbsolutePath();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sdPath = "/sdcard/opencv";
        String image_file = sdPath + File.separator + "dog.jpg";
        System.out.println("Reading File"+image_file);

        Mat im = Imgcodecs.imread(image_file, Imgcodecs.IMREAD_COLOR);
        // get a new Frame
        // Mat im = inputFrame.rgba();
        String[] names = new String[]{
                "aeroplane","bicycle","bird","boat","bottle",
                "bus","car","cat","chair","cow",
                "diningtable","dog","horse","motorbike","person",
                "pottedplant","sheep","sofa","train","tvmonitor"
        };
        String cfg_path= sdPath + File.separator + "yolov4.cfg";
        String model_path= sdPath + File.separator + "yolov4.weights";
        Net net = Dnn.readNetFromDarknet(cfg_path, model_path);
        if ( net.empty() ) {
            System.out.println("Reading Net error");
        }

        if( im.empty() ) {
            System.out.println("Reading Image error");
        }

        Mat frame = new Mat();
        Size sz1 = new Size(im.cols(),im.rows());
        Imgproc.resize(im, frame, sz1);
        Mat resized = new Mat();
        Size sz = new Size(320,320);
        Imgproc.resize(im, resized, sz);
        float scale = 1.0F / 255.0F;
        Mat inputBlob = Dnn.blobFromImage(im, scale, sz, new Scalar(0), false, true);
        net.setInput(inputBlob, "data");//
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

            if (confidence > 0.7)
            {
//                System.out.println("Result Object: "+i);
                for (int j=0; j < detectionMat.cols();j++) {
//                    System.out.print(" "+j+":"+ data[j]);
                }
//                System.out.println("");
                float x = data[0];
                float y = data[1];
                float width = data[2];
                float height = data[3];
                float xLeftBottom = (x - width / 2) * frame.cols();
                float yLeftBottom = (y - height / 2) * frame.rows();
                float xRightTop = (x + width / 2) * frame.cols();
                float yRightTop = (y + height / 2) * frame.rows();
                System.out.println("Class: "+ names[objectClass]);
                System.out.println("Confidence: "+confidence);
                System.out.println("ROI: "+xLeftBottom+" "+yLeftBottom+" "+xRightTop+" "+yRightTop+"\n");

                Imgproc.rectangle(frame, new Point(xLeftBottom, yLeftBottom),
                        new Point(xRightTop,yRightTop),new Scalar(0, 255, 0),3);
            }
        }

        Imgcodecs.imwrite(sdPath + File.separator +"out_test.jpg", frame );

        ///

    }
}