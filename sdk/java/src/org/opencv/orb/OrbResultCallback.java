package org.opencv.orb;

import org.opencv.core.Mat;

public interface OrbResultCallback {
    void rrr(Mat mat);

    void showOrb(double x, double y, double w, double h);
}
