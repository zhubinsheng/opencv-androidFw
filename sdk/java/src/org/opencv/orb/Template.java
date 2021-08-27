package org.opencv.orb;

public class Template {
//    struct Template
//    {
//        Keypoints keypoints;
//        cv::Mat descriptors;
//        Vertices pane;
//        std::vector<Object> objects;
//    };
    private long MatNativeObj;
    private long outImageMatNativeObj;

    private long keypointsNativeObj;
    private long descriptorsNativeObj;
    private long paneNativeObj;
    private long objectsNativeObj;

    public Template() {
    }

//    public Template templateConstructor() {
//        Template t = new Template();
//        t.setDescriptorsNativeObj();
//        t.setKeypointsNativeObj();
//        t.setObjectsNativeObj();
//        t.setPaneNativeObj();
//        return t;
//    }

    public long getKeypointsNativeObj() {
        return keypointsNativeObj;
    }

    public void setKeypointsNativeObj(long keypointsNativeObj) {
        this.keypointsNativeObj = keypointsNativeObj;
    }

    public long getDescriptorsNativeObj() {
        return descriptorsNativeObj;
    }

    public void setDescriptorsNativeObj(long descriptorsNativeObj) {
        this.descriptorsNativeObj = descriptorsNativeObj;
    }

    public long getPaneNativeObj() {
        return paneNativeObj;
    }

    public void setPaneNativeObj(long paneNativeObj) {
        this.paneNativeObj = paneNativeObj;
    }

    public long getObjectsNativeObj() {
        return objectsNativeObj;
    }

    public void setObjectsNativeObj(long objectsNativeObj) {
        this.objectsNativeObj = objectsNativeObj;
    }

    public long getMatNativeObj() {
        return MatNativeObj;
    }

    public void setMatNativeObj(long matNativeObj) {
        MatNativeObj = matNativeObj;
    }

    public long getOutImageMatNativeObj() {
        return outImageMatNativeObj;
    }

    public void setOutImageMatNativeObj(long outImageMatNativeObj) {
        this.outImageMatNativeObj = outImageMatNativeObj;
    }
}