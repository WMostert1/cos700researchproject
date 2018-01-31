import weka.core.Instances;

public class FSBundle {
    private Instances data;
    private double accuracy;
    private int [] attributes;

    public FSBundle(Instances data, double accuracy, int[] attributes) {
        this.data = data;
        this.accuracy = accuracy;
        this.attributes = attributes;
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public int[] getAttributes() {
        return attributes;
    }

    public void setAttributes(int[] attributes) {
        this.attributes = attributes;
    }
}
