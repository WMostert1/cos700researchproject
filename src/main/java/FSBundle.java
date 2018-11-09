import lons.examples.BinarySolution;
import weka.core.Instances;

public class FSBundle {
    private Instances data;
    private double accuracy;
    private BinarySolution binarySolution;


    public FSBundle(Instances data, double accuracy, BinarySolution binarySolution) {
        this.data = data;
        this.accuracy = accuracy;
        this.binarySolution = binarySolution;
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

    public BinarySolution getBinarySolution() {
        return binarySolution;
    }

    public void setBinarySolution(BinarySolution binarySolution) {
        this.binarySolution = binarySolution;
    }
}
