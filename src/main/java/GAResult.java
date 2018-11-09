public class GAResult {
    private FSBundle bestBundle;
    private double meanAccuracy;
    private double successRatio;

    public GAResult(FSBundle bestBundle, double meanAccuracy, double successRatio) {
        this.bestBundle = bestBundle;
        this.meanAccuracy = meanAccuracy;
        this.successRatio = successRatio;
    }

    public FSBundle getBestBundle() {
        return bestBundle;
    }

    public void setBestBundle(FSBundle bestBundle) {
        this.bestBundle = bestBundle;
    }

    public double getMeanAccuracy() {
        return meanAccuracy;
    }

    public void setMeanAccuracy(double meanAccuracy) {
        this.meanAccuracy = meanAccuracy;
    }

    public double getSuccessRatio() {
        return successRatio;
    }

    public void setSuccessRatio(double successRatio) {
        this.successRatio = successRatio;
    }
}
