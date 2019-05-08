package fs;

import fs.FeatureSelectionResult;

public class GAResult {
    private FeatureSelectionResult bestBundle;
    private double meanAccuracy;
    private double successRatio;

    public GAResult(FeatureSelectionResult bestBundle, double meanAccuracy, double successRatio) {
        this.bestBundle = bestBundle;
        this.meanAccuracy = meanAccuracy;
        this.successRatio = successRatio;
    }

    public FeatureSelectionResult getBestBundle() {
        return bestBundle;
    }

    public void setBestBundle(FeatureSelectionResult bestBundle) {
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
