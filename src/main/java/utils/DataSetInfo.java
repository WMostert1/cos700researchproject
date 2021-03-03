package utils;

public class DataSetInfo {

    private String name;
    int noAttrs, noInstances, noClasses;

    public DataSetInfo(String name, int noAttrs, int noInstances, int noClasses) {
        this.name = name;
        this.noAttrs = noAttrs;
        this.noInstances = noInstances;
        this.noClasses = noClasses;
    }

    public String getName() {
        return name;
    }

    public int getNoAttrs() {
        return noAttrs;
    }

    public int getNoInstances() {
        return noInstances;
    }

    public int getNoClasses() {
        return noClasses;
    }
}
