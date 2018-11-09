import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lons.EdgeType;
import lons.LONGenerator;
import lons.RVisualizationFormatter;
import lons.Weight;
import lons.examples.*;
import mutators.UniformSampleMutator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;


/**
 * Created by bbdnet1339 on 2016/08/05.
 *
 */
public class MainApplication {
    public static final int DECIMAL_PLACES = 8;
    private static void sortIntArr(int [] arr){
        for(int i = 0; i < arr.length; i++){
            for(int j = 0; j < arr.length; j++){
                if(i == j)
                    continue;

                if(arr[j] > arr[i]){
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    private List<String> getResourceFiles( String path ) throws IOException {
        List<String> filenames = new ArrayList<>();

        try(
                InputStream in = getResourceAsStream( path );
                BufferedReader br = new BufferedReader( new InputStreamReader( in ) ) ) {
            String resource;

            while( (resource = br.readLine()) != null ) {
                filenames.add( resource );
            }
        }

        return filenames;
    }

    private InputStream getResourceAsStream( String resource ) {
        final InputStream in
                = getContextClassLoader().getResourceAsStream( resource );

        return in == null ? getClass().getResourceAsStream( resource ) : in;
    }

    private ClassLoader getContextClassLoader() {
        return Thread.currentThread().getContextClassLoader();
    }



    private static Instances getDataSet(String name) throws Exception {
        DataSource source = new DataSource(MainApplication.class.getResourceAsStream("data-sets/used/" + name+".arff"));
        boolean removeRedundantManual = false;

        if(removeRedundantManual){
            System.out.println("!!!!!MANUALLY REMOVING REDUNDANT VARIABLES FOR DATA SETS!!!!!");
        }

        Instances originalData = source.getDataSet();

        //REMOVE REDUNDANT FEATURES FROM VOWEL
        if(removeRedundantManual && name.equals(VOWEL)){
            originalData.deleteAttributeAt(0);
            originalData.deleteAttributeAt(0);
        }

        if(removeRedundantManual && name.equals(BREAST_W)){
            originalData.deleteAttributeAt(8);
            originalData.deleteAttributeAt(6);
            originalData.deleteAttributeAt(3);
        }

        if(removeRedundantManual && name.equals(ZOO)){
            originalData.deleteAttributeAt(15);
            originalData.deleteAttributeAt(0);
        }

        originalData.setClassIndex(originalData.numAttributes()-1);

        return originalData;
    }

    private static String intArrToStr(int [] arr){
        sortIntArr(arr);
        StringBuilder ret = new StringBuilder("[ ");
        for(int i : arr){
            ret.append(i).append(" ");
        }
        return ret+"]";
    }

    private static final String VOWEL = "vowel";
    private static final String BREAST_W = "breast-w";
    private static final String ZOO = "zoo";

    public static void main(String [] args) throws Exception {
        List<String> dataSets = new ArrayList<>();
        //dataSets.add("hepatitis");
//        dataSets.add("breast-cancer");
//        dataSets.add(ZOO);
//        dataSets.add("page-blocks");
//        dataSets.add(VOWEL);
//        dataSets.add(BREAST_W);
//        dataSets.add("heart-statlog");
//        dataSets.add("diabetes");
          dataSets.add("credit-g");

        OutputFormatter outF = new OutputFormatter("out/test.csv");
        List<String> badSets = new ArrayList<>();
        for (String dataSet : dataSets) {
            try {
                System.out.println("------------  "+dataSet+"  ------------");
                Instances originalData = getDataSet(dataSet);

                saveDataSetInfo(new DataSetInfo(dataSet, originalData.numAttributes()-1, originalData.numInstances(), originalData.numClasses()));

                LandscapeEvaluator filterEval = new LandscapeEvaluator(new IBkClassifier());
                Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);

//                for(ConcreteBinarySolution sol : fitnessMap.keySet()){
//                    if(sol.getIndex() == 506){
//                        System.out.println("X");
//                    }
//                }

                HashMap<BinarySolution, Weight> optimaBasins = new HashMap<>();
                HashMap<BinarySolution, Double> optimaQuality = new HashMap<>();
                HashMap<BinarySolution,HashMap<BinarySolution,Weight>> mapOfAdjacencyListAndWeight = new HashMap<>();

                LONGenerator.exhaustiveLON(new FSBinaryProblem(fitnessMap), new BinaryHammingNeighbourhood(),
                        optimaBasins,
                        optimaQuality,
                        mapOfAdjacencyListAndWeight,
                        EdgeType.ESCAPE_EDGE);

                System.out.println("Done initiating landscape and constructing LONs");

                RVisualizationFormatter.format(dataSet, optimaBasins, optimaQuality, mapOfAdjacencyListAndWeight, true);

                List<BinarySolution> globalOptima = findGlobalOptima(optimaQuality);

                System.out.println("NUMBER OF GLOBAL OPTIMA: "+globalOptima.size());

                System.out.println("Applying filter method");
                FSBundle filterMethodResult = FeatureSelectorUtils.getFilterMethodAttributes(getDataSet(dataSet), fitnessMap);
//                System.out.println("Applying SFS wrapper method");
//                FSBundle wrapperMethodResult = FeatureSelectorUtils.getWrapperMethodAttributes(getDataSet(dataSet));
                System.out.println("Applying GA wrapper method");
                GAResult wrapperGAMethodResult = FeatureSelectorUtils.getWrapperGAAttributes(getDataSet(dataSet), globalOptima, fitnessMap);
                System.out.println("Applying SFS wrapper method");
                FSBundle wrapperGreedyStepwise = FeatureSelectorUtils.getWrapperGreedyStepwiseAttributes(getDataSet(dataSet), fitnessMap);

                double globalFitness = fitnessMap.get(globalOptima.get(0));

                FSReportInfo filter =  new FSReportInfo(
                        "Filter",
                        dataSet,
                        globalOptima.size(),
                        optimaQuality.keySet().contains(filterMethodResult.getBinarySolution()),
                        globalOptima.contains(filterMethodResult.getBinarySolution()),
                        calculatePercentageCap(filterMethodResult.getAccuracy(), globalFitness),
                        -1.0,
                        filterMethodResult.getAccuracy(),
                        globalFitness
                );
                saveFSInfo(filter);

                FSReportInfo GA =  new FSReportInfo(
                        "GA",
                        dataSet,
                        globalOptima.size(),
                        optimaQuality.keySet().contains(wrapperGAMethodResult.getBestBundle().getBinarySolution()),
                        globalOptima.contains(wrapperGAMethodResult.getBestBundle().getBinarySolution()),
                        calculatePercentageCap(wrapperGAMethodResult.getMeanAccuracy(), globalFitness),
                        wrapperGAMethodResult.getSuccessRatio(),
                        wrapperGAMethodResult.getMeanAccuracy(),
                        globalFitness
                );
                saveFSInfo(GA);

                FSReportInfo SFS =  new FSReportInfo(
                        "SFS",
                        dataSet,
                        globalOptima.size(),
                        optimaQuality.keySet().contains(wrapperGreedyStepwise.getBinarySolution()),
                        globalOptima.contains(wrapperGreedyStepwise.getBinarySolution()),
                        calculatePercentageCap(wrapperGreedyStepwise.getAccuracy(), globalFitness),
                        -1.0,
                        wrapperGreedyStepwise.getAccuracy(),
                        globalFitness
                );
                saveFSInfo(SFS);

//                //filter method
//                addLocalOptimaInfoForFeatureSelectionAlgorithm(outF, "Filter Method",
//                        filterMethodResult.getBinarySolution(),
//                        filterMethodResult.getAccuracy(),
//                        optimaQuality,
//                        globalOptima
//                        );
//
//                //wrapper method 2
//                addLocalOptimaInfoForFeatureSelectionAlgorithm(outF, "Wrapper GA Method",
//                        wrapperGAMethodResult.getBestBundle().getBinarySolution(),
//                        wrapperGAMethodResult.getMeanAccuracy(),
//                        optimaQuality,
//                        globalOptima
//                );
//
//                //wrapper method 3
//                addLocalOptimaInfoForFeatureSelectionAlgorithm(outF, "Wrapper GreedyStepwise Method",
//                        wrapperGreedyStepwise.getBinarySolution(),
//                        wrapperGreedyStepwise.getAccuracy(),
//                        optimaQuality,
//                        globalOptima
//                );

                StringBuilder fsSolutions = new StringBuilder();

                fsSolutions.append(originalData.numAttributes()-1).append("\n");

                fsSolutions.append(filterMethodResult.getBinarySolution().getIndex())
                             .append(" ")
                             .append("Fil:"+filterMethodResult.getAccuracy())
                             .append("\n");

//                fsSolutions.append(wrapperBinarySolution.getIndex())
//                        .append(" ")
//                        .append("SFS:"+wrapperMethodResult.getAccuracy())
//                        .append("\n");

                fsSolutions.append(wrapperGAMethodResult.getBestBundle().getBinarySolution().getIndex())
                        .append(" ")
                        .append("GA:"+wrapperGAMethodResult.getMeanAccuracy())
                        .append("\n");

                fsSolutions.append(wrapperGreedyStepwise.getBinarySolution().getIndex())
                        .append(" ")
                        .append("SFS:"+wrapperGreedyStepwise.getAccuracy())
                        .append("\n");

                try (PrintWriter out = new PrintWriter("../"+dataSet +".fs")) {
                    out.print(fsSolutions.toString());
                }

                try {
                    outF.save();
                } catch (FileNotFoundException | UnsupportedEncodingException ex) {
                    ex.printStackTrace();
                }

                //Done with the LON visualization formatting
            }catch (Exception e){

                System.err.println(e.getMessage());
                System.err.println("Skipping "+dataSet);
                badSets.add(dataSet);
                throw e;
            }
        }

        flushDataSetInfo(outF);
        flushFSInfo(outF);

        outF.save();


        for(String s : badSets){
            System.err.println(s);
        }
    }

    public static double calculatePercentageCap(double bestFitness, double globalOptimaFitness){
        BigDecimal bd = new BigDecimal(1.0 - ((globalOptimaFitness - bestFitness)/globalOptimaFitness));
        bd = bd.setScale(DECIMAL_PLACES, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


    private static List<BinarySolution> findGlobalOptima(HashMap<BinarySolution, Double> optimaQuality){
        List<BinarySolution> globalOptimas = Lists.newArrayList();
        BinarySolution best = null;
        Double bestFitness = null;
        for(BinarySolution solution : optimaQuality.keySet()){

            if(best == null){
                best = solution;
                bestFitness = optimaQuality.get(solution);
            }else{
                Double currentFitness = optimaQuality.get(solution);
                if(BigDecimal.valueOf(currentFitness).compareTo(BigDecimal.valueOf(bestFitness))  >= 0){
                    if(BigDecimal.valueOf(currentFitness).compareTo(BigDecimal.valueOf(bestFitness)) > 0){
                        globalOptimas.clear();
                    }

                    best = solution;
                    bestFitness = currentFitness;
                    globalOptimas.add(solution);
                }
            }
        }

        return globalOptimas;

    }


    private static int dataSetNumber = 0;


    private static void buildDataSetInfo(OutputFormatter outF, String name, int noAttrs, int noInstances, int noClasses){
        if(dataSetNumber == 0){
            outF.addAsColumns("Datasets");
            outF.addAsColumns("Index", "Name", "No. Attributes", "No. Instances", "No. Classes");
        }

        outF.addAsColumns(Integer.toString(dataSetNumber++), name, Integer.toString(noAttrs),
                Integer.toString(noInstances), Integer.toString(noClasses));
    }

    public static class DataSetInfo{
        private String name;
        int noAttrs,  noInstances,  noClasses;

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
    private static List<DataSetInfo> dataSetReport = Lists.newArrayList();
    private static void saveDataSetInfo(DataSetInfo info){
        dataSetReport.add(info);
    }
    private static void flushDataSetInfo(OutputFormatter outF){
        for(DataSetInfo dataSetInfo : dataSetReport){
            buildDataSetInfo(outF, dataSetInfo.getName(), dataSetInfo.getNoAttrs(), dataSetInfo.getNoInstances(), dataSetInfo.getNoClasses());
        }
    }

    private static List<String> algorithmNames = Lists.newArrayList();
    private static void buildFSInfo(OutputFormatter outF, String algorithmName, String dataSetName, int numberOfGlobalOptima,
                                    boolean foundLocalOpt, boolean foundGlobalOpt, double percentageCap, double successRatio, double fsAccuracy, double globalAccuracy){
        if(!algorithmNames.contains(algorithmName)){
            algorithmNames.add(algorithmName);
            outF.addEmptyRow();
            outF.addAsColumns(algorithmName);
            outF.addAsColumns("Dataset name", "No. Global Optima", "Found Local Optima", "Found Global Optima", "Percentage Cap", "Success Ratio", "Feat. S. Fitness", "Global Opt. Fitness");
        }

        outF.addAsColumns(dataSetName, Integer.toString(numberOfGlobalOptima), Boolean.toString(foundLocalOpt),
                Boolean.toString(foundGlobalOpt), Double.toString(percentageCap), Double.toString(successRatio), Double.toString(fsAccuracy), Double.toString(globalAccuracy));

    }

    private static class FSReportInfo{
        private String algorithmName;
        private String dataSetName;
        private int numberOfGlobalOptima;
        private boolean foundLocalOpt;
        private boolean foundGlobalOpt;
        private double percentageCap;
        private double fsAccuracy;
        private double globalAccuracy;
        private double successRatio;

        public FSReportInfo(String algorithmName, String dataSetName, int numberOfGlobalOptima, boolean foundLocalOpt, boolean foundGlobalOpt, double percentageCap, double successRatio, double fsAccuracy, double globalAccuracy) {
            this.algorithmName = algorithmName;
            this.dataSetName = dataSetName;
            this.numberOfGlobalOptima = numberOfGlobalOptima;
            this.foundLocalOpt = foundLocalOpt;
            this.foundGlobalOpt = foundGlobalOpt;
            this.percentageCap = percentageCap;
            this.fsAccuracy = fsAccuracy;
            this.globalAccuracy = globalAccuracy;
            this.successRatio = successRatio;
        }

        public double getSuccessRatio() {
            return successRatio;
        }

        public String getAlgorithmName() {
            return algorithmName;
        }

        public String getDataSetName() {
            return dataSetName;
        }

        public int getNumberOfGlobalOptima() {
            return numberOfGlobalOptima;
        }

        public boolean isFoundLocalOpt() {
            return foundLocalOpt;
        }

        public boolean isFoundGlobalOpt() {
            return foundGlobalOpt;
        }

        public double getPercentageCap() {
            return percentageCap;
        }

        public double getFsAccuracy() {
            return fsAccuracy;
        }

        public double getGlobalAccuracy() {
            return globalAccuracy;
        }
    }

    private static HashMap<String, HashMap<String, FSReportInfo>> fsReportInfo = Maps.newHashMap();

    private static void saveFSInfo(FSReportInfo reportInfo){
        HashMap<String, FSReportInfo> algorithmReport = fsReportInfo.get(reportInfo.algorithmName);
        if(algorithmReport == null){
            algorithmReport = Maps.newHashMap();
        }

        algorithmReport.put(reportInfo.getDataSetName(), reportInfo);
        fsReportInfo.put(reportInfo.getAlgorithmName(), algorithmReport);
    }

    private static void flushFSInfo(OutputFormatter outF){
        for(String algorithmName : fsReportInfo.keySet()){
            for(String dataSetName : fsReportInfo.get(algorithmName).keySet()) {
                FSReportInfo report = fsReportInfo.get(algorithmName).get(dataSetName);
                buildFSInfo(outF, algorithmName, dataSetName, report.getNumberOfGlobalOptima(), report.isFoundLocalOpt(),
                        report.isFoundGlobalOpt(), report.getPercentageCap(), report.getSuccessRatio(), report.getFsAccuracy(), report.getGlobalAccuracy());
            }
        }
    }


//    private static void addLocalOptimaInfoForFeatureSelectionAlgorithm(OutputFormatter outF, String algorithmDescription,
//                                                                       BinarySolution featureSelectionSolution,
//                                                                       double featureSelectionFitness,
//                                                                       HashMap<BinarySolution, Double> optimaQuality,
//                                                                       List<BinarySolution> globalOptima){
//        outF.addEmptyRow();
//        outF.addAsColumns("Feature Selection Algorithm: ", algorithmDescription);
//        outF.nextRow();
//        outF.addAsColumns("Solution Index : ", Integer.toString(featureSelectionSolution.getIndex()));
//        outF.nextRow();
//        outF.addAsColumns("FS Solution Fitness : ", Double.toString(featureSelectionFitness));
//        outF.nextRow();
//        outF.addAsColumns("Is a local optima?", optimaQuality.get(featureSelectionSolution) == null ? "NO" : "YES");
//        outF.nextRow();
//
//        for(int i = 0; i < globalOptima.size(); i++) {
//            outF.addAsColumns("Global optima " + i +" index : ", Integer.toString(globalOptima.get(i).getIndex()));
//            outF.nextRow();
//        }
//        outF.addAsColumns("Global optima fitness : ", Double.toString(optimaQuality.get(globalOptima.get(0))));
//        outF.nextRow();
//        outF.addAsColumns("FS Solution is Global Optima?", globalOptima.contains(featureSelectionSolution) ? "YES" : "NO");
//
//    }


}
