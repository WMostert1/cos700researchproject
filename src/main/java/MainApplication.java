import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by bbdnet1339 on 2016/08/05.
 *
 */
public class MainApplication {

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

    private static DataSource getSource(String name){
        return new DataSource(MainApplication.class.getResourceAsStream("data-sets/used/" + name));
    }

    private static String intArrToStr(int [] arr){
        sortIntArr(arr);
        StringBuilder ret = new StringBuilder("[ ");
        for(int i : arr){
            ret.append(i).append(" ");
        }
        return ret+"]";
    }

    public static void main(String [] args) throws Exception {
        List<String> dataSets = new MainApplication().getResourceFiles("data-sets/used");

        dataSets = new ArrayList<String>();
        dataSets.add("anneal.ORIG.arff");
        dataSets.add("audiology.arff");
        dataSets.add("colic.ORIG.arff");
        dataSets.add("cylinder-bands.arff");
        dataSets.add("hepatitis.arff");
        dataSets.add("page-blocks.arff");
        dataSets.add("vowel.arff");

        //if (args != null && args.length != 0){

        
            List<String> badSets = new ArrayList<>();
            for (String dataSet : dataSets) {
                
                try {
                    OutputFormatter outF = new OutputFormatter("out/"+dataSet+".HAMMING.csv");
                    Instances originalData = getSource(dataSet).getDataSet();
                   // outF.addAsColumns(new String[]{dataSet,Integer.toString(originalData.numAttributes()-1) + "Features ", Integer.toString(originalData.numInstances())+" Instances"});
                    
//                    OutputFormatter outF = new OutputFormatter("out/" + dataSet.replace(".arff", "") + ".csv");

//                    outF.addAsColumns(new String[]{"--- FILTER-METHOD ---"});
//                    outF.addEmptyRow();

                    LandscapeEvaluator filterEval = new LandscapeEvaluator(50, new UniformSampleMutator(), new IBkClassifier(), dataSet, outF);
//                    FSBundle filterBundle = FeatureSelectorUtils.getFilterMethodAttributes(getSource(dataSet));
//                    //Instances filterData = filterBundle.getData();
//                    Instances origFilterData = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(originalData, filterBundle.getAttributes());
                    filterEval.eval(originalData);
//
////                    outF.addEmptyRow();
////                    outF.addAsColumns(new String[]{"--- WRAPPER-METHOD ---"});
////                    outF.addEmptyRow();
//
//                    LandscapeEvaluator wrapperEval = new LandscapeEvaluator(50, new UniformSampleMutator(), new IBkClassifier(), dataSet, outF);
//                    FSBundle wrapperBundle = FeatureSelectorUtils.getWrapperMethodAttributes(getSource(dataSet));
//                    Instances origWrapperData = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(originalData, wrapperBundle.getAttributes());
//                    //Instances wrapperData = wrapperBundle.getData();
//                    wrapperEval.eval(origWrapperData);
//
////                    outF.addEmptyRow();
////                    outF.addAsColumns(new String[]{"--- EMBEDDED-METHOD ---"});
////                    outF.addEmptyRow();
//
//
//                    LandscapeEvaluator embeddedEval = new LandscapeEvaluator(50, new UniformSampleMutator(), new IBkClassifier(), dataSet, outF);
//                    FSBundle embeddedBundle = FeatureSelectorUtils.getEmbeddedMethodAttributes(getSource(dataSet));
//                    //Instances embeddedData = embeddedBundle.getData();
//                    Instances origEmbdeddedData = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(originalData, embeddedBundle.getAttributes());
//                    embeddedEval.eval(origEmbdeddedData);
//                    System.out.println(dataSet);
//
//                    System.out.println("Filter Method attributes " + (filterData.numAttributes() - 1) + " accuracy " + filterBundle.getAccuracy()+" features "+intArrToStr(filterBundle.getAttributes()));
//                    System.out.println("Wrapper Method attributes " + (wrapperData.numAttributes() - 1) + " accuracy " + wrapperBundle.getAccuracy()+" features "+intArrToStr(wrapperBundle.getAttributes()));
//                    System.out.println("Embedded Method attributes " + (embeddedData.numAttributes() - 1) + " accuracy " + embeddedBundle.getAccuracy()+" features "+intArrToStr(embeddedBundle.getAttributes()));
//                    outF.nextRow();
//                    outF.addAsColumns(new String[]{"Type", "Accuracy", "Feature Count Used","Features"});
//                    outF.nextRow();
//                    outF.addAsColumns(new String[]{"Filter Method", Double.toString(filterBundle.getAccuracy()), Integer.toString((filterData.numAttributes() - 1)), intArrToStr(filterBundle.getAttributes())});
//                    outF.nextRow();
//                    outF.addAsColumns(new String[]{"Wrapper Method", Double.toString(wrapperBundle.getAccuracy()), Integer.toString((wrapperData.numAttributes() - 1)), intArrToStr(wrapperBundle.getAttributes())});
//                    outF.nextRow();
//                    outF.addAsColumns(new String[]{"Embedded Method", Double.toString(embeddedBundle.getAccuracy()), Integer.toString((embeddedData.numAttributes() - 1)), intArrToStr(embeddedBundle.getAttributes())});
//                    outF.nextRow();
//                    outF.addEmptyRow();

                    try {
                        outF.save();
                    } catch (FileNotFoundException | UnsupportedEncodingException e) {
                        e.printStackTrace();
                    }
                }catch (Exception e){
                    System.err.println("Skipping "+dataSet);
                    badSets.add(dataSet);
                }
            }


            for(String s : badSets){
                System.err.println(s);
            }

//        }else{
//            System.out.println("Supply path data set to use.");
//        }
    }
}
