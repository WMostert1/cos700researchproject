import lons.EdgeType;
import lons.LONGenerator;
import lons.Weight;
import lons.examples.*;
import mutators.UniformSampleMutator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


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

        List<String> dataSets = new ArrayList<String>();
//        dataSets.add("anneal.ORIG.arff");
//        dataSets.add("audiology.arff");
//        dataSets.add("colic.ORIG.arff");
//        dataSets.add("cylinder-bands.arff");
//        dataSets.add("hepatitis.arff");
//        dataSets.add("page-blocks.arff");
                dataSets.add("vowel.arff");
       // dataSets.add("diabetes.arff");


//

        List<String> badSets = new ArrayList<>();
        for (String dataSet : dataSets) {

            try {
                OutputFormatter outF = new OutputFormatter("out/"+dataSet+".HAMMING.csv");
                Instances originalData = getSource(dataSet).getDataSet();
                originalData.setClassIndex(originalData.numAttributes()-1);
                LandscapeEvaluator filterEval = new LandscapeEvaluator(50, new UniformSampleMutator(), new IBkClassifier(), dataSet, outF);
                Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);


                HashMap<BinarySolution, Weight> optimaBasins = new HashMap<>();
                HashMap<BinarySolution, Double> optimaQuality = new HashMap<>();
                HashMap<BinarySolution,HashMap<BinarySolution,Weight>> mapOfAdjacencyListAndWeight = new HashMap<>();

                LONGenerator.exhaustiveLON(new FSBinaryProblem(fitnessMap), new BinaryHammingNeighbourhood(),
                        optimaBasins,
                        optimaQuality,
                        mapOfAdjacencyListAndWeight,
                        EdgeType.ESCAPE_EDGE);

                System.out.println("Done");


                try {
                    outF.save();
                } catch (FileNotFoundException | UnsupportedEncodingException e) {
                    e.printStackTrace();
                }
            }catch (Exception e){

                System.err.println(e.getMessage());
                System.err.println("Skipping "+dataSet);
                badSets.add(dataSet);
                throw e;
            }
        }


        for(String s : badSets){
            System.err.println(s);
        }
    }
}
