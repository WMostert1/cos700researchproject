import lons.EdgeType;
import lons.LONGenerator;
import lons.RVisualizationFormatter;
import lons.Weight;
import lons.examples.BinaryHammingNeighbourhood;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import lons.examples.FSBinaryProblem;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JournalArticleApplication {


    private InputStream getResourceAsStream(String resource ) {
        final InputStream in
                = getContextClassLoader().getResourceAsStream( resource );

        return in == null ? getClass().getResourceAsStream( resource ) : in;
    }

    private ClassLoader getContextClassLoader() {
        return Thread.currentThread().getContextClassLoader();
    }



    private static Instances getDataSet(String name) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(EvoCOPPaperApplication.class.getResourceAsStream("data-sets/large/" + name+".arff"));
        Instances originalData = source.getDataSet();
        originalData.setClassIndex(originalData.numAttributes()-1);

        return originalData;
    }
    public static void main(String [] args) throws Exception {
        List<String> dataSets = new ArrayList<>();
        dataSets.add("autos");

        List<String> badSets = new ArrayList<>();
        for (String dataSet : dataSets) {
            try {
                System.out.println("------------  " + dataSet + "  ------------");
                Instances originalData = getDataSet(dataSet);


                LandscapeEvaluator filterEval = new LandscapeEvaluator(new IBkClassifier());
                Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);

                HashMap<BinarySolution, Weight> optimaBasins = new HashMap<>();
                HashMap<BinarySolution, Double> optimaQuality = new HashMap<>();
                HashMap<BinarySolution, HashMap<BinarySolution, Weight>> mapOfAdjacencyListAndWeight = new HashMap<>();

                LONGenerator.exhaustiveLON(new FSBinaryProblem(fitnessMap), new BinaryHammingNeighbourhood(),
                        optimaBasins,
                        optimaQuality,
                        mapOfAdjacencyListAndWeight,
                        EdgeType.ESCAPE_EDGE);

                RVisualizationFormatter.format(dataSet, optimaBasins, optimaQuality, mapOfAdjacencyListAndWeight, true);
            } catch (Exception e) {

                System.err.println(e.getMessage());
                System.err.println("Skipping " + dataSet);
                badSets.add(dataSet);
                throw e;
            }
        }

        for(String s : badSets){
            System.err.println(s);
        }
    }
}
