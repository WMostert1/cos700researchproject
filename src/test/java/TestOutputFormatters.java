import com.google.common.collect.Sets;
import org.junit.Test;
import utils.LatexOutputFormatter;
import utils.OutputFormatter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

public class TestOutputFormatters {

    @Test
    public void testLatex() throws FileNotFoundException, UnsupportedEncodingException {
        OutputFormatter outF = new LatexOutputFormatter("out/test.tex", "Test Caption", "test:label", "Column A", "Column B", "Column C");
        outF.addAsColumns("test string", BigDecimal.valueOf(78.123456), 1002.123);
        outF.addAsColumns(BigDecimal.valueOf(0.01123456), 0.12345, "x");
        outF.save();
    }


    @Test
    public void voidTestUrbanLandGenerate() throws Exception {
        StringBuilder arffOutput = new StringBuilder("@relation 'urbanland'\n\n");

        File dataFile = new File("/Users/wmostert/Development/cos700researchproject/src/main/resources/data-sets/urbanland.csv");
        Scanner dataReader = new Scanner(dataFile);
        StringBuilder finalData = new StringBuilder();

        Set<String> uniqueLabels = Sets.newHashSet();
        //skip the first line
        int numberOfFeatures = dataReader.nextLine().split(",").length;
        while (dataReader.hasNextLine()) {
            String data = dataReader.nextLine();
            if (data.isEmpty()) {
                continue;
            }

            List<String> dataList = new LinkedList<>(Arrays.asList(data.split(",")));
            String classVal = dataList.get(0);
            uniqueLabels.add(classVal);
            dataList.remove(0);
            dataList.add(classVal);

            finalData.append(String.join(",", dataList)).append("\n");
        }

        for (long i = 0L; i < numberOfFeatures - 1; i++) {
            arffOutput.append("@attribute X").append(i).append(" numeric\n");
        }

        arffOutput.append("@attribute class {").append(String.join(",", uniqueLabels)).append("}\n\n");

        arffOutput.append("@data\n");
        arffOutput.append(finalData);

        FileWriter myWriter = new FileWriter("/Users/wmostert/Development/cos700researchproject/src/main/resources/data-sets/used/urbanland.arff");
        myWriter.write(arffOutput.toString());
        myWriter.close();
    }

    @Test
    public void testArceneArffGenerate() throws Exception {
        StringBuilder arffOutput = new StringBuilder("@relation 'arcene'\n\n");

        File dataFile = new File("/Users/wmostert/Development/cos700researchproject/src/main/resources/data-sets/arcene_train.data");
        File labelFile = new File("/Users/wmostert/Development/cos700researchproject/src/main/resources/data-sets/arcene_train.labels");
        Scanner dataReader = new Scanner(dataFile);
        Scanner labelReader = new Scanner(labelFile);
        StringBuilder finalData = new StringBuilder();

        Set<String> uniqueLabels = Sets.newHashSet();

        while (dataReader.hasNextLine()) {
            String data = dataReader.nextLine();
            String label = labelReader.nextLine();
            uniqueLabels.add(label);
            if (data.isEmpty()) {
                continue;
            }
            data = data.replace(" ", ",").replace("\n","").replace("\r","");

            finalData.append(data).append(",").append(label).append("\n");
        }
        dataReader.close();

        for (long i = 0L; i < 10000; i++) {
            arffOutput.append("@attribute X").append(i).append(" numeric\n");
        }
        arffOutput.append("@attribute class {").append(String.join(",", uniqueLabels)).append("}\n\n");

        arffOutput.append("@data\n");
        arffOutput.append(finalData);

        FileWriter myWriter = new FileWriter("/Users/wmostert/Development/cos700researchproject/src/main/resources/data-sets/used/arcene.arff");
        myWriter.write(arffOutput.toString());
        myWriter.close();
    }

}
