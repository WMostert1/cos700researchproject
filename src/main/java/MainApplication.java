import jxl.Workbook;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by bbdnet1339 on 2016/08/05.
 *
 */
public class MainApplication {
    public static void main(String [] args) throws Exception {
        String [] dataSets = new String[]{
//                "anneal.ORIG.arff",
//                //"arrhythmia.arff",
//                "audiology.arff",
//               // "autos.arff",
//                "balance-scale.arff",
//                "breast-w.arff",
//                "letter.arff"
//                "bridges_version1.arff",
                //"car.arff"
                "autos.arff"
        };

        for (String dataSet : dataSets) {
            DataSource source = new DataSource(MainApplication.class.getResourceAsStream("data-sets/used/" + dataSet));
            Instances data = source.getDataSet();

            RandomWalk walk = new RandomWalk(300, new UniformBitMutator(0.1), new IBkClassifier(), dataSet);
            walk.doWalk(data);
        }


    }
}
