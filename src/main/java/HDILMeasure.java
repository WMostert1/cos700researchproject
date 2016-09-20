import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by bbdnet1339 on 2016/08/11.
 *
 */
public class HDILMeasure implements IMeasure {
    private int numberOfIsoLevels;
    private double lowerBound;
    private double upperBound;
    private Map<Integer,Map<boolean[],Double>> isoLevels;
    private Map<Integer,Double> hdil;
    private OutputFormatter outF;

    public HDILMeasure(int numberOfIsoLevels, double lowerBound, double upperBound, OutputFormatter outF) {
        this.numberOfIsoLevels = numberOfIsoLevels;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        isoLevels = new HashMap<>();
        hdil = new HashMap<>();
        for(int i = 0; i < numberOfIsoLevels;i++){
            isoLevels.put(i,new HashMap<>());
            hdil.put(i,0.0);
        }
        this.outF = outF;
    }

    public void get(Map<boolean [], Double> results){
        double increment = (upperBound-lowerBound)/numberOfIsoLevels;
        results.forEach((arr,fitness)->{
            for(int i = 0; i < numberOfIsoLevels; i++){
                if(fitness >= lowerBound+(increment*i) && fitness < lowerBound+(increment*(i+1))){
                    isoLevels.get(i).put(arr,fitness);
                }
            }
        });

        for(int levelNo = 0; levelNo < numberOfIsoLevels; levelNo++){
          long hdilForLevel = 0;
           Object [] usages = isoLevels.get(levelNo).keySet().toArray();
          int A_2 = usages.length*usages.length;
           for(int i = 0; i < usages.length; i++ ){
               for(int j = 0; j < usages.length; j++){
                   hdilForLevel += calculateHammingDistance((boolean[])usages[i],(boolean [])usages[j]);
               }
           }
            if(hdilForLevel!=0)
            hdil.put(levelNo,(1.0/A_2)*(double)hdilForLevel);
        }

        outF.addAsColumns(new String[]{"Count per Level"});
        outF.nextRow();
        System.out.println("Count:");
        isoLevels.forEach((x,y)->{
            double f0 = lowerBound+(Math.abs(upperBound-lowerBound)/(double)numberOfIsoLevels*x);
            double f1 = lowerBound+(Math.abs(upperBound-lowerBound)/(double)numberOfIsoLevels*(x+1));

            outF.addAsColumns(new String[]{"Level",x.toString()+" f("+round(f0,1)+" -> "+round(f1,1)+")",Integer.toString(y.size())});
            outF.nextRow();
            System.out.println(x + " : " +y.size());
        });
        System.out.println("------------");
        System.out.println("HDIL:");
        outF.addAsColumns(new String[]{"HDIL"});
        outF.nextRow();

        outF.addEmptyRow();
        hdil.forEach((x,y)->{
            double f0 = lowerBound+(Math.abs(upperBound-lowerBound)/(double)numberOfIsoLevels*x);
            double f1 = lowerBound+(Math.abs(upperBound-lowerBound)/(double)numberOfIsoLevels*(x+1));
            outF.addAsColumns(new String[]{"Level ",x.toString()+" f("+round(f0,1)+" -> "+round(f1,1)+")",Double.toString(y)});
            outF.nextRow();
            System.out.println(x + " : " +y);
        });
        try {
            outF.save();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    private static double round (double value, int precision) {
        int scale = (int) Math.pow(10, precision);
        return (double) Math.round(value * scale) / scale;
    }

    public int calculateHammingDistance(boolean [] arr1, boolean [] arr2){
        ArrayList<Boolean> l1 = new ArrayList<>();
        ArrayList<Boolean> l2 = new ArrayList<>();
        for(boolean b : arr1) l1.add(b);
        for(boolean b : arr2) l2.add(b);
        while(l1.size() < l2.size()) l1.add(0,false);
        while(l2.size() < l1.size()) l2.add(0,false);
        int count = 0;
        for(int i = 0; i < l1.size();i++){
            if(l1.get(i) == l2.get(i))
                count++;
        }
        return count;
    }


}
