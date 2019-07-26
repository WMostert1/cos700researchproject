package utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

/**
 * Created by bbdnet1339 on 2016/08/12.
 */
public class OutputFormatter {
    private String fileName;

    public OutputFormatter(String fileName) {
        this.fileName = fileName;
        csvString = "";
    }

    //This is for Excel
    private String csvString;

    public void addEmptyRow(){
        csvString+="\n\n";
    }

    public void nextRow(){
        csvString += "\n";
    }

    public void addAsColumns(String ... vals){
        int i = 0;
        for(;i<vals.length-1;i++){
            csvString += vals[i]+";";
        }
        csvString += vals[i];
        nextRow();
    }


    public void save() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(fileName, "UTF-8");
        writer.write(csvString);
        writer.close();
    }
}
