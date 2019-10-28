package utils;


import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class CsvOutputFormatter extends OutputFormatter {
    //This is for Excel

    public CsvOutputFormatter(final String fileName, String... headers) {
        super(fileName, headers);
        this.addHeaders(headers);
    }

    @Override
    protected void addHeaders(final String... headers) {
        addAsColumns(headers);
    }

    @Override
    public void addEmptyRow() {
        append("\n\n");
    }

    @Override
    public void nextRow() {
        append("\n");
    }

    @Override
    public void addAsColumns(String... vals) {
//        if (vals.length > this.numberOfColumns) {
//            throw new RuntimeException("Recording more columns than number of rows!");
//        }

        for (final String val : vals) {
            if(val == null){
                append("null;");
            }else {
                append(val.replace(".", ",") + ";");
            }
        }

        nextRow();
    }

    @Override
    public void save() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(fileName, "UTF-8");
        writer.write(this.outputString);
        writer.close();
    }
}
