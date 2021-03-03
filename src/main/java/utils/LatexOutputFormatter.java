package utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import static utils.latex.LatexItalic.latexItalic;

public class LatexOutputFormatter extends OutputFormatter {

    private final String caption;

    private final String label;

    public LatexOutputFormatter(final String fileName, String caption, String label, final String... headers) {
        super(fileName, headers);
        this.caption = caption;
        this.label = label;
        this.addHeaders(headers);
    }

    @Override
    protected void addHeaders(final String... headers) {
        String header = "\\begin{table*}[h]\n" +
                "\\caption{" + caption + "}\n" +
                "\\label{" + label + "}";

        StringBuilder columnsDefinition = new StringBuilder("");
        StringBuilder columnNames = new StringBuilder();
        for (String h : headers) {
            columnsDefinition.append("l");
            columnNames.append("&").append(h);
        }
        columnNames.delete(0, 1);
        columnNames.append("\\\\");
        header += "\\begin{tabular}{" + columnsDefinition + "}\n" +
                "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n" +
                columnNames +
                "\\noalign{\\smallskip}\\hline\n";

        append(header);
    }

    @Override
    public void addEmptyRow() {
        String val = "";
        for (int i = 0; i < this.numberOfColumns; i++) {
            val += "&";
        }
        append(val);
        nextRow();
    }

    @Override
    public void nextRow() {
        append("\n");
    }

    @Override
    public void addAsColumns(final String... vals) {
        if (vals.length > this.numberOfColumns) {
            throw new RuntimeException("Recording more columns than number of rows!");
        }

        StringBuilder rowVal = new StringBuilder();
        for (final String val : vals) {
            rowVal.append("&").append(val);
        }
        rowVal.delete(0,1);
        append(rowVal.toString());
        append("\\\\");
        nextRow();
    }

    @Override
    public void save() throws FileNotFoundException, UnsupportedEncodingException {
        String end = "\\noalign{\\smallskip}\\hline\n\\end{tabular}\n" +
                "\\end{table*}";
        append(end);

        PrintWriter writer = new PrintWriter(fileName, "UTF-8");
        writer.write(this.outputString);
        writer.close();
    }
}
