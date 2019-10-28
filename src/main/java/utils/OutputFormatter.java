package utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.Arrays;

import static utils.GlobalConstants.REPORTED_DECIMAL_PLACES;
import static utils.MathUtils.bigDecimalToString;
import static utils.MathUtils.doubleToString;

/**
 * Created by bbdnet1339 on 2016/08/12.
 */
public abstract class OutputFormatter {

    protected String fileName;
    protected String outputString;
    protected final int numberOfColumns;

    public OutputFormatter(String fileName, String... headers) {
        this.fileName = fileName;
        this.outputString = "";
        this.numberOfColumns = headers.length;
    }

    protected void append(String val) {
        this.outputString += val;
    }

    protected abstract void addHeaders(String... headers);

    public abstract void addEmptyRow();

    public abstract void nextRow();

    public void addAsColumns(Object... vals) {
        addAsColumns(Arrays.stream(vals).map((val) -> {
            if (val instanceof BigDecimal) {
                return bigDecimalToString((BigDecimal) val);
            }
            if (val instanceof Double) {
                return doubleToString((Double) val);
            }
            return val.toString();
        }).toArray(String[]::new));
    }

    public abstract void addAsColumns(String... vals);

    public abstract void save() throws FileNotFoundException, UnsupportedEncodingException;


}
