package utils;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class CompositeOutputFormatter extends OutputFormatter {
    List<OutputFormatter> outputFormatters;


    public CompositeOutputFormatter(OutputFormatter ... outputFormatters) {
        super(null, "");
        this.outputFormatters = Arrays.asList(outputFormatters);
    }

    @Override
    protected void addHeaders(final String... headers) {
        outputFormatters.forEach((of)->of.addHeaders(headers));
    }

    @Override
    public void addEmptyRow() {
        outputFormatters.forEach(OutputFormatter::addEmptyRow);
    }

    @Override
    public void nextRow() {
        outputFormatters.forEach(OutputFormatter::nextRow);
    }

    @Override
    public void addAsColumns(final String... vals) {
        outputFormatters.forEach((of)->of.addAsColumns(vals));
    }

    @Override
    public void save() throws FileNotFoundException, UnsupportedEncodingException {
        for(OutputFormatter of: outputFormatters){
            of.save();
        }
    }
}
