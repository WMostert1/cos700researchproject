package lons;

import lons.examples.BinarySolution;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class RVisualizationFormatter {
    public static void format(String dataSetName, HashMap<BinarySolution, Weight> optimaBasins,
                              HashMap<BinarySolution, Double> optimaQuality,
                              HashMap<BinarySolution,HashMap<BinarySolution,Weight>> mapOfAdjacencyListAndWeight) throws IOException {
        RVisualizationFormatter.format(dataSetName, optimaBasins, optimaQuality, mapOfAdjacencyListAndWeight, false);
    }

    public static void format(String dataSetName, HashMap<BinarySolution, Weight> optimaBasins,
                              HashMap<BinarySolution, Double> optimaQuality,
                              HashMap<BinarySolution,HashMap<BinarySolution,Weight>> mapOfAdjacencyListAndWeight,
                              boolean useLogScale) throws IOException {
        StringBuilder nodesBuilder = new StringBuilder();

        //Start building the format for the local optima network visualization tool

        for(BinarySolution node : optimaBasins.keySet()){
            nodesBuilder.append(Integer.toString(node.getIndex()))
                    .append(" ")
                    .append(Double.toString(optimaQuality.get(node)))
                    .append(" ")
                    .append(Double.toString(useLogScale ? Math.log((double) optimaBasins.get(node).getWeight()) : optimaBasins.get(node).getWeight()))
                    .append("\n");
        }

        try (PrintWriter out = new PrintWriter("../"+dataSetName +".nodes")) {
            out.print(nodesBuilder.toString());
        }

        StringBuilder edgesBuilder = new StringBuilder();

        for(BinarySolution nodeEdges : mapOfAdjacencyListAndWeight.keySet()){
            for(BinarySolution edge : mapOfAdjacencyListAndWeight.get(nodeEdges).keySet()){
                edgesBuilder.append(Integer.toString(nodeEdges.getIndex()))
                        .append(" ")
                        .append(Integer.toString(edge.getIndex()))
                        .append(" ")
                        .append(Double.toString(mapOfAdjacencyListAndWeight.get(nodeEdges).get(edge).getWeight()))
                        .append("\n");
            }
        }

        try (PrintWriter out = new PrintWriter("../"+dataSetName +".edges")) {
            out.print(edgesBuilder.toString());
        }

        zip(dataSetName, edgesBuilder, nodesBuilder);
    }

    private static void zip(String dataSetName, StringBuilder edges, StringBuilder nodes) throws IOException {

        File f = new File("../"+dataSetName +".zip");
        ZipOutputStream out = new ZipOutputStream(new FileOutputStream(f));
        ZipEntry edgesZip = new ZipEntry(dataSetName+".edges");
        out.putNextEntry(edgesZip);
        byte[] edgesData = edges.toString().getBytes();
        out.write(edgesData, 0, edgesData.length);
        out.closeEntry();

        ZipEntry nodesZip = new ZipEntry(dataSetName+".nodes");
        out.putNextEntry(nodesZip);
        byte[] nodesData = nodes.toString().getBytes();
        out.write(nodesData, 0, nodesData.length);
        out.closeEntry();

        out.close();
    }
}
