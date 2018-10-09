package lons;

import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.io.File;
import java.nio.file.Path;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * LONGenerator is essentially the entry point for the package, providing methods 
 * for Local Optima Network calculation and file writing.
 * 
 * @author Jonathan Fieldsend 
 * @version 6/7/2018
 */
public class LONGenerator
{
    private static int greedyHillclimbCalls; // counter to track how many times the internals of the greeyHillClimb function has been called

    /**
     * Method returns how many times the hillclimb method has been invoked since the counter was last reset
     *
     * @return number of hillclimb calls
     */
    public static int getGreedyHillclimbCalls() {
        return greedyHillclimbCalls;
    }
    
    /**
     * Method calculates an exact LON by exhaustive enumeration, and fills the data structure arguments with the results.
     *
     * @param problem problem defining landscape 
     * @param neighbourhood neighbourhood defining landscape
     * @param optimaBasins map of solutions which are local optima, and the corresponding weight of their basin -- should be passed in empty and will be filled on method returning 
     * @param optimaQuality map of solutions which are local optima, and their corresponding quality -- should be passed in empty and will be filled on method returning
     * @param mapOfAdjacencyListAndWeight map representing an adjacency list, and corersponding weight (key is local optima, and returns map of adjacent basins, and weights) -- should be passed in empty and will be filled on method returning
     * @param edgeType edge type to use in LON construction
     * 
     * @deprecated use {@link #exhaustiveLON()} instead. Code for illustrative purposes of less efficient implementation. 
     */
    @Deprecated
    public static <K extends Solution> void naiveExhaustiveLON(Problem<K> problem, Neighbourhood<K> neighbourhood,  HashMap<K,Weight> optimaBasins,HashMap<K,Double> optimaQuality,HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, EdgeType edgeType) {
        exhaustiveLON(problem,neighbourhood,optimaBasins,optimaQuality,mapOfAdjacencyListAndWeight,edgeType,false);
    }

    
    /**
     * Method calculates an exact LON by exhaustive enumeration, and fills the data structure arguments with the results.
     *
     * @param problem problem defining landscape 
     * @param neighbourhood neighbourhood defining landscape
     * @param optimaBasins map of solutions which are local optima, and the corresponding weight of their basin -- should be passed in empty and will be filled on method returning 
     * @param optimaQuality map of solutions which are local optima, and their corresponding quality -- should be passed in empty and will be filled on method returning
     * @param mapOfAdjacencyListAndWeight map representing an adjacency list, and corersponding weight (key is local optima, and returns map of adjacent basins, and weights) -- should be passed in empty and will be filled on method returning
     * @param edgeType edge type to use in LON construction
     */ 
    public static <K extends Solution> void exhaustiveLON(Problem<K> problem, Neighbourhood<K> neighbourhood, HashMap<K,Weight> optimaBasins, HashMap<K,Double> optimaQuality, HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, EdgeType edgeType) {
        exhaustiveLON(problem,neighbourhood,optimaBasins,optimaQuality,mapOfAdjacencyListAndWeight,edgeType,true);
    }

    
    /**
     * Method calculates an estimated LON by randomly sampling from the domain and hill-climbing from eash of these
     * random initial samples, and fills the data structure arguments with the results.
     *
     * @param problem problem defining landscape 
     * @param neighbourhood neighbourhood defining landscape
     * @param optimaBasins map of solutions which are local optima, and the corresponding weight of their basin -- should be passed in empty and will be filled on method returning 
     * @param optimaQuality map of solutions which are local optima, and their corresponding quality -- should be passed in empty and will be filled on method returning
     * @param mapOfAdjacencyListAndWeight map representing an adjacency list, and corersponding weight (key is local optima, and returns map of adjacent basins, and weights) -- should be passed in empty and will be filled on method returning
     * @param edgeType edge type to use in LON construction
     * @param numberOfSamples number of random samplesin legal domain to take to construct estimated LON
     * 
     * @deprecated use {@link #exhaustiveLON()} instead. Code for illustrative purposes of less efficient implementation. 
     */
    @Deprecated
    public static <K extends Solution> void naiveSampledLON(Problem<K> problem, Neighbourhood<K> neighbourhood, HashMap<K,Weight> optimaBasins, HashMap<K,Double> optimaQuality, HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, EdgeType edgeType, int numberOfSamples) {
        System.out.print("\n cleaning...");
        clean();
        HashSet<K> optima = new HashSet<>(); // set of optima

        // sample to get basin sizes
        for (int i=0; i<numberOfSamples; i++) {
            K randomDesign = problem.getRandomSolution();
            ArrayList<K> visitedOnPath = new ArrayList<>();
            K optimum = greedyHillclimb(randomDesign,problem, neighbourhood);
            optima.add(optimum);
            if (optimaBasins.containsKey(optimum)) {
                optimaBasins.get(optimum).increment();
            } else {
                optimaBasins.put(optimum,new Weight(1));
            }
            if (edgeType.equals(EdgeType.BASIN_TRANSITION)) {
                if (!mapOfAdjacencyListAndWeight.containsKey(optimum)) // check if optimum previously in map
                    mapOfAdjacencyListAndWeight.put(optimum,new HashMap<K,Weight>());
                for (K s : neighbourhood.neighbouringSolutions(randomDesign)) {
                    K neighboursOptimum = greedyHillclimb(s,problem, neighbourhood);
                    HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(optimum); //get map of adjacent optima
                    // now add neighbouring optima if not present, or update the weight
                    if (linkedOptima.containsKey(neighboursOptimum))
                        linkedOptima.get(neighboursOptimum).increment();
                    else
                        linkedOptima.put(neighboursOptimum,new Weight(1));    
                }
            }
        }
        // now approximate edges
        if (edgeType.equals(EdgeType.ESCAPE_EDGE)) {  
            for (K optimum : optima)
                mapOfAdjacencyListAndWeight.put(optimum,new HashMap<K,Weight>()); //initialise maps to adjacent optima, and their edge weights

            for (K o : optima) { //for each optima
                for (K s : neighbourhood.neighbouringSolutionsUpToDistance(o)) { // for each neighbour up to distance
                    K adjacentOptima = greedyHillclimb(s,problem, neighbourhood);

                    HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(o); //get map of adjacent optima
                    // now add neighbouring optima if not present, or update the weight
                    if (linkedOptima.containsKey(adjacentOptima))
                        linkedOptima.get(adjacentOptima).increment();
                    else
                        linkedOptima.put(adjacentOptima,new Weight(1));

                }
            }

        }

        System.out.println(optima.size());
    }

    /**
     * Method calculates an estimated LON by randomly sampling from the domain and hill-climbing from eash of these
     * random initial samples, and fills the data structure arguments with the results.
     *
     * @param problem problem defining landscape 
     * @param neighbourhood neighbourhood defining landscape
     * @param optimaBasins map of solutions which are local optima, and the corresponding weight of their basin -- should be passed in empty and will be filled on method returning 
     * @param optimaQuality map of solutions which are local optima, and their corresponding quality -- should be passed in empty and will be filled on method returning
     * @param mapOfAdjacencyListAndWeight map representing an adjacency list, and corersponding weight (key is local optima, and returns map of adjacent basins, and weights) -- should be passed in empty and will be filled on method returning
     * @param edgeType edge type to use in LON construction
     * @param numberOfSamples number of random samplesin legal domain to take to construct estimated LON
     */ 
    public static <K extends Solution> void sampledLON(Problem<K> problem, Neighbourhood<K> neighbourhood, HashMap<K,Weight> optimaBasins, HashMap<K,Double> optimaQuality,HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, EdgeType edgeType, int numberOfSamples) {
        //System.out.print("\n cleaning...");
        clean();
        HashSet<K> optima = new HashSet<>(); // list of optima
        HashMap<K,K> mapFromSolutionToOptima = new HashMap<>(numberOfSamples); // map from each solution queried on a path to its optima
        HashMap<K,Double> fitnesses  = new HashMap<>(numberOfSamples); // map from queries designs to their fitness 
        ArrayList<K> visitedOnPath = new ArrayList<>();
        // sample to get basin sizes
        for (int i=0; i<numberOfSamples; i++) {
            K randomDesign = problem.getRandomSolution();
            //if (i%1000==0)
            //    System.out.println("sample: " + i+ ", fitness map: " + fitnesses.size() + ", optima map" + mapFromSolutionToOptima.size());
            visitedOnPath.clear();
            K optimum = greedyHillclimb(randomDesign,problem,neighbourhood, fitnesses, mapFromSolutionToOptima, visitedOnPath);
            optima.add(optimum);
            for (K design : visitedOnPath) 
                if (!mapFromSolutionToOptima.containsKey(design))
                    mapFromSolutionToOptima.put(design,optimum);

            if (optimaBasins.containsKey(optimum)) {
                optimaBasins.get(optimum).increment();
            } else {
                optimaBasins.put(optimum,new Weight(1));
            }
            if (edgeType.equals(EdgeType.BASIN_TRANSITION)) {
                if (!mapOfAdjacencyListAndWeight.containsKey(optimum)) // check if optimum previously in map
                    mapOfAdjacencyListAndWeight.put(optimum,new HashMap<K,Weight>());
                for (K s : neighbourhood.neighbouringSolutions(randomDesign)) {
                    visitedOnPath.clear();
                    K neighboursOptimum = greedyHillclimb(s,problem, neighbourhood, fitnesses, mapFromSolutionToOptima, visitedOnPath);
                    for (K design : visitedOnPath) 
                        if (!mapFromSolutionToOptima.containsKey(design))
                            mapFromSolutionToOptima.put(design,neighboursOptimum);
                    // note , we don't update basin size here as this would introduce bias 

                    HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(optimum); //get map of adjacent optima
                    // now add neighbouring optima if not present, or update the weight
                    if (linkedOptima.containsKey(neighboursOptimum))
                        linkedOptima.get(neighboursOptimum).increment();
                    else
                        linkedOptima.put(neighboursOptimum,new Weight(1));    
                }
            }
            //if (i%10==0)
            //    System.out.print(optima.size() + " ");
            
        }
        // now approximated edges
        if (edgeType.equals(EdgeType.ESCAPE_EDGE)) {  
            for (K optimum : optima)
                mapOfAdjacencyListAndWeight.put(optimum,new HashMap<K,Weight>()); //initialise maps to adjacent optima, and their edge weights

            for (K o : optima) { //for each optima
                for (K s : neighbourhood.neighbouringSolutionsUpToDistance(o)) { // for each neighbour up to distance
                    K adjacentOptima = null;
                    if (mapFromSolutionToOptima.containsKey(s)) {
                        adjacentOptima = mapFromSolutionToOptima.get(s);
                        HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(o); //get map of adjacent optima
                    } else {
                        visitedOnPath.clear();
                        // note, there is a possiblity that adjacentOptima is not a member of the optima HashSet!
                        adjacentOptima = greedyHillclimb(s,problem, neighbourhood, fitnesses, mapFromSolutionToOptima, visitedOnPath);
                        for (K design : visitedOnPath) 
                            if (!mapFromSolutionToOptima.containsKey(design))
                                mapFromSolutionToOptima.put(design,adjacentOptima);
                        // note , we don't update basin size here as this would introduce bias        
                    }

                    HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(o); //get map of adjacent optima
                    // now add neighbouring optima if not present, or update the weight
                    if (linkedOptima.containsKey(adjacentOptima))
                        linkedOptima.get(adjacentOptima).increment();
                    else
                        linkedOptima.put(adjacentOptima,new Weight(1));

                }
            }

        }

        //System.out.println("\n" + optima.size());
        //System.out.println(fitnesses.size());
        //System.out.println(mapFromSolutionToOptima.size());
    }
    //helper methods
    /*
     * resets the hill climb counter for stats analysis
     */
    private static void clean() {
        greedyHillclimbCalls=0;
    }

    /**
     * Method write out the LON stored in the arguments to files. These are LON_vertex_labels.txt, LON_vertex_quality.txt, LON_basin_sizes.txt,
     * LON_adjacency_list_by_label.txt and LON_adjacency_weights.txt
     * 
     * @param optimaBasins map of solutions which are local optima, and the corresponding weight of their basin  
     * @param optimaQuality map of solutions which are local optima, and their corresponding quality -
     * @param mapOfAdjacencyListAndWeight map representing an adjacency list, and corersponding weight (key is local optima, and returns map of adjacent basins, and weights) 
     */
    public static <K extends Solution> void writeOutLON(HashMap<K,Weight> optimaBasins, HashMap<K,Double> optimaQuality, HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight) {
        int[] vertexList = new int[optimaBasins.size()];
        int[] basinSizes = new int[optimaBasins.size()];
        int[][] adjacencyList  = new int[optimaBasins.size()][];
        int[][] adjacencyWeights  = new int[optimaBasins.size()][];
        String vertexFilename = "LON_vertex_labels.txt";
        String qualityFilename = "LON_vertex_quality.txt";
        String basinFilename = "LON_basin_sizes.txt";
        String adjacencyListVertexLabelsFilename = "LON_adjacency_list_by_label.txt";
        String adjacencyWeightsFilename = "LON_adjacency_weights.txt";
        
        Charset charset = Charset.forName("US-ASCII");
        // CODE BELOW NEEDS REFACTORING
        
        try (BufferedWriter writer = Files.newBufferedWriter(new File(vertexFilename).toPath(), charset)) {
            //String s = "";
            for (K key : optimaBasins.keySet()) {
                String s = key.getIndex() +"\n";
                writer.write(s, 0, s.length());
                //s += key.getIndex() +"\n";
            }
            //writer.write(s, 0, s.length());
        } catch (IOException x) {
            System.err.format("IOException: problem writing to file " + vertexFilename, x);
        }
        
        try (BufferedWriter writer = Files.newBufferedWriter(new File(qualityFilename).toPath(), charset)) {
            //String s = "";
            for (K key : optimaQuality.keySet()) {
                String s = optimaQuality.get(key).doubleValue() +"\n";
                writer.write(s, 0, s.length());
                //s += optimaQuality.get(key).doubleValue() +"\n";
            }
            //writer.write(s, 0, s.length());
        } catch (IOException x) {
            System.err.format("IOException: problem writing to file " + qualityFilename, x);
        }
        
        try (BufferedWriter writer = Files.newBufferedWriter(new File(basinFilename).toPath(), charset)) {
            String s = "";
            for (K key : optimaBasins.keySet()) {
                s += optimaBasins.get(key).getWeight() +"\n";
            }
            writer.write(s, 0, s.length());
        } catch (IOException x) {
            System.err.format("IOException: problem writing to file " + basinFilename, x);
        }
        
        try (BufferedWriter writer = Files.newBufferedWriter(new File(adjacencyListVertexLabelsFilename).toPath(), charset)) {
            String s = "";
            for (K key : mapOfAdjacencyListAndWeight.keySet()) {
                HashMap<K,Weight> list = mapOfAdjacencyListAndWeight.get(key);
                for (K listKey : list.keySet()) {
                    s += listKey.getIndex() +" ";
                }
                s += "\n";
            }
            writer.write(s, 0, s.length());
        } catch (IOException x) {
            System.err.format("IOException: problem writing to file " + adjacencyListVertexLabelsFilename, x);
        }
        
        try (BufferedWriter writer = Files.newBufferedWriter(new File(adjacencyWeightsFilename).toPath(), charset)) {
            String s = "";
            for (K key : mapOfAdjacencyListAndWeight.keySet()) {
                HashMap<K,Weight> list = mapOfAdjacencyListAndWeight.get(key);
                for (K listKey : list.keySet()) {
                    s += optimaBasins.get(listKey).getWeight() +" ";
                }
                s += "\n";
            }
            writer.write(s, 0, s.length());
        } catch (IOException x) {
            System.err.format("IOException: problem writing to file " + adjacencyWeightsFilename, x);
        }
    }
    
    /*
     * Fills optimaBasins hashmap with the optima solutions and the corresponding basin sizes, and 
     * mapOfAdjacencyList with the corresponding list of linked (outward) optima and weights (in the 
     * Edge instances). 
     */
    private static <K extends Solution> void exhaustiveLON(Problem<K> problem, Neighbourhood<K> neighbourhood , HashMap<K,Weight> optimaBasins, HashMap<K,Double> optimaQuality, HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, EdgeType edgeType, boolean efficient) {
        System.out.print("\n cleaning...");
        clean();
        ArrayList<K> optima = new ArrayList<>(); // list of optima
        K[] designs = problem.getExhaustiveSetOfSolutions(); // all designs
        HashMap<K,K> mapFromSolutionToOptima = new HashMap<>(designs.length); // map from each solution to its optima
        System.out.print("getting fitnesses...");
        double[] solutionFitness = getAllQualities(designs,problem); // fitnesses of all solutions, assumed maximisation
        System.out.print(solutionFitness.length + " getting neighbourhood indices...");
        //System.out.println(getNumberOfOptima(neighbourhood,designs,solutionFitness,new ArrayList<>()));
        
        int[][] neighbourhoodIndices = getNeighboursAndSetOptima(neighbourhood,designs,solutionFitness,optima);
        System.out.print("getting basins...");
        fillBasins(designs,optima,optimaBasins,mapFromSolutionToOptima,solutionFitness,neighbourhoodIndices,efficient);
        System.out.print("setting edges...");
        setEdges(neighbourhood,designs,optima,mapOfAdjacencyListAndWeight,mapFromSolutionToOptima,solutionFitness,neighbourhoodIndices,edgeType,efficient);
        for (K o : optima)
             optimaQuality.put(o,solutionFitness[o.getIndex()]);
             
        System.out.println("\n number of optima: " + optima.size());
        System.out.println("map from solution to optima size: " + mapFromSolutionToOptima.size());     
    }

    private static <K extends Solution> K greedyHillclimb(K designToSearchFrom, Problem<K> problem, Neighbourhood<K> neighbourhood) {
        greedyHillclimbCalls++;
        K nextOnPath = designToSearchFrom;
        double currentFitness = problem.getQuality(designToSearchFrom);
        K[] neighbours = neighbourhood.neighbouringSolutions(designToSearchFrom);
        for (K neighbour : neighbours) {
            double fitness = problem.getQuality(neighbour);
            if (fitness > currentFitness) {
                currentFitness = fitness;
                nextOnPath = neighbour;
            }
        }
        if (nextOnPath==designToSearchFrom)
            return designToSearchFrom;
        return greedyHillclimb(nextOnPath, problem, neighbourhood);    
    }

    private static <K extends Solution> K greedyHillclimb(K designToSearchFrom, Problem<K> problem, Neighbourhood<K> neighbourhood, HashMap<K,Double> fitnesses, HashMap<K,K> mapFromSolutionToOptima, ArrayList<K> visitedOnPath) {
        if (mapFromSolutionToOptima.containsKey(designToSearchFrom)){
            return mapFromSolutionToOptima.get(designToSearchFrom);
        } else {
            greedyHillclimbCalls++;
            visitedOnPath.add(designToSearchFrom);
            K nextOnPath = designToSearchFrom;
            Double currentFitness = fitnesses.get(designToSearchFrom);
            if (currentFitness==null) {
                currentFitness = problem.getQuality(designToSearchFrom);
                fitnesses.put(designToSearchFrom,currentFitness);
            }
            K[] neighbours = neighbourhood.neighbouringSolutions(designToSearchFrom);
            for (K neighbour : neighbours) {
                Double fitness = fitnesses.get(neighbour);
                if (fitness==null) {
                    fitness = problem.getQuality(neighbour);
                    fitnesses.put(neighbour,fitness);
                } 
                if (fitness.doubleValue() > currentFitness.doubleValue()) {
                    currentFitness = fitness;
                    nextOnPath = neighbour;
                }
            }
            if (nextOnPath==designToSearchFrom)
                return designToSearchFrom;
            return greedyHillclimb(nextOnPath, problem, neighbourhood, fitnesses, mapFromSolutionToOptima,visitedOnPath);    
        }
    }

    private static <K extends Solution> void setEdges(Neighbourhood<K> neighbourhood, K[] designs, ArrayList<K> optima, HashMap<K,HashMap<K,Weight>> mapOfAdjacencyListAndWeight, HashMap<K,K> mapFromSolutionToOptima, double[] solutionFitness, int[][] neighbourhoodIndices, EdgeType edgeType, boolean efficient) {
        for (K optimum : optima)
            mapOfAdjacencyListAndWeight.put(optimum,new HashMap<K,Weight>()); //initialise maps to adjacent optima, and their edge weights
        if (efficient==false) {
           if (edgeType.equals(EdgeType.BASIN_TRANSITION)) {   
                for (int i=0; i< designs.length; i++) { // for each design
                    K currentOptima = mapFromSolutionToOptima.get(designs[i]);
                    for (K s : neighbourhood.neighbouringSolutions(designs[i])) { //get design neighbours
                        K adjacentOptima = naiveGreedyHillClimb(designs,s.getIndex(),solutionFitness,neighbourhoodIndices); // find optima of ith design
                        HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(currentOptima); //get map of adjacent optima
                        // now add neighbouring optima if not present, or update the weight
                        if (linkedOptima.containsKey(adjacentOptima))
                            linkedOptima.get(adjacentOptima).increment();
                        else
                            linkedOptima.put(adjacentOptima,new Weight(1));
                    }
                }
            } else { // escape edge
                for (K o : optima) { //for each optima
                    for (Solution s : neighbourhood.neighbouringSolutionsUpToDistance(designs[o.getIndex()])) { // for each neighbour up to distance
                        K adjacentOptima = naiveGreedyHillClimb(designs,s.getIndex(),solutionFitness,neighbourhoodIndices); // find optima of ith design
                        HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(o); //get map of adjacent optima
                        // now add neighbouring optima if not present, or update the weight
                        if (linkedOptima.containsKey(adjacentOptima))
                            linkedOptima.get(adjacentOptima).increment();
                        else
                            linkedOptima.put(adjacentOptima,new Weight(1));

                    }
                }
            }
            
        } else {
            if (edgeType.equals(EdgeType.BASIN_TRANSITION)) {   
                for (int i=0; i< designs.length; i++) { // for each design
                    K currentOptima = mapFromSolutionToOptima.get(designs[i]);
                    for (K s : neighbourhood.neighbouringSolutions(designs[i])) { //get design neighbours
                        K adjacentOptima = mapFromSolutionToOptima.get(s);
                        HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(currentOptima); //get map of adjacent optima
                        // now add neighbouring optima if not present, or update the weight
                        if (linkedOptima.containsKey(adjacentOptima))
                            linkedOptima.get(adjacentOptima).increment();
                        else
                            linkedOptima.put(adjacentOptima,new Weight(1));
                    }
                }
            } else { // escape edge
                for (K o : optima) { //for each optima
                    for (Solution s : neighbourhood.neighbouringSolutionsUpToDistance(designs[o.getIndex()])) { // for each neighbour up to distance
                        K adjacentOptima = mapFromSolutionToOptima.get(s);
                        HashMap<K,Weight> linkedOptima = mapOfAdjacencyListAndWeight.get(o); //get map of adjacent optima
                        // now add neighbouring optima if not present, or update the weight
                        if (linkedOptima.containsKey(adjacentOptima))
                            linkedOptima.get(adjacentOptima).increment();
                        else
                            linkedOptima.put(adjacentOptima,new Weight(1));

                    }
                }
            }
        }
    }

    private static <K extends Solution> void fillBasins(K[] designs, ArrayList<K> optima, HashMap<K,Weight> optimaBasins,HashMap<K,K> mapFromSolutionToOptima, double[] solutionFitness, int[][] neighbourhoodIndices, boolean efficient) {
        // connect each optima to an empty initial basin
        for (int i=0; i<optima.size(); i++) {
            optimaBasins.put(optima.get(i), new Weight());
        }
        if (efficient == false) {
            // now compute basin sizes, by hillclimbing from every design
            for (int i=0; i< designs.length; i++) {
                K basinOptima = naiveGreedyHillClimb(designs,i,solutionFitness,neighbourhoodIndices); // find optima of ith design
                mapFromSolutionToOptima.put(designs[i],basinOptima); // store optima for ith design
                optimaBasins.get(basinOptima).increment(); // increment basin size of optima
            }
        } else { // now compute basin sizes, by hillclimbing from every design, an utilising previous visits
            ArrayList<K> visitedOnPath = new ArrayList<>();
            for (int i=0; i< designs.length; i++) {
                visitedOnPath.clear();
                K basinOptima = efficientGreedyHillClimb(mapFromSolutionToOptima,visitedOnPath,designs,i,solutionFitness,neighbourhoodIndices); // find optima of ith design
                for (K s : visitedOnPath) {
                    mapFromSolutionToOptima.put(s,basinOptima);
                    optimaBasins.get(basinOptima).increment();
                }
            }
        }
    }

    private static <K extends Solution> K efficientGreedyHillClimb(HashMap<K,K> mapFromSolutionToOptima, ArrayList<K> visitedOnPath,K[] designs, int index, double[] solutionFitness, int[][] neighbourhoodIndices) {
        if (mapFromSolutionToOptima.containsKey(designs[index]))
            return mapFromSolutionToOptima.get(designs[index]);
        greedyHillclimbCalls++;
        int bestIndex = -1;
        double bestValue = solutionFitness[index];
        for (int neighbourIndex : neighbourhoodIndices[index]) {
            if (solutionFitness[neighbourIndex]>bestValue) {
                bestIndex = neighbourIndex;
                bestValue = solutionFitness[neighbourIndex];
            }
        }
        visitedOnPath.add(designs[index]);
        if (bestIndex!=-1){
            return efficientGreedyHillClimb(mapFromSolutionToOptima,visitedOnPath,designs,bestIndex,solutionFitness,neighbourhoodIndices);
        }
        return designs[index];
    }

    private static <K extends Solution> K naiveGreedyHillClimb(K[] designs, int index, double[] solutionFitness, int[][] neighbourhoodIndices) {
        greedyHillclimbCalls++;
        int bestIndex = -1;
        double bestValue = solutionFitness[index];
        for (int neighbourIndex : neighbourhoodIndices[index]) {
            if (solutionFitness[neighbourIndex]>bestValue) {
                bestIndex = neighbourIndex;
                bestValue = solutionFitness[neighbourIndex];
            }
        }
        if (bestIndex!=-1){
            return naiveGreedyHillClimb(designs,bestIndex,solutionFitness,neighbourhoodIndices);
        }
        return designs[index];
    }

    private static <K extends Solution> int getNumberOfOptima(Neighbourhood<K> neighbourhood,K[] designs,double[] solutionFitness,ArrayList<K> optima){ 
        int n=0;
        for (int i=0; i< designs.length; i++) {
            K[] neighbours = neighbourhood.neighbouringSolutions(designs[i]);
            boolean isOptima = true;
            for (K neighbour : neighbours) {
                if (solutionFitness[i]<solutionFitness[neighbour.getIndex()]){
                    isOptima = false;
                    break;
                }
            }
            if (isOptima) {
                System.out.println(n++);
            }
        }
        System.out.println("completed");
        return n;
    }
    
    private static <K extends Solution> int[][] getNeighboursAndSetOptima(Neighbourhood<K> neighbourhood,K[] designs,double[] solutionFitness,ArrayList<K> optima){ 
        int[][] neighbourhoodIndices = new int[designs.length][];
        for (int i=0; i< designs.length; i++) {
            K[] neighbours = neighbourhood.neighbouringSolutions(designs[i]);
            neighbourhoodIndices[i] = new int[neighbours.length];
            for (int k=0; k<neighbours.length; k++)
                neighbourhoodIndices[i][k] = neighbours[k].getIndex();
            boolean isOptima = true;
            for (int index : neighbourhoodIndices[i]) {
                if (solutionFitness[i]<solutionFitness[index]){
                    isOptima = false;
                    break;
                }
            }
            if (isOptima) {
                optima.add(designs[i]);
            }
        }
        return neighbourhoodIndices;
    }

    private static <K extends Solution> double[] getAllQualities(K[] designs, Problem<K> problem) {
        double[] solutionFitnesses = new double[designs.length]; // fitnesses of all solutions, assumed maximisation
        for (int i=0; i< designs.length; i++) 
            solutionFitnesses[i] = problem.getQuality(designs[i]);
        return solutionFitnesses;    
    }

}
