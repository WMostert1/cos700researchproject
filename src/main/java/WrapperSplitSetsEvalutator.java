import lons.examples.ConcreteBinarySolution;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

public class WrapperSplitSetsEvalutator extends ASEvaluation implements SubsetEvaluator, OptionHandler, TechnicalInformationHandler {

    private int m_numAttribs;
    private Map<ConcreteBinarySolution, Double> fitnessMap;

    public WrapperSplitSetsEvalutator(Map<ConcreteBinarySolution, Double> fitnessMap){
        this.fitnessMap = fitnessMap;
    }

    @Override
    public void buildEvaluator(Instances instances) throws Exception {
         this.m_numAttribs = instances.numAttributes() -1;
    }

    @Override
    public double evaluateSubset(BitSet subset) throws Exception {

        int numAttributes = 0;

        Remove delTransform = new Remove();
        delTransform.setInvertSelection(true);

        int i;
        for(i = 0; i < this.m_numAttribs; ++i) {
            if (subset.get(i)) {
                ++numAttributes;
            }
        }

        int[] featArray = new int[numAttributes];


        int j = 0;
        for(i = 0; i < this.m_numAttribs; ++i) {
            if (subset.get(i)) {
                featArray[j++] = i;
            }
        }


        ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat(featArray, this.m_numAttribs));

        Double fitness = this.fitnessMap.get(subSolution);
        if(fitness == null){
            throw new RuntimeException("Unknown solution!");
        }
        return fitness;
    }

    @Override
    public Enumeration listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] strings) throws Exception {
        //No options to set
    }

    @Override
    public String[] getOptions() {
        return new String[]{};
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        return null;
    }
}
