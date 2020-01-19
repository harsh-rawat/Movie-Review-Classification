import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
        
    	if(trainData.size() == 0 || (k < 2 || k > trainData.size())) return 0.0;
    	
    	List<Double> accuracy = new ArrayList<>();
    	
    	List<Instance> trainFold = new ArrayList<>();
    	List<Instance> testFold = new ArrayList<>();
    	
    	int size = trainData.size() / k;
    	
    	for(int i=0; i < k; i++) {
    		trainFold.clear();
    		testFold.clear();
    		
    		trainFold = new ArrayList<>(trainData);
    		
    		for(int j = size*i; j < size*(i+1); j++) {
    			testFold.add(trainData.get(j));
    			trainFold.remove(size*i);
    		}
    		
    		clf.train(trainFold, v);
    		
    		int correct = 0, total = testFold.size();
    		for(Instance ins : testFold) {
    			ClassifyResult res = clf.classify(ins.words);
    			if(res.label == ins.label) correct++;
    		}
    		
    		accuracy.add(correct/(1.0*total));
    	}
    	
    	double avg = 0;
    	for(double accu : accuracy) avg += accu;
    	
    	return avg/k;
    }
}
