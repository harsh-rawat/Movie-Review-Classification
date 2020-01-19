import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
	
	Map<Label,Integer> documentCount = null;
	Map<Label, Integer> wordCount = null;
	Map<String, Map<Label, Integer>> wordGivenLabel = null;
	int totalDocuments = 0;
	int totalWords = 0;
	int vocab = 0;
	
    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
    	this.totalDocuments = 0;
		this.totalWords = 0;
		this.vocab = 0;
		this.wordGivenLabel = new HashMap<>();
		
    	this.getDocumentsCountPerLabel(trainData);
    	this.getWordsCountPerLabel(trainData);
    	this.getWordGivenLabelData(trainData);
    	this.vocab = v;
    }
    
    private void getWordGivenLabelData(List<Instance> instances) {
    	
    	for(Instance ins : instances) {
    		Label label = ins.label;
    		
    		for(String word : ins.words) {
    			this.wordGivenLabel.putIfAbsent(word, new HashMap<>());
    			Map<Label, Integer> map = this.wordGivenLabel.get(word);
    			map.put(label, map.getOrDefault(label, 0)+1);
    		}
    	}
    	
    }

    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        
    	Map<Label, Integer> map = new HashMap<>();
    	map.putIfAbsent(Label.POSITIVE, 0);
    	map.putIfAbsent(Label.NEGATIVE, 0);
    	
    	for(Instance instance : trainData) {
    		Label label = instance.label;
    		map.put(label,map.getOrDefault(label, 0)+instance.words.size());
    		this.totalWords += instance.words.size();
    	}
    	
    	this.wordCount = map;
    	return map;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        
    	Map<Label, Integer> map = new HashMap<>();
    	map.putIfAbsent(Label.POSITIVE, 0);
    	map.putIfAbsent(Label.NEGATIVE, 0);
    	
    	for(Instance instance : trainData) {
    		Label label = instance.label;
    		map.put(label, map.getOrDefault(label, 0)+1);
    		this.totalDocuments += 1;
    	}
    	
    	this.documentCount = map;
    	return map;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
    	int labelDocs = this.documentCount.getOrDefault(label,0);
    	return labelDocs/(totalDocuments*1.0);
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
        // Calculate the probability with Laplace smoothing for word in class(label)
    	Map<Label, Integer> map = this.wordGivenLabel.get(word);
    	int wordCountForLabel = 0;
    	if(map != null && map.containsKey(label)) wordCountForLabel = map.get(label);
    	
    	return (1.0*(wordCountForLabel + 1))/(1.0*(this.vocab + this.wordCount.getOrDefault(label,0)));
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability

    	ClassifyResult classification = new ClassifyResult();
    	
    	//Calculate Probability of Label Positive
    	double p_y_positive = Math.log(this.p_l(Label.POSITIVE));
    	double p_y_negative = Math.log(this.p_l(Label.NEGATIVE));
    	double p_x_positive = 0.0;
    	double p_x_negative = 0.0;
    	
    	for(String word : words) {
    		p_x_positive += Math.log(p_w_given_l(word, Label.POSITIVE));
    		p_x_negative += Math.log(p_w_given_l(word, Label.NEGATIVE));
    	}
    	
    	double p_positive = p_y_positive + p_x_positive;
    	double p_negative = p_y_negative + p_x_negative;
    	
    	Map<Label, Double> map = new HashMap<>();
    	map.put(Label.POSITIVE,p_positive);
    	map.put(Label.NEGATIVE, p_negative);
    	classification.logProbPerLabel = map;
    	
    	if(p_positive < p_negative)
    		classification.label = Label.NEGATIVE;
    	else
    		classification.label = Label.POSITIVE;
    	
    	return classification;
    }


}
