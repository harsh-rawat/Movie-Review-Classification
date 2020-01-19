# Movie-Review-Classification
This is a Machine Learning project in which we classify the movie review as Positive or Negative using a Naive Bayes Classifier.

In this project, we implemented a Naïve Bayes classifier for categorizing movie reviews as either POSITIVE or NEGATIVE. 

The dataset for training consists of online movie reviews derived from an IMDb dataset: https://ai.stanford.edu/~amaas/data/sentiment/ that have been labeled based on the review scores. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10.

While using the Naive Bayes algorithm in this project, we perform Laplace Smoothing while calculating the conditional probabilities.

# Input - Output format

	java SentimentAnalysis <mode> <trainFilename> [<testFilename> | <K>]
 
where -
  
	trainingFilename and testFilename are the names of the training set and test set files, respectively. 
  
	mode is an integer from 0 to 3, controlling what the program will output. 
  
	When mode is 0 or 1, there are only two arguments, mode and trainFilename; 
  
	When the mode is 2 the third argument is testFilename; 
  
	When mode is 3, the third argument is K, the number of folds used for cross validation. 
  
The output for these four modes should be:

	0. Prints the number of documents for each label in the training set
	
	1. Prints the number of words for each label in the training set
	
	2. For each instance in test set, prints a line displaying the predicted class and the log probabilities for both classes
	
	3. Prints the accuracy score for K-fold cross validation
