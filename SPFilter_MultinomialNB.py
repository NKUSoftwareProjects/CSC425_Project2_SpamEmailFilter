import os
import math as Math
import numpy as np


most_common_word = 3000
# avoid 0 terms in features
smooth_alpha = 1.0
class_num = 2 # we have only two classes: ham and spam
class_log_prior = [0.0, 0.0] # probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word)) # feature parameterized probability
SPAM = 1 # spam class label
HAM = 0 # ham class label


class MultinomialNB_class:
    
	#multinomial naive bayes
	def MultinomialNB(self, features, labels):
		'''
		calculate class_log_prior
		
		loop over labels
			if the value of the term in labels = 1 then ham++ 
			if the value of the term in labels = 0 then spam++
		class_log_prior[0] = Math.log(ham)
		class_log_prior[1] = Math.log(spam)
		'''
		ham_file_count = 0
		spam_file_count = 0
		for label in labels:
			if label == HAM:
				ham_file_count += 1
			if label == SPAM:
				spam_file_count += 1
		
		class_log_prior[0] = Math.log(ham_file_count / labels.size)
		class_log_prior[1] = Math.log(spam_file_count / labels.size)

		'''
		calculate feature_log_prob
		
		nested loop over features
		for row = features.length
			for col = most_common
				ham[col] += features[row][col]
				spam[col] += features[row][col]
				sum of ham
				sum of spam

		for i = most_common
			ham[i] + smooth_alpha
			spam[i] + smooth_alpha

		sum of ham += most_common*smooth_alpha
		sum of spam += most_common*smooth_alpha

		for j = most_common
			feature_log_prob[0] = Math.log(ham[i]/sum of ham)
			feature_log_prob[1] = Math.log(spam[i]/sum of spam)
		'''
		ham = np.zeros(most_common_word)
		spam = np.zeros(most_common_word)
		sum_of_ham = 0
		sum_of_spam =0
		
		row_index = 0
		print(len(features[0]))
		for row in features:
			for col in range(most_common_word):
				ham[col] += features[row_index][col]
				spam[col] += features[row_index][col]
				sum_of_ham += features[row_index][col]
				sum_of_spam += features[row_index][col]
		row_index += 1

		for i in range(most_common_word):
			ham[i] += smooth_alpha
			spam[i] += smooth_alpha

		sum_of_ham += most_common_word * smooth_alpha
		sum_of_spam += most_common_word * smooth_alpha

		for j in range(most_common_word):
			feature_log_prob[0] = Math.log(ham[i] / sum_of_ham)
			feature_log_prob[1] = Math.log(spam[i] / sum_of_spam)

	#multinomial naive bayes prediction
	def MultinomialNB_predict(self, features):
		classes = np.zeros(len(features))

		ham_prob_total = 0.0
		spam_prob_total = 0.0

		'''
		 nested loop over features with i and j
		 calculate ham_prob and spam_prob
		 add ham_prob and spam_prob with class_log_prior
		 if ham_prob > spam_prob
		 	HAM
		 else SPAM
		 return  classes
		 '''
		i_index = 0
		j_index = 0
		for i in features:
			for j in features[i_index]:
				ham_prob_total = feature_log_prob[HAM][i_index]
				spam_prob_total = feature_log_prob[SPAM][i_index]
				j_index += 1
			if ham_prob_total > spam_prob_total:
				classes[i_index] = HAM
			else:
				classes[i_index] = SPAM
		i_index += 1

		return classes
