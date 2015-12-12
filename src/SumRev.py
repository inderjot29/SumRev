'''
Created on Nov 15, 2015

@author: inderjot
'''


from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import os
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC

# global variables
trainphraseOpinions={}
testphraseOpinions={}
selectedFeatures={}	


# getting hyponyms of the selected featurewords
def GetHyponyms(index,feature,featureSets):
	features=[]
	features.append(feature)
	ss=wn.synsets(feature)[index]
	hyponyms=list(chain(*[i.lemma_names() for i in ss.hyponyms()]))
	for hyponym in hyponyms:
		features.append(hyponym)
	featureSets[feature]=features
	featureSets.update()
	
#expanding featureset using Wordnet
def GetFeatureSet():
#following 7 words are taken into account for a feature set.
	featureSet=['staff','service','ambience','food','meal','dish','menu','cost']
	featureSets={}
	for feature in featureSet:
		#extend featureset using direct hyponyms
		#choose appropriate synset based on your knowlegde (kind of hand annotation here)
		if feature=='food':
			GetHyponyms(1,feature,featureSets)
		elif feature == 'service':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'staff':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'meal':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'dish':
			GetHyponyms(1,feature,featureSets)
		elif feature == 'menu':
			GetHyponyms(0,feature,featureSets)
		elif feature == 'cost':
			GetHyponyms(0,feature,featureSets)
			
	return featureSets		
	


	
stoplist = set(stopwords.words('english'))


def RemovePunctAndStopWords(tokens):
	nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
	filtered = [w for w in tokens if nonPunct.match(w)]
	#Remove the stopwords from filtered text
	filtered_words = [word for word in filtered if word.lower() not in stoplist]
	frequent_words=['the','and','of','this']
	filtered_words = [word for word in filtered_words if word.lower() not in frequent_words]
	return filtered_words

#returns phrases/aspects which meet the set patterns	
def GetPhrases(taggedTokens):
	phrases=[]
	i=0
	for i in range(0,len(taggedTokens)):
		#phrases starting with adjective- check for (adjective noun) or (adjective noun noun)
		if taggedTokens[i][1]=='JJ':
			if i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='NN':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='NN':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
		
		#phrases starting with verb- check for (verb noun) or (verb adverb)
		elif taggedTokens[i][1]=='VBP':
			if  i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='NN':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
					phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='RB':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
					phrases.append(phrase)
		#phrases starting with adverb- check for (adverb verb) or (adverb adverb adjective) or(adverb adjective noun) or (adverb adjective)
		elif taggedTokens[i][1]=='RB':
			if i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='VBP':
				phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
				if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='RB':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='JJ':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
			elif i< len(taggedTokens)-1 and taggedTokens[i+1][1]=='JJ':
				if i< len(taggedTokens)-2 and taggedTokens[i+2][1]=='NN':
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]+' '+taggedTokens[i+2][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
				else:
					phrase=taggedTokens[i][0]+' '+taggedTokens[i+1][0]
					if phrase is not None and len(phrase)>0:
						phrases.append(phrase)
	return phrases
	
#returns dictionary of phrases along with the opinions based on lexical corpus used
def GetOpinions(phrases,phraseOpinions):
	positiveWords=[]
	negativeWords=[]
	#phraseOpinions={}
	for line in open('positive-words.txt'):
		positiveWords.append(line)
		
	for line in open('negative-words.txt'):
		negativeWords.append(line)
	
	positiveWords=map(lambda s: s.strip(), positiveWords)
	negativeWords=map(lambda s: s.strip(), negativeWords)
	
	for phrase in phrases:
		polarity=0
		words=phrase.split(' ')
		for word in words:
			if 'not_' in word:
				word=word.split('not_')[1]
			if word.encode('utf-8') in positiveWords:
				polarity=polarity+1
			elif word.encode('utf-8') in negativeWords:
				polarity=polarity-1
		if polarity>0:
			if 'not_' in phrase:
				phraseOpinions[phrase]='bad'
			else:
				phraseOpinions[phrase]='good'
		elif polarity<0:
			if 'not_' in phrase:
				phraseOpinions[phrase]='good'
			else:
				phraseOpinions[phrase]='bad'
		else:
			phraseOpinions[phrase]='neutral'
		
	return phraseOpinions
		
# checks if the phrases obtained contain any of the feature words from featureSet				
def CheckForFeatureWords(allPhrases,featureSet):
	flag=0
	for phrase in allPhrases:
		for feature in featureSet:
			for word in featureSet[feature]:
				if word in phrase:
					selectedFeatures[phrase]=feature
					flag=1
		if flag==0:
			allPhrases.remove(phrase)
		else:
			flag=0
	return allPhrases
	
#categorizing good,bad and neutral phrases to use data for classifiers	
def good_phrases(phrases):
	return dict([(phrase, phrases[phrase]) for phrase in phrases if phrases[phrase]=='good'])

def bad_phrases(phrases):
	return dict([(phrase, phrases[phrase]) for phrase in phrases if phrases[phrase]=='bad'])
	
def neutral_phrases(phrases):
	return dict([(phrase, 'neutral') for phrase in phrases if phrases[phrase]=='neutral'])
			
# evaluating results using classifiers
def EvaluateUsingClassifiers():
	#Train the classifier
	
	training_set=[]
	training_set.append((good_phrases(trainphraseOpinions),'good'))
	training_set.append((bad_phrases(trainphraseOpinions),'bad'))
	training_set.append((neutral_phrases(trainphraseOpinions),'neutral'))
	''' in order to cross-validate,i used three different training sets interchangeably in these three classifiers'''
	testing_set=[]
	testing_set.append((good_phrases(testphraseOpinions),'good'))
	testing_set.append((bad_phrases(testphraseOpinions),'bad'))
	testing_set.append((neutral_phrases(testphraseOpinions),'neutral'))
	#print testing_set
	print("training classifier Original Naive Bayes")
	classifier=nltk.NaiveBayesClassifier.train(training_set)
	print("Training classifier Original Naive Bayes completed!")
	print("Original Naive Bayes classifier accuracy percent:",nltk.classify.accuracy(classifier,testing_set)*100)
	
	print(classifier.show_most_informative_features())
	#print(classifier.classify(testing_set))
	
	print("training classifier LinearSVC")
	LinearSVC_Classifier=SklearnClassifier(LinearSVC())
	LinearSVC_Classifier.train(training_set)
	print("Training classifier SVC completed!")
	print("Linear SVC classifier accuracy percent:",nltk.classify.accuracy(LinearSVC_Classifier,testing_set)*100)
		
# checks if the sentence parsed contains negative values which can change the sentiment of sentence		
def CheckForNOT(tokens):
	for token in tokens:
		if token in ('not',"n't","w'nt","'nt"):
			#print "inside check not if"
			return True
	return False

#writing the results in a CSV file
def WritingResults(OpinionsSet):
	fout=open('Results.csv','w+')
	fout.write('Feature\t Category \t Polarity \n')
	for feature in selectedFeatures:
		fout.write(feature.encode('utf-8'))
		fout.write('\t')
		fout.write(selectedFeatures[feature])
		fout.write('\t')
		
		fout.write(OpinionsSet[feature])
		fout.write('\n')
		
	fout.close()
	
# Reading reviews from files
def ReadReviews(path,featureSet):
	phraseOpinions={}
	allPhrases=[]
	for filename in os.listdir(path):
		print "Reading-",filename
		for line in open(os.path.join(path,filename)):
			line=unicode(line,'utf-8')
			sentences=line.split('.')
			for sentence in sentences:	
				tokens=word_tokenize(sentence)
				isNot=CheckForNOT(tokens)
				#print isNot
				tokens=RemovePunctAndStopWords(tokens)
				taggedTokens=nltk.pos_tag(tokens)
				phrases=GetPhrases(taggedTokens)
				if phrases is not None and len(phrases)>0:
					for phrase in phrases:
						if isNot is True:
							allPhrases.append('not_'+phrase)
						else:
							allPhrases.append(phrase)
	#checking if these phrases contain feature-words
	updatedPhrases=CheckForFeatureWords(allPhrases,featureSet)
	# finding polarity of phrases obtained
	phraseOpinions=GetOpinions(updatedPhrases,phraseOpinions)
	return phraseOpinions

if __name__ == '__main__':
	#returns list of featureWords
	featureSet=GetFeatureSet()
	
	#reading reviews
	path='./TrainData/'
	trainphraseOpinions=ReadReviews(path,featureSet)

	path='./TestData/'
	testphraseOpinions=ReadReviews(path,featureSet)
	EvaluateUsingClassifiers()
	#combining training and testing set to furthur evaluate results
	OpinionsSet=dict(trainphraseOpinions.items()+ testphraseOpinions.items())
	
	WritingResults(OpinionsSet)
	
	
	
	
	
	
	
	
	
	
		
		
		
	