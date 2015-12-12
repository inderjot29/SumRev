'''
Created on Nov 15, 2015

@author: inderjot
'''


from nltk.corpus import stopwords
import re
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import os
import operator
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC

trainphraseOpinions={}
testphraseOpinions={}
# getting hyponyms of the selected featurewords
def GetHyponyms(index,feature,featureSets):
	featureSet=[]
	ss=wn.synsets(feature)[index]
	hyponyms=list(chain(*[i.lemma_names() for i in ss.hyponyms()]))
	for hyponym in hyponyms:
		featureSet.append(hyponym)
	featureSets[feature]=featureSet
	featureSets.update()
	

def GetFeatureSet():
    #following 7 words are taken into account for a feature set.
	featureSet=['staff','service','ambience','food','meal','dish','menu','cost']
	featureSets={}
	for feature in featureSet:
		#extend featureset using direct hyponyms
		#choose appropriate synset based on your knowlegde (kind of hand annotation here)
		if feature=='food':
			GetHyponyms(1,feature,featureSets)
			#print "food",featureSet
		elif feature == 'service':
			GetHyponyms(0,feature,featureSets)
			#print "service",featureSet
		elif feature == 'staff':
			GetHyponyms(0,feature,featureSets)
			#print "staff",featureSet
	
		elif feature == 'meal':
			GetHyponyms(0,feature,featureSets)
			#print "meal",featureSet
		elif feature == 'dish':
			GetHyponyms(1,feature,featureSets)
			#print "dish",featureSet
		elif feature == 'menu':
			GetHyponyms(0,feature,featureSets)
			#print "menu",featureSet
		elif feature == 'cost':
			GetHyponyms(0,feature,featureSets)
			#print "cost",featureSet
			
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
			if word.encode('utf-8') in positiveWords:
				polarity=polarity+1
			elif word.encode('utf-8') in negativeWords:
				polarity=polarity-1
		if polarity>0 :
			phraseOpinions[phrase]='good'
		elif polarity<0:
			phraseOpinions[phrase]='bad'
		else:
			phraseOpinions[phrase]='neutral'
		
	return phraseOpinions
		
selectedFeatures={}					
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
	
def good_phrases(phrases):
	
		
			
    return dict([(phrase, phrases[phrase]) for phrase in phrases if phrases[phrase]=='good'])

def bad_phrases(phrases):
    return dict([(phrase, phrases[phrase]) for phrase in phrases if phrases[phrase]=='bad'])
	
def neutral_phrases(phrases):
    return dict([(phrase, 'neutral') for phrase in phrases if phrases[phrase]=='neutral'])
			
def TrainAndTestclassifiers():
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
			
	

if __name__ == '__main__':
	#returns list of featureWords
	featureSet=GetFeatureSet()
	reviews=[]
	allPhrases=[]
	#reading reviews
	
	#f=open('out.txt','w+')
	path='./TrainData/'
	
	for filename in os.listdir(path):
		print "Reading-",filename
		for line in open(os.path.join(path,filename)):
	#for line in open('./TrainData/AuPiedDeCochen.txt'):
			line=unicode(line,'utf-8')
			tokens=word_tokenize(line)
			tokens=RemovePunctAndStopWords(tokens)
			taggedTokens=nltk.pos_tag(tokens)
			phrases=GetPhrases(taggedTokens)
			if phrases is not None and len(phrases)>0:
				for phrase in phrases:
					allPhrases.append(phrase)
		#checking if these phrases contain feature-words
		trainupdatedPhrases=CheckForFeatureWords(allPhrases,featureSet)
		# finding polarity of phrases obtained
		trainphraseOpinions=GetOpinions(trainupdatedPhrases,trainphraseOpinions)
		#trainphraseOpinions.update(trainphraseOpinions)
		
		#for opinion in trainphraseOpinions:
			#f.write(opinion.encode('utf-8'))
	#f.close()
	#test data
	path='./TestData/'
	for filename in os.listdir(path):
		print "Reading-",filename
		for line in open(os.path.join(path,filename)):
	#for line in open('./TestData/SaintSushiBar.txt'):
			line=unicode(line,'utf-8')
			tokens=word_tokenize(line)
		
			tokens=RemovePunctAndStopWords(tokens)
		
			taggedTokens=nltk.pos_tag(tokens)
		
			phrases=GetPhrases(taggedTokens)
		#print phrases
			if phrases is not None and len(phrases)>0:
				for phrase in phrases:
					allPhrases.append(phrase)
		#checking if these phrases contain feature-words
		testupdatedPhrases=CheckForFeatureWords(allPhrases,featureSet)
		# finding polarity of phrases obtained
		testphraseOpinions=GetOpinions(testupdatedPhrases,testphraseOpinions)
		#print testphraseOpinions
		#testphraseOpinions.update(testphraseOpinions)
	
	
		
	
	TrainAndTestclassifiers()
	
	OpinionsSet=dict(trainphraseOpinions.items()+ testphraseOpinions.items())
	#print OpinionsSet
	#OpinionsSet=OpinionsSet.update(testphraseOpinions)
	#print OpinionsSet
	fout=open('features.csv','w+')
	fout.write('Feature\t category \t polarity \n')
	for feature in selectedFeatures:
		fout.write(feature.encode('utf-8'))
		fout.write('\t')
		fout.write(selectedFeatures[feature])
		fout.write('\t')
		
		fout.write(OpinionsSet[feature])
		fout.write('\n')
		
	fout.close()
	
	
	
	
		
		
		
	