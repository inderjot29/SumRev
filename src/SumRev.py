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
	featureSet=['staff','service','place','food','meal','dish','menu']
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
		elif feature == 'place':
			GetHyponyms(0,feature,featureSets)
			#print "place",featureSet
		elif feature == 'meal':
			GetHyponyms(0,feature,featureSets)
			#print "meal",featureSet
		elif feature == 'dish':
			GetHyponyms(1,feature,featureSets)
			#print "dish",featureSet
		elif feature == 'menu':
			GetHyponyms(0,feature,featureSets)
			#print "menu",featureSet
			
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
	

def GetOpinions(phrases):
	positiveWords=[]
	negativeWords=[]
	phraseOpinions={}
	for line in open('positive-words.txt'):
		positiveWords.append(line)
		
	for line in open('negative-words.txt'):
		negativeWords.append(line)
	
	positiveWords=map(lambda s: s.strip(), positiveWords)
	negativeWords=map(lambda s: s.strip(), negativeWords)
	polarity=0
	for phrase in phrases:
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
		
					
def CheckForFeatureWords(allPhrases,featureSet):
	flag=0
	
	for phrase in allPhrases:
		for feature in featureSet:
			for word in featureSet[feature]:
			
				if word in phrase:
				
					flag=1
		if flag==0:
			allPhrases.remove(phrase)
		else:
			flag=0
	return allPhrases
			
				
			
	

if __name__ == '__main__':
	#returns list of featureWords
	featureSet=GetFeatureSet()
	reviews=[]
	allPhrases=[]
	#reading reviews 
	for line in open('./Data/NamasteMontreal.txt'):
		line=unicode(line,'utf-8')
		tokens=word_tokenize(line)
		tokens=RemovePunctAndStopWords(tokens)
		taggedTokens=nltk.pos_tag(tokens)
		phrases=GetPhrases(taggedTokens)
		if phrases is not None and len(phrases)>0:
			for phrase in phrases:
				allPhrases.append(phrase)
	#checking if these phrases contain feature-words
	updatedPhrases=CheckForFeatureWords(allPhrases,featureSet)
	# finding polarity of phrases obtained
	phraseOpinions=GetOpinions(updatedPhrases)
	print phraseOpinions
	
	
		
		
		
	