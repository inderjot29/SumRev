'''
Created on April 17, 2016

@author: inderjot, aprajita
'''
import semantic_analysis
import predict_rating
from semantic_analysis import GetFeatureSet,ReadReviews,WritingResults,EvaluateUsingClassifiers,CalculatingPrecisionRecall
from predict_rating import PredictRating

if __name__ == '__main__':
	#returns list of featureWords
	featureSet=GetFeatureSet()
	fout=open('Count.txt','w+')
	fout.write('GOOD \t BAD \t NEUTRAL \n')
	#reading reviews
	path='./InputData/Reviews.txt'
	phraseOpinions=ReadReviews(path,featureSet,fout)
	fout.close()
	#combining training and testing set to furthur evaluate results
	#print selectedFeatures
	#OpinionsSet=dict(trainphraseOpinions.items()+ testphraseOpinions.items())
	WritingResults(phraseOpinions)
	EvaluateUsingClassifiers(phraseOpinions)
	CalculatingPrecisionRecall()
	PredictRating()