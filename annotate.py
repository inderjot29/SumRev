#!/usr/bin/python

import sys, getopt
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import numpy as np
import csv

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile
   ReadFileandPrint(inputfile,outputfile)
   
def ReadFileandPrint(inputfile,outputfile):
	#Load the dataset
	fout=open(outputfile,'w+')
	with open(inputfile, 'rb') as csvfile:
		albumreader = csv.reader(csvfile, delimiter='\t')
		
		for row in albumreader:
			print row[0] 
			response=input("Enter 1(good) or 2 (bad) or 3(neutral):")
			if not response:
				continue
			elif response==1:
				fout.write("good\n")
			elif response==2:
				fout.write("bad\n")
			elif response==3 :
				fout.write("neutral\n")
			else:
				continue
	fout.close()
			
			
				
	

if __name__ == "__main__":
   main(sys.argv[1:])