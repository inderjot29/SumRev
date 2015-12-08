import urllib
import bs4
from bs4 import BeautifulSoup
import re
from threading import Thread

#List of yelp urls to scrape
url=['http://www.yelp.ca/biz/namast%C3%A9-montr%C3%A9al-montr%C3%A9al-2']
i=0
#function that will do actual scraping job
def scrape(ur):
	restaurants={}
	html = urllib.urlopen(ur).read()
	soup = BeautifulSoup(html,'html.parser')
	reviewsText=[]
	
	pageNo=soup.find("div",{"class":"page-of-pages arrange_unit arrange_unit--fill"})
	
	totalPages=pageNo.text.strip()
	totalPages=totalPages.split('of ')[1]
	print totalPages

	reviewsList=[]
	counter=0
	i=1
	f=open('NamasteMontreal.txt','w+')
	for i in range(1,int(totalPages)+1):
		if i==1 :
			newUrl=None
		else:
			counter=counter+20
			newUrl=url[0]+"?start="+str(counter)
		if newUrl is None:
		
			reviewsList=soup.findAll('p',itemprop="description")
		else:
			html1 = urllib.urlopen(newUrl).read()
			soup1 = BeautifulSoup(html1,'html.parser')
			reviewsList=soup.findAll('p',itemprop="description")

		for review in reviewsList:
			review=review.text.strip()
			f.write(review.encode('utf-8'))

	f.close()
     
	print "-------------------"


scrape(url[0])
