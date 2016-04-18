import urllib
import bs4
from bs4 import BeautifulSoup
import re
from threading import Thread

i=0
#function that will do actual scraping job
def scrape(ur,f,frating):
	restaurants={}
	html = urllib.urlopen(ur).read()
	soup = BeautifulSoup(html,'html.parser')
	reviewsText=[]
	restaurantList=[]
	restaurantList=soup.findAll("a", { "class" : "biz-name" })
	#del restaurantList[-8:]
	for resturant in restaurantList:
		
		if resturant.contents[0].name == 'span':
			name=resturant.contents[0]
			name=name.contents
			name=name[0]
		else:
			name=resturant.contents[0]
		
		url=resturant['href']
		if len(url)<150 :
			restaurants[name]='http://www.yelp.ca'+url
	for restaurant in restaurants:
		html1 = urllib.urlopen(restaurants[restaurant]).read()
		soup1 = BeautifulSoup(html1,'html.parser')
		
		reviewsText=[]
	
		pageNo=soup1.find("div",{"class":"page-of-pages arrange_unit arrange_unit--fill"})
		print pageNo
		totalPages=pageNo.text.strip()
		totalPages=totalPages.split('of ')[1]
		print totalPages

		
		counter=0
		i=1
		for i in range(1,int(totalPages)+1):
			reviewsList=[]
			ratingsList=[]
			if i==1 :
				newUrl=None
			else:
				counter=counter+20
				newUrl=restaurants[restaurant]+"?start="+str(counter)
			if newUrl is None:
		
				for content in soup1.findAll("div",{"class":"review-content"}):
					reviewsList.append(content.find('p',itemprop="description"))
					ratingsList.append(content.find('meta',itemprop="ratingValue")['content'])
					
			else:
				html2 = urllib.urlopen(newUrl).read()
				soup2 = BeautifulSoup(html2,'html.parser')
				for content in soup2.findAll("div",{"class":"review-content"}):
					reviewsList.append(content.find('p',itemprop="description"))
					ratingsList.append(content.find('meta',itemprop="ratingValue")['content'])
			
			for review in reviewsList:
				review=review.text.strip()
				f.write(review.encode('utf-8'))
				f.write("\n")
			
			for rating in ratingsList:
				frating.write(rating.encode('utf-8'))
				frating.write("\n")

		print "-------------------"
			
#Open new files for collecting reviews and ratings
f=open('Reviews.txt','w+')
frating=open('Ratings.txt','w+')
purl = 'https://www.yelp.ca/search?find_loc=Montr%C3%A9al,+QC,+Canada&start=';
i = 1
counter = 0
for i in range(1,15):
	url = purl + str(counter) + '&cflt=food';
	scrape(url,f,frating)
	counter = counter + 10
			
f.close()
frating.close()