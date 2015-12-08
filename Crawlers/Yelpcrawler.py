import urllib
import bs4
from bs4 import BeautifulSoup
import re
from threading import Thread

#List of yelp urls to scrape
url=['http://www.yelp.ca/c/montr%C3%A9al/restaurants']
i=0
#function that will do actual scraping job
def scrape(ur):
	restaurants={}
	html = urllib.urlopen(ur).read()
	soup = BeautifulSoup(html,'html.parser')
	reviewsText=[]
	restaurantColumns = soup.findAll("ul", { "class" : "best-of-columns_column ylist ylist-bordered" })
	restaurantList=[]
	for column in restaurantColumns :
		#restaurantList.append(soup.findAll("li", { "class" : "media-block media-block--12" }))
		restaurantList=soup.findAll("a", { "class" : "biz-name" })
	for resturant in restaurantList:
		#resturant=BeautifulSoup(resturant)
		#rest = resturant.find("a",{ "class" : "biz-name" })
		
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
		#print restaurant
		html1 = urllib.urlopen(restaurants[restaurant]).read()
		soup1 = BeautifulSoup(html1)
		
		reviewsList=[]
		
		
		reviewsList=soup1.findAll('p',itemprop="description")
		
		#print len(reviewsList)
		#print reviewsList
		f=open(restaurant+'.txt','w+')
		for review in reviewsList:
			review=review.text.strip()
			
			f.write(review.encode('utf-8'))
			f.write('\n')
		f.close()

         
	print "-------------------"


scrape(url[0])
