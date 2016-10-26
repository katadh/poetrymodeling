from __future__ import print_function
from bs4 import BeautifulSoup
import dryscrape
import urllib2
import re
import time
import HTMLParser

def get_poems_for_poet_name(poet):
    poet = poet.lower()
    poet = re.sub('[^a-z]+','-',poet)
    
    url = "http://www.poetryfoundation.org/poems-and-poets/poets/detail/"+poet+"#about"

    get_poems_for_poet(url)

def get_poems_for_poet(url):

    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page.read(), "html5lib")
    parser = HTMLParser.HTMLParser()

    poet_content = soup.find(id='tab-content2')
    
    #poems = poet_content.find_all('a',href=re.compile('.*/poetrymagazine/browse/.*'))
    poems = poet_content.find_all('a',href=re.compile('.*/poems/.*'))
    
    #poems.extend(poems2)
    
    for poem in poems:

        poemUrl = 'http:' + poem.get('href')
        print(poemUrl)
        poemPage = urllib2.urlopen(poemUrl)
        poemSoup = BeautifulSoup(poemPage.read(), "html5lib")
        
        poemTitle = poemSoup.find("span", class_="hdg_1")
        
        if poemTitle:
            poemId = poemUrl.split('/')[-1]
            title = parser.unescape(poemTitle.text).encode('ascii', 'ignore')
            fileout = title + "_" + poemId + ".txt"
            output = open(fileout, 'w')
            #print(parser.unescape(poemTitle.text).encode('ascii', 'ignore'),file=output)
            
            poem = poemSoup.find('div',{'class':'poem'})
            poemContent = poem.find_all('div')
            
            for line in poemContent:
                text = parser.unescape(line.text)
                out = text.encode('utf8').replace("\xc2\xa0", " ").strip()
                #out = text.encode('ascii', 'ignore').strip()
                if out:
                    print(out,file=output)

def get_poets_on_page(url, session):
    #page = urllib2.urlopen(url)
    session.visit(url)
    time.sleep(3)
    response = session.body()
    soup = BeautifulSoup(response)

    poets = soup.find_all('a',href=re.compile('.*/poems-and-poets/poets/detail/.*'))

    poetUrls = set()
    for poet in poets:
        poetUrl = 'http:' + poet.get('href') + '#about'
        poetUrl = poetUrl.encode('ascii', 'ignore')
        poetUrls.add(poetUrl)

    #print(poetUrls)
    return poetUrls

def get_all_poets():

    base_url = 'https://www.poetryfoundation.org/poems-and-poets/poets#page='

    poetUrls = set()
    session = dryscrape.Session()
    for i in range(1,195):
        url = base_url + str(i)
        print(url)
        new_poets = get_poets_on_page(url, session)
        print("Got poets for page")
        poetUrls = poetUrls.union(new_poets)

    with open('poet_found_poets.txt', 'w') as poet_list:
        for poet in poetUrls:
            poet_list.write("%s\n" % poet)

    return poetUrls

def get_poets_from_file(path):

    with open(path, 'r') as poet_list:
        poets = [poet.strip() for poet in poet_list.readlines()]

    return poets

def get_all_poems():

    poets = get_poets_from_file('poet_found_poets.txt')

    for poet in poets:
       get_poems_for_poet(poet) 
