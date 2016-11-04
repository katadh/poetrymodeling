from __future__ import division
from ast import literal_eval as make_tuple
from bs4 import BeautifulSoup
from urllib2 import urlopen
import unicodedata
import string
import math

BASE_URL = "http://metrolyrics.com"
ARTISTS_ON_PAGE = 40

PAGE_URLS = [BASE_URL + '/artists-' + ch + '-' for ch in string.ascii_lowercase]
artist_links = []
song_links = []


def get_artist_links(page_url):

	html = urlopen(page_url).read()
	soup = BeautifulSoup(html, "html.parser")	
	links  = []
	links = [str(link.get("href")) for link in soup.find('tbody').find_all('a')]
	genres = [str(link.string) for link in soup.find('tbody').find_all('td') if not link.a and not link.span]
	names = [link.get("content").encode('ascii','ignore') for link in soup.find('tbody').find_all('meta') if link.get("itemprop")!="url"]
	return zip(names, genres, links)

def get_song_links(name, genre, link, checkSub):
	
	links = []

	try:	
		html = urlopen(link).read()
		soup = BeautifulSoup(html, "html.parser")
		links = [(name, genre, str(link.get("href"))) for link in soup.find('tbody').find_all('a')]
		
		if (checkSub and soup.find('span', class_="pages")):
			sub_links += [str(a.get("href")) for a in soup.find('span', class_="pages").find_all('a') if a.get("href") not in artist_links]
			for sublink in sub_links:
				links += get_song_links(name, genre, sublink, False)
	except:	
		return links

	return links

def get_song_lyrics(link):
	lyrics = []
	try:	
		html = urlopen(link).read()
		soup = BeautifulSoup(html, "html.parser")
		html_lyrics = soup.find('div', id="lyrics-body-text").find_all('p')

		for v in html_lyrics:
			lyrics += [str(v).replace("<br>"," ").replace("</br>"," ").replace("<p class=\"verse\">","").replace("</p>","").strip()]

	except:
		return "None"

	return " \n".join(lyrics)


# for url in PAGE_URLS:
# 	html = urlopen(url + "1.html").read()
# 	soup = BeautifulSoup(html, "html.parser")
# 	artist_count = math.ceil(int(soup.find('p', class_="letter-info").strong.string)/ARTISTS_ON_PAGE) 
# 	LETTER_URLS = [url + str(count) + ".html" for count in range(1,int(artist_count)+1)]
	
# 	for page in LETTER_URLS:
# 		artist_links += get_artist_links(page)


# artist_links = [str(entry) for entry in artist_links]

# print "Writing artist links file..."
# with open("metrolyrics_artist_urls.txt", "w") as f:
# 	f.write("\n".join(artist_links))

# with open("metrolyrics_artist_urls.txt") as f:
# 	artist_links = f.readlines()

# for link in artist_links:
# 	artist = make_tuple(link.strip())
# 	song_links += get_song_links(artist[0], artist[1], artist[2], True)

# song_links = [str(entry) for entry in song_links]

# print "Writing song links file..."
# with open("metrolyrics_song_urls.txt", "w") as f:
# 	f.write("\n".join(song_links))

with open("metrolyrics_song_urls.txt") as f:
	song_links = f.readlines()

i = 696678
for link in song_links[710159:]:
	song = make_tuple(link.strip())
	song = list(song)
	song.insert(3, song[2].split('/')[3].split('-lyrics')[0].replace("-"," "))
	song = tuple(song)
	lyrics = get_song_lyrics(song[2])
	if (lyrics!="None"):
		with open("songs/song" + str(i) + ".txt","w") as f:
			f.write(str(song) + "\n")
			f.write(lyrics)
		i += 1
