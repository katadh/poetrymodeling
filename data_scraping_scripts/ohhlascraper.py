from bs4 import BeautifulSoup
from urllib2 import urlopen

BASE_URL = "http://ohhla.com"

PAGE_URLS = ["http://ohhla.com/all.html",
			 "http://ohhla.com/all_two.html",
			 "http://ohhla.com/all_three.html",
			 "http://ohhla.com/all_four.html",
			 "http://ohhla.com/all_five.html",]

def get_artist_links(page_url):
    html = urlopen(page_url).read()
    soup = BeautifulSoup(html, "lxml")
    basic_links = [BASE_URL + "/" + a.get("href") for a in soup.findAll("a") if a.get("href") and a.get("href")[:4] == 'anon']
    complex_links = [BASE_URL + "/" + a.get("href") for a in soup.findAll("a") if a.get("href") and a.get("href")[-4:] == 'html']
    return basic_links, complex_links

def basic_get_album_links(artist_url):
    html = urlopen(artist_url).read()
    soup = BeautifulSoup(html, "lxml")
    links = [artist_url + a.get("href") for a in soup.findAll("a") if a.get("href") and a.get("href") != '/anonymous/']
    return links

def basic_get_song_links(album_url):
    html = urlopen(album_url).read()
    soup = BeautifulSoup(html, "lxml")
    links = [album_url + a.get("href") for a in soup.findAll("a") if a.get("href") and a.get("href") != '/anonymous/']
    return links

def complex_get_song_links(artist_url):
    html = urlopen(artist_url).read()
    soup = BeautifulSoup(html, "lxml")
    links = [BASE_URL + "/" + a.get("href") for a in soup.findAll("a") if a.get("href") and a.get("href")[-4:] == '.txt']
    return links


basic_artist_links = []
complex_artist_links = []
for page in PAGE_URLS:
	b, c = get_artist_links(page)
	basic_artist_links += b
	complex_artist_links += c

album_links = []
for a in basic_artist_links:
	print "getting artist", a
	try:
		album_links += basic_get_album_links(a)
	except:
		print a, "has failed!"

song_links = []
for a in album_links:
	print "getting album", a
	try:
		song_links += basic_get_song_links(a)
	except:
		print a, "has failed!"	
for a in complex_artist_links:
	print "getting artist", a
	try:
		song_links += complex_get_song_links(a)
	except:
		print a, "has failed!"	

song_links = [link for link in song_links if not "//anonymous/" in link]

with open("ohhla_song_urls.txt", "w") as f:
	f.write("\n".join(song_links))
