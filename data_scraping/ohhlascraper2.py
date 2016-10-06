from bs4 import BeautifulSoup
from urllib2 import urlopen

with open("ohhla_song_urls.txt") as f:
	lines = f.read()[:-1].split("\n")

def get_doc(page_url):
	html = urlopen(page_url).read()
	soup = BeautifulSoup(html, "lxml")
	text = soup.find('pre').string
	title = '-'.join(page_url[27:].split("/"))
	with open("../data/ohhlaraw/"+title, "w") as f:
		f.write(text.encode('utf-8'))
	print "successfully wrote", title
	return 1
	
for url in lines:
	try:
		get_doc(url)
	except:
		print "failed on", url
	
