from bs4 import BeautifulSoup
import urllib2
import HTMLParser

def get_limerick(url):

    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page.read(), "html5lib")
    #print soup
    parser = HTMLParser.HTMLParser()

    try:
        main_content = soup.find(class_="widetable")
        poem_content = main_content.find(class_="limerickverse")
    except:
        return None

    text = parser.unescape(poem_content.text)
    out = text.encode('utf8', 'ignore').strip()

    return out
    

def write_limerick(output_dir, limerick_id, text):
    fileout = output_dir + '/' + str(limerick_id) + ".txt"
    with open(fileout, 'w') as limerick_file:
        limerick_file.write(text)

def scrape(output_dir):

    base_url = "http://www.oedilf.com/db/Lim.php?LimerickId="

    for i in range(1, 96860):
        print i
        limerick = get_limerick(base_url + str(i))
        if limerick:
            write_limerick(output_dir, i, limerick)
