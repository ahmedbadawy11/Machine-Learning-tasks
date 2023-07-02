import requests
from bs4 import BeautifulSoup


class Scraping:
    def __init__(self, url):
        self.url = url

    def scraping(self, attr, find_all=True):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")
        if find_all:
            a = [fonds_title.text for fonds_title in soup.find_all('div', attrs=attr)]
        else:
            a = soup.find('div', attrs=attr).text

        return a


r = Scraping('https://biblio.uottawa.ca/en/women-in-stem')
print(r.scraping({'class': 'field field-name-title-field field-type-text field-label-hidden'}))
