import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime
from time import sleep
import requests

"""Scraping News"""

def pprint(html: str):
    """pretty print the arg`html` HTML code

    Args:
        html (str): HTML code
    """
    bs = BeautifulSoup(html)
    print(bs.prettify())

def make_disallowed_list(disallow_string: str, homepage_url: str) -> list:
    """makes a list of webpages disallowed by the website's 
    robots.txt file

    Args:
        disallow_string (str): string of all the disallowed webpages 
        copied from website's robots.txt file
        homepage_url (str): index URL of the website

    Returns:
        list: list of URLs not allowed to be scraped
    """
    disallowed_hrefs = [homepage_url + str(href[len("disallow: "):]) for href in disallow_string.strip().split('\n')]
    return disallowed_hrefs

class NewsScraper(ABC):
    @abstractmethod
    def __init__(
        self, 
        homepage_url: str,
        disallow_string: str, 
        path_to_webdriver: str,
        data_dir: str
    ) -> None:
        pass

    @abstractmethod
    def scrape_navigation_bar(self):
        """
        finds the categorised news article collections in the navigation bar
        """
        pass
    
    @abstractmethod
    def scrape_article_links(self):
        """
        add all the article links found on the page to a article_hrefs list  
        """
        pass
    
    @abstractmethod
    def scrape_article_contents(self):
        """
        NOTE: use beautifulsoup, it's faster
        for each link in `article_hrefs`:
            - scrap:
                - title  
                - URL  
                - author  
                - raw_text  
                - publish_date  
                - images  
                - tags
        """
        pass

class BBCNewsScraper(NewsScraper):
    def __init__(self, homepage_url: str, disallow_string: str, path_to_webdriver: str, data_dir: str) -> None:
        self.homepage_url = homepage_url
        self.path_to_webdriver = path_to_webdriver
        self.disallowed_urls = make_disallowed_list(
            disallow_string=disallow_string,
            homepage_url=homepage_url
        )
        self.article_hrefs = []
        self.driver = webdriver.Chrome(executable_path=path_to_webdriver)
        self.data_dir = data_dir

    def scrape_navigation_bar(self): 
        driver = self.driver
        driver.get(self.homepage_url)
        print(f"[{self.__class__.__name__}] scraping: {self.homepage_url}")
        nav_bar = driver.find_element_by_xpath("//ul[@class='gs-o-list-ui--top-no-border nw-c-nav__wide-sections']")
        preview_hrefs_this_page = nav_bar.find_elements_by_class_name("nw-o-link")
        preview_hrefs = [tag.get_attribute("href") for tag in preview_hrefs_this_page]
        self.preview_hrefs = preview_hrefs[1:] # first link leads to same page

    def scrape_article_links(self):
        driver = self.driver
        for preview_href in self.preview_hrefs:
            if not(preview_href in self.disallowed_urls):
                print(f"[{self.__class__.__name__}] finding article links in: {preview_href}")
                driver.get(preview_href)
                href_elements = driver.find_elements_by_xpath(
                    xpath="//a[@class='gs-c-promo-heading gs-o-faux-block-link__overlay-link gel-pica-bold nw-o-link-split__anchor']"
                )
                article_hrefs_this_page = [tag.get_attribute("href") for tag in href_elements]
                for article_href_this_page in article_hrefs_this_page:
                    self.article_hrefs.append(article_href_this_page)

                for reload in range(20): # @TODO: increase this value 
                    # scrap links given in lower part of the page
                    try:
                        href_elements = driver.find_elements_by_xpath(
                            xpath="//a[@class='qa-heading-link lx-stream-post__header-link']"
                        )
                        article_hrefs_this_page = [tag.get_attribute("href") for tag in href_elements]
                    except:
                        break
                    for article_href_this_page in article_hrefs_this_page:
                        self.article_hrefs.append(article_href_this_page)
                    # once all links for current page have been srcaped -> move to next page
                    try:
                        button = driver.find_element_by_xpath(
                            xpath="//a[@class='lx-pagination__btn gs-u-mr+ qa-pagination-next-page lx-pagination__btn--active']"
                        )
                        button.click()
                        sleep(1)
                    except:
                        break
                    
    def scrape_article_contents(self):
        news_data = pd.DataFrame(columns=["url", "title", "raw_text", "author", "publish_datetime"])
        driver = self.driver

        # title class: ssrcss-1pl2zfy-StyledHeading e1fj1fc10
        # raw_text class: ssrcss-18snukc-RichTextContainer e5tfeyi1
        # publish_date data-testid: timestamp
        # url
        self.article_hrefs = list(set(self.article_hrefs))
        for article_index, article_url in enumerate(tqdm(self.article_hrefs)):
            if "news/av/" in str(article_url):
                print(f"[{self.__class__.__name__}] video article found, skipping")
                continue
            if article_url not in self.disallowed_urls:
                html = requests.get(article_url).text
                soup = BeautifulSoup(html, "html.parser")
                # title
                try:
                    title = soup.find(lambda tag: tag.name == 'h1' and tag['class'] == "ssrcss-1pl2zfy-StyledHeading e1fj1fc10".split(" "))
                    title = title.text.strip()
                except:
                    print(f"[{self.__class__.__name__}] title not found, skipping")
                    continue
                # raw_text
                try:
                    raw_text = soup.find_all(class_="ssrcss-18snukc-RichTextContainer e5tfeyi1")
                    raw_text = "".join([str(raw_text_.text.strip()) + " " for raw_text_ in raw_text])
                except:
                    raw_text = ""
                # author
                try:
                    author = soup.find(class_="ssrcss-1pjc44v-Contributor e5xb54n2").text
                    author = author[len("By "):]
                except:
                    author = "bbc"
                # publish_date
                try:
                    publish_datetime = soup.find("time")["datetime"]
                except:
                    publish_datetime = datetime.now()

                this_article_data = pd.DataFrame(data={
                    "url": [article_url],
                    "title": [title],
                    "raw_text": [raw_text],
                    "author": [author],
                    "publish_datetime": [publish_datetime], 
                })
                
                news_data = news_data.append(this_article_data, ignore_index=True)
            if article_index%5 == 0:
                news_data.to_csv(
                    f"{self.data_dir}/{self.__class__.__name__}_at_{str(datetime.today().date())}.csv",
                    index=False
                )
            if article_index%20 == 0:
                print(news_data.tail())

if __name__ == "__main__":
    BBC_DISALLOW_STRING = """
    Disallow: /bitesize/search$
    Disallow: /bitesize/search/
    Disallow: /bitesize/search?
    Disallow: /cbbc/search/
    Disallow: /cbbc/search$
    Disallow: /cbbc/search?
    Disallow: /cbeebies/search/
    Disallow: /cbeebies/search$
    Disallow: /cbeebies/search?
    Disallow: /chwilio/
    Disallow: /chwilio$
    Disallow: /chwilio?
    Disallow: /education/blocks$
    Disallow: /education/blocks/
    Disallow: /newsround
    Disallow: /search/
    Disallow: /search$
    Disallow: /search?
    Disallow: /sport/videos/*
    Disallow: /food/favourites
    Disallow: /food/search*?*
    Disallow: /food/recipes/search*?*
    Disallow: /education/my$
    Disallow: /education/my/
    Disallow: /bitesize/my$
    Disallow: /bitesize/my/
    Disallow: /food/recipes/*/shopping-list
    Disallow: /food/menus/*/shopping-list
    Disallow: /news/0
    Disallow: /ugc$
    Disallow: /ugc/
    Disallow: /ugcsupport$
    Disallow: /ugcsupport/
    Disallow: /userinfo/
    Disallow: /userinfo
    Disallow: /u5llnop$
    Disallow: /u5llnop/
    Disallow: /sounds/search$
    Disallow: /sounds/search/
    Disallow: /sounds/search?
    Disallow: /ws/includes
    Disallow: /radio/imda
    """

    bbc_scrapper = BBCNewsScraper(
        homepage_url="https://www.bbc.com/news",
        disallow_string=BBC_DISALLOW_STRING,
        path_to_webdriver="chromedriver.exe",
        data_dir="data"
    )
    bbc_scrapper.scrape_navigation_bar()
    bbc_scrapper.scrape_article_links()
    bbc_scrapper.scrape_article_contents()