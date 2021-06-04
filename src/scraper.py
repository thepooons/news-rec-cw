import pandas as pd
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

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
    def __init__(self, homepage_url: str, disallow_string: str, path_to_webdriver: str) -> None:
        self.homepage_url = homepage_url
        self.path_to_webdriver = path_to_webdriver
        self.disallowed_urls = make_disallowed_list(
            disallow_string=disallow_string,
            homepage_url=homepage_url
        )

    @abstractmethod
    def scrap_navigation_bar(self):
        """
        finds the categorised news article collections in the navigation bar
        """
        pass
    
    @abstractmethod
    def scrap_article_links(self):
        """
        scraps all the article links found in each categor  
        """
        pass
    
    @abstractmethod
    def scrap_article_contents(self):
        pass

class BBCNewsScraper(NewsScraper):
    def __init__(self, homepage_url: str, disallow_string: str, path_to_webdriver: str) -> None:
        super().__init__(homepage_url, disallow_string, path_to_webdriver)
        self.driver = webdriver.Chrome(executable_path=path_to_webdriver)

    def scrap_navigation_bar(self):
        
