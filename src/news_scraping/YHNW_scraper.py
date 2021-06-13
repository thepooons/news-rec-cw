import numpy as np
import pandas as pd
import time
import datetime

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_links(links_to_scrape, link_type="one", attribute_chosen="article-body"):
    # Placeholders
    list_heading = []
    list_content = []
    list_links = []
    bar = tqdm(iterable=links_to_scrape, leave=True, position=0)
    # Traverse over the links
    for link in bar:
        # Request
        page_req = requests.get(link)

        # Check Status code
        if page_req.status_code == 200:
            # Perform scraping
            flag = False
            soup = BeautifulSoup(markup=page_req.content, features="lxml")
            for tag in soup.findAll("div", attrs={"class": str(attribute_chosen)}):
                flag = True
                if link_type == "one":
                    list_content.append(tag.get_text())
                else:
                    list_content.append(
                        " ".join([i.get_text() for i in tag.findAll("p")]))

            # Check if body was found
            if flag:
                # Collect the potential heading
                heading = soup.find("title").string

                # Check if heading is none
                if heading == None:
                    list_heading.pop(-1)
                    continue
                else:
                    list_heading.append(heading)
                    list_links.append(link)
            else:
                # Move to the next point
                continue

            # Printer
            bar.set_description("%s, %s, %s, %s" % (
                heading[:50], len(list_content), len(list_heading), len(list_links)))

        else:
            # Bad request
            raise Exception("Error Code --> %d" % (page_req.status_code))

    # Return
    return list_content, list_heading, list_links


class YHNW_scraper(object):
    """
    Creates the Yahoo News Scraper
    """

    def __init__(self, sleep_duration, scroll_depth):
        # Make the instance variables
        self.sleep_duration = sleep_duration
        self.driver = webdriver.Chrome("chromedriver.exe")
        self.scroll_depth = scroll_depth
        self.global_var = 0

    def infinite_scroll(self, wait_time=1.5):
        # Fetch webpage
        self.driver.maximize_window()
        scroll = 0
        for i in range(self.scroll_depth):
            self.driver.execute_script(
                "window.scrollTo(%d, 1000000);" % scroll)
            scroll = self.driver.execute_script(
                "return document.body.scrollHeight")
            time.sleep(wait_time)

        # sleep
        time.sleep(2)

    def infinite_scroll_with_restart(self, wait_time=3, depth=20):
        # Fetch webpage
        self.driver.maximize_window()
        scroll = 0
        for i in range(depth):
            if i == 3 or i == 1:
                self.driver.execute_script(
                    "window.scrollTo(%d, %d);" % (0,  scroll - 500000))
            else:
                self.driver.execute_script(
                    "window.scrollTo(%d, 1000000);" % scroll)
                scroll = self.driver.execute_script(
                    "return document.body.scrollHeight")
            time.sleep(wait_time)

        # sleep
        time.sleep(2)

    def _link_collector(self):
        # Harcode the links that have the same format
        link_library = {
            "type_one": ["//h3[@class='Mb(5px)']//a", [
                "https://in.news.yahoo.com/",
                "https://in.news.yahoo.com/world",
                "https://in.news.yahoo.com/national",
                "https://in.finance.yahoo.com/",
                "https://in.finance.yahoo.com/personal-finance",
                "https://in.finance.yahoo.com/topic/careers",
                "https://in.finance.yahoo.com/topic/real-estate",
                "https://in.finance.yahoo.com/topic/loans-deposits",
                "https://in.finance.yahoo.com/topic/insurance",
                "https://in.finance.yahoo.com/topic/tax",
                "https://in.finance.yahoo.com/topic/mutual-fund",
                "https://in.finance.yahoo.com/topic/tech",
                "https://in.finance.yahoo.com/topic/autos",
                "https://in.style.yahoo.com/",
                "https://in.news.yahoo.com/sports"

            ]],
            "type_two": ["//div[@class='article-thumbnail']//a", [
                "https://cricket.yahoo.net/news",
            ]]
        }

        # Scape the links
        list_links_type_one = []
        list_links_type_two = []

        for link_type, data in link_library.items():
            # link to scrape in
            for link_to_scrape in data[-1]:
                # Fetch the link
                self.driver.get(link_to_scrape)

                # Scroll to end
                if link_type == "type_one":
                    self.infinite_scroll()
                    xpath = data[0]
                    scroll = self.driver.find_elements_by_xpath(xpath=xpath)
                    links_curr = [s.get_attribute("href") for s in scroll]
                    list_links_type_one.extend(links_curr)
                else:
                    self.infinite_scroll_with_restart(wait_time=3)
                    xpath = data[0]
                    scroll = self.driver.find_elements_by_xpath(xpath=xpath)
                    links_curr = [s.get_attribute("href") for s in scroll]
                    list_links_type_two.extend(links_curr)

                self.global_var += len(links_curr)
                print("%s : %d -- %d" %
                      (link_to_scrape, len(links_curr), self.global_var))

        # Reurn links
        return list_links_type_one, list_links_type_two

    def _scrape(self):
        # Collect type1 and type 2 links
        list_links_type_one, list_links_type_two = self._link_collector()

        # Close the driver
        self.driver.quit()

        # Collect seperate links
        list_content_t1, list_heading_t1, list_links_t1 = scrape_links(
            links_to_scrape=list_links_type_one, attribute_chosen="caas-body")
        list_content_t2, list_heading_t2, list_links_t2 = scrape_links(
            links_to_scrape=list_links_type_two, link_type="two")

        # Save as dataframe
        data_df = pd.DataFrame({
            "title": list_heading_t1 + list_heading_t2,
            "content": list_content_t1 + list_content_t2,
            "links": list_links_t1 + list_links_t2
        })

        data_df.to_csv("YHNW_data_%s.csv" %
                       datetime.datetime.now().date(), index=False)

        return "Done"

    def run_pipe(self):
        status = self._scrape()


# Run the scraper
inst = YHNW_scraper(1, 5)
inst.run_pipe()
