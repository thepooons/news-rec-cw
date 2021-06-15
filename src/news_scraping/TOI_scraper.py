import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests as req
import time
from tqdm import tqdm
import datetime


def collect_data_via_links(link_list, sleep_duration):
    """
    The link collected by scraped pages

    Args:
        link_list (list): All the links to scrape

        sleep_duration (float): Sleep in between scrapes

    Returns:
        [tuple]: heading, content
    """
    # Scrape collected links
    list_heading = []
    list_body = []

    # Create a bar to monitor progress
    bar = tqdm(link_list, position=0, leave=True)

    # Scrape the links
    for link in bar:
        # Sleep to avoid quick request rejection
        time.sleep(sleep_duration)

        # Collect page
        get_page = req.get(link)
        if get_page.status_code == 200:
            content = get_page.content
            soup = BeautifulSoup(markup=content, features="lxml")

            # Get body and heading
            flag = False
            for tag in soup.findAll("div", attrs={"class": "_3YYSt clearfix"}):
                flag = True
                list_body.append(str(tag.get_text()))

            if flag:
                # Get heading
                title = soup.find("title").string
                if title == None:
                    if flag == True:
                        list_body.pop(-1)
                    continue
                else:
                    heading = title.split("|")[0].strip()
                    list_heading.append(heading)

                bar.set_description(
                    "%s, %s, %s" % (heading[:10], len(list_body), len(list_heading))
                )
        else:
            continue

    # Return Data
    return list_heading, list_body


def collect_links_consistent(soup, attrs, link_to_collect, element_tag="div"):
    """
    Collect links from sraped page

    Args:
        soup (bs4.soup): The soup of page

        attrs (dict): Pages tags to look for

        link_to_collect ([type]): [description]

        element_tag (str, optional): Which tag to look for. Defaults to "div".

    Returns:
        [list]: Scraped links
    """
    # Placeholder
    links = []
    for tag in soup.findAll(element_tag, attrs=attrs):
        for tg in tag.findAll("a"):
            href = tg.get("href")
            if "articleshow" in href and "cms" in href:
                links.append(link_to_collect + href)

    # Return links
    return links


def collect_soup(link_to_collect):
    """
    Collects the soup

    Args:
        link_to_collect (string): Link to scrape
    """
    # Make Request
    get_page = req.get(link_to_collect)

    # Check status code
    if get_page.status_code == 200:
        content = get_page.content
        soup = BeautifulSoup(markup=content, features="lxml")

        # Return the soup
        return soup
    else:

        # Return Error
        raise Exception("Error Collecting soup")


class TOI_SCRAPER(object):
    """
    Scrape all the pages of Times of India

    """

    def __init__(self, sleep_duration, total_pages):
        # Initilaize variables
        self.sleep_duration = sleep_duration
        self.total_pages = total_pages

    def _scraper(self, total_pages):
        """
        The main scraper function

        Args:
            total_pages (int): Total pages to scrape
        Returns:
            [tuple]: List of headings and corresponding body of text
        """
        # Make some assertions
        assert total_pages > 1, "Total Pages collected will be zero"

        # Collect the tag
        links_all = []
        link_global = {
            "https://timesofindia.indiatimes.com/city": [
                True,
                "ul",
                {"class": "cvs_wdt clearfix"},
            ],
            "https://timesofindia.indiatimes.com/india": [
                True,
                "div",
                {"id": "c_wdt_list_1"},
            ],
            "https://timesofindia.indiatimes.com/world/rest-of-world": [
                True,
                "div",
                {"id": "c_010205"},
            ],
            "https://timesofindia.indiatimes.com/sports/cricket": [
                False,
                "div",
                {"class": "top-newslist small"},
            ],
            "https://timesofindia.indiatimes.com/sports/football": [
                False,
                "div",
                {"class": "top-newslist small"},
            ],
            "https://timesofindia.indiatimes.com/sports/hockey": [
                False,
                "div",
                {"class": "top-newslist small js-main-news-list"},
            ],
            "https://timesofindia.indiatimes.com/sports/tennis": [
                False,
                "div",
                {"class": "top-newslist small"},
            ],
            "https://timesofindia.indiatimes.com/sports/wwe": [
                False,
                "div",
                {"class": "top-newslist small js-main-news-list"},
            ],
            "https://timesofindia.indiatimes.com/sports/nfl": [
                False,
                "div",
                {"class": "top-newslist small js-main-news-list"},
            ],
            "https://timesofindia.indiatimes.com/business/economy": [
                True,
                {"id": "c_articlelist_stories_1"},
            ],
            "https://timesofindia.indiatimes.com/business/international-business": [
                True,
                {"id": "c_articlelist_stories_2"},
            ],
            "https://timesofindia.indiatimes.com/business/cryptocurrency": [
                False,
                "div",
                {"id": "c_articlelist_stories_2"},
            ],
            "https://timesofindia.indiatimes.com/business/real-estate": [
                True,
                "div",
                {"id": "c_articlelist_stories_1"},
            ],
            "https://timesofindia.indiatimes.com/business/telecom": [
                True,
                "div",
                {"id": "c_articlelist_stories_1"},
            ],
            "https://timesofindia.indiatimes.com/business/aviation": [
                True,
                "div",
                {"id": "c_articlelist_stories_1"},
            ],
            "https://timesofindia.indiatimes.com/business/personal-finance": [
                True,
                "div",
                {"id": "c_articlelist_stories_1"},
            ],
            "https://timesofindia.indiatimes.com/business/corporate": [
                True,
                "div",
                {"id": "c_articlelist_stories_1"},
            ],
        }

        # Collect the links
        for link_to_collect, attribute in link_global.items():
            # Make a checker
            len_curr = 0

            # Check if multipage
            if attribute[0] == True:
                for page in range(1, total_pages):
                    # Get the page
                    if page != 1:
                        collect_link = link_to_collect + "/" + str(page)
                    else:
                        collect_link = link_to_collect

                    # Collec the soup
                    soup = collect_soup(link_to_collect=collect_link)

                    # Start script
                    links_curr = collect_links_consistent(
                        soup=soup,
                        attrs=attribute[-1],
                        link_to_collect=link_to_collect,
                        element_tag=attribute[0],
                    )
                    len_curr += len(links_curr)
                    links_all.extend(links_curr)

                # Printer
                print("%s : %d -- %d" % (link_to_collect, len_curr, len(links_all)))

            else:

                # Collect the soup
                soup = collect_soup(link_to_collect=link_to_collect)

                # Start script
                links_curr = collect_links_consistent(
                    soup=soup,
                    attrs=attribute[-1],
                    link_to_collect=link_to_collect,
                    element_tag=attribute[0],
                )
                links_all.extend(links_curr)

                # Printer
                print(
                    "%s : %d -- %d" % (link_to_collect, len(links_curr), len(links_all))
                )

            # Sleep here as well
            time.sleep(1)

        # Scrape collected links
        list_heading, list_body = collect_data_via_links(
            link_list=links_all, sleep_duration=self.sleep_duration
        )

        # Return
        return list_heading, list_body

    def make_dataframe_combined(self):
        # Loading all the fucntions
        list_headings_all, list_body_all = self._scraper(total_pages=self.total_pages)

        assert len(list_headings_all) == len(
            list_body_all
        ), "Error Heading and body mismatch %d-%d" % (
            len(list_headings_all),
            len(list_body_all),
        )

        # Make DataFrame
        print("Creating dataframe.....")
        global_df = pd.DataFrame(
            {"heading": list_headings_all, "content": list_body_all}
        )

        global_df.to_csv(
            "TOI_DATA_%s.csv" % (datetime.datetime.now().date()), index=False
        )


SLEEP_DURATION = 0.1
TOTAL_PAGES = 5
instance = TOI_SCRAPER(sleep_duration=SLEEP_DURATION, total_pages=TOTAL_PAGES)
instance.make_dataframe_combined()
