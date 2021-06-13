from abc import ABC, abstractmethod

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
                - (images  
                - tags)
        """
        pass