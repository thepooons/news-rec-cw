from src import scraper

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

bbc_scrapper = scraper.BBCNewsScraper(
    homepage_url="https://www.bbc.com/news",
    disallow_string=BBC_DISALLOW_STRING,
    path_to_webdriver="chromedriver.exe"
)