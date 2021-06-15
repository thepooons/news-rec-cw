# news-rec-cw
News Recommendation Coursework by Puneet Singh and Karanjot Singh.

## How To Run 
- Install all the dependencies using `pip install -r requirements.txt` or `conda env create -f news_env.yml`
- Run the `src/news_scraping/*.py` scripts to scrape news articles from the following websites:
    - BBC News: `src/news_scraping/BBC_scraper.py`
    - Times Of India News: `src/news_scraping/TOI_scraper.py`
    - Yahoo! News: `src/news_scraping/YHNW_scraper.py`
- Change the parameters in `config.yaml` file to intended values.
- [optional] Run the script `src/data_generator/generator.py` to generate clickstream data
- [optional] Run the script `src/data_manager.py` to split the clickstream data into train and test set
- Run `main.py` file to:
    - Train a hybrid Recommender System
    - or Generate recommendations from (and finetune) a hybrid Recommender System