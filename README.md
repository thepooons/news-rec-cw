# news-rec-cw
News Recommendation Coursework by Puneet Singh and Karanjot Singh.

## How To Run 
Note: [optional] steps must only be run if you wish to scrape new data, use different GloVe model, and generate new clickstream data. To skip all the optional steps, download data from [here](https://drive.google.com/drive/folders/1QIpL1x9sdTEtsVCvgUczVq_7D3zboSbY?usp=sharing) and extract `data` directory parallel to `main.py`.

- Install all the dependencies using `pip install -r requirements.txt` or `conda env create -f news_env.yml`
- Change the parameters in `config.yaml` file to intended values.
- [optional] Run the `src/news_scraping/*.py` scripts to scrape news articles from the following websites:
    - BBC News: `src/news_scraping/BBC_scraper.py`
    - Times Of India News: `src/news_scraping/TOI_scraper.py`
    - Yahoo! News: `src/news_scraping/YHNW_scraper.py`
- [optional] Merge all the data scraped in last step into a csv file.
- [optional] Download a GloVe model into `data/GloVe`, alternatively, use given custom trained GloVe vectors.
- [optional] Run the script `src/text_preprocessing.py` to:
    - preprocess the scraped news article text
    - create vector representation of the articles
    - create clusters of news articles from these vectors
- [optional] Run the script `src/data_generator/generator.py` to generate clickstream data
- [optional] Run the script `src/data_manager.py` to split the clickstream data into train and test set
- Run `main.py` file to:
    - Train a hybrid Recommender System
    - or Generate recommendations from (and finetune) a hybrid Recommender System