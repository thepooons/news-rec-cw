call activate news_env
python src\data_generator\generator.py
python src\data_manager.py
python main.py
call conda deactivate