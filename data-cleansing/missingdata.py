import pandas as pd


def clean_data(df):
    return df.dropna() # Drop null values


def main():
    # data files can be found at https://grouplens.org/datasets/movielens/
    print("Loading movies data...")
    movies = pd.read_csv('./movielens/movies.csv', sep=',')
    print("Movies data successfully loaded.")
            
    print("Loading tags data...")
    tags = pd.read_csv('./movielens/tags.csv', sep=',')
    print("Tags data successfully loaded.")
    
    print("Loading ratings data...")
    ratings = pd.read_csv('./movielens/ratings.csv', sep=',')
    print("Ratings data successfully loaded.")
    
    movies = clean_data(movies)
    tags = clean_data(tags)
    ratings = clean_data(ratings)
    

if __name__ == '__main__':
    main()
