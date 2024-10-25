import pandas as pd
import numpy as np

# Use the raw link to the CSV file
url = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/a6bcba4b6f9b87d8f924df1dacad300785571cfe/Movies%20Recommendation.csv'
df = pd.read_csv(url)

# Display the first few rows of the DataFrame
print(df.head())
#check other details
print(df.info())
print(df.shape)
print(df.columns)
#get feature selection . Below the double brackets returns DataFrame and the single bracket returns Series 
df_features=df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']].fillna('')
#fillna('') fills Nan values inside columns with empty string or make that particular cell empty

print(df_features.shape)
print(df_features.head())

x=df_features['Movie_Genre']+' '+df_features['Movie_Keywords']+' '+df_features['Movie_Tagline']+' '+df_features['Movie_Cast']+' '+df_features['Movie_Director']
print(x.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
# Fit and transform the text data into a TF-IDF matrix
x=tfidf.fit_transform(x)
print(x.shape)
from sklearn.metrics.pairwise import cosine_similarity
#measuring document similarity
similarity_score=cosine_similarity(x)
# get movie name as input from user and validate for closest spelling
movie_name=input('enter you favorite movie name:')
#tolist() method converts Dataframe column into a python list
all_movie_title_list=df['Movie_Title'].tolist()
import difflib
movie_recommendation=difflib.get_close_matches(movie_name,all_movie_title_list)
close_match=movie_recommendation[0]
#index of close match movie
c_index=df[df.Movie_Title==close_match]['Movie_ID'].values[0]

#getting the list of similar values
recommendation_score=list(enumerate(similarity_score[c_index]))

#get all movies sort based on recommendation score wrt favorite movie
sorted_similar_movies=sorted(recommendation_score,key=lambda x:x[1],reverse=True)

                          
print("the top 10 movies suggested for you: \n")
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title=df[df.index==index]["Movie_Title"].values[0]
    if(i<11):
        print(i,'-',title)
        i+=1
        

      





