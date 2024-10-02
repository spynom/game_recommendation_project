# import libraries
import numpy as np
import pandas as pd
import os
import joblib

from nest_asyncio import apply
from numpy.ma.core import remainder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    FunctionTransformer,
    MaxAbsScaler
)

# function to convert overall_player_rating into int values contain frame
def overall_player_rating_transform(ser):
    def rating_transform(x):
        return ['Overwhelmingly Positive', 'Very Positive', 'Mostly Positive','Mixed', 'Mostly Negative'].index(x)

    return ser.apply(rating_transform).to_frame()


game_detail_transform_1=ColumnTransformer(
    transformers=[
        ("GenresTransformer",CountVectorizer(),"genres"),
        ("overall_player_rating_transform",FunctionTransformer(overall_player_rating_transform),"overall_player_rating")
    ],remainder="passthrough"
)
game_detail_transform_2=Pipeline(
    steps=[
        ("game_detail_transformer_1",game_detail_transform_1),
        ("scaler_release_year",MaxAbsScaler()),
    ]
)
game_detail_transform_3=ColumnTransformer(
    transformers=[
        ("game_detail_transformer_2",game_detail_transform_2,["genres","release_year","number_of_reviews_from_purchased_people","overall_player_rating","game_avg_hrs_played"]),

    ],remainder="passthrough"
)
# function to convert series to numpy
def ser_to_numpy(ser):
    return ser.to_numpy().reshape(-1,1)

# function to transform target column into model reading values
def target_transform(ser):
    return (ser=="Recommended").astype(np.float16).to_numpy().reshape(-1,1)
    #ser.apply( lambda x:["Not Recommended","Recommended"].index(x)  ).to_numpy().reshape(-1,1)
column_transformer=ColumnTransformer(
    transformers=[
        ("game_id",FunctionTransformer(ser_to_numpy),"game_id"),
        ("user_id",FunctionTransformer(ser_to_numpy),"user_id"),
        ("game details transform",game_detail_transform_3,["genres","release_year","number_of_reviews_from_purchased_people","game_avg_hrs_played","overall_player_rating"]),
        ("minmaxscaler",MaxAbsScaler(),["user_avg_hrs_played"]),
        ("target_transformer",FunctionTransformer(target_transform),"recommendation")
    ],remainder="drop"
)
column_transformer

# function to read file
def read_data(filename)->pd.DataFrame:
    path=os.path.join("data",f"{filename}.csv") # path of file
    return pd.read_csv(path)[["genres","release_year","number_of_reviews_from_purchased_people","game_avg_hrs_played","overall_player_rating","game_id",
                                   "user_avg_hrs_played","user_id","recommendation"
                                   ]]
def main(file_name): # function to load, transform, save the files
    df=read_data(file_name)

    df=column_transformer.fit_transform(df) if file_name == "train" else column_transformer.transform(df) # applying transform
    df=df.toarray() # converting sparse numpy to array
    index=np.where(df[:,-1]==0) # index of instances contain target value 0
    np.save(f'data/{file_name}_class_0.npy',df[index]) # saving the numpy array contain target value 0
    output=f"{file_name}, class_0_shape: {df[index].shape}"
    index=np.where(df[:,-1]==1) # index of instances contain target value 1
    np.save(f'data/{file_name}_class_1.npy',df[index]) # saving the numpy array contain target value 1
    output+=f", class_1_shape: {df[index].shape}, unique(user_id:{np.unique(df[:,1]).size},game_id:{np.unique(df[:,0]).size})"
    print(output)
    return output # returning short information regarding save filed


if __name__=="__main__":
    with open("data/data_details.txt","w")as f:
        text=main("train") # applying main function on train dataset
        text+="\n"
        text+=main("test") # applying main function on test dataset
        text+="\n"
        text+=main("val") # applying main function on val dataset
        f.write(text)
        joblib.dump(column_transformer, 'models/column_transformer.pkl') # saving trained column transformer


