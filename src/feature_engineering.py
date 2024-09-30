from itertools import count

import numpy as np
import pandas as pd
import os

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
from wajig.perform import output

game_detail_transform_1=ColumnTransformer(
    transformers=[
        ("GenresTransformer",CountVectorizer(),"genres"),
        ("overall_player_rating_transform",FunctionTransformer(lambda ser: ser.apply( lambda x:['Overwhelmingly Positive', 'Very Positive', 'Mostly Positive','Mixed', 'Mostly Negative'].index(x)  ).to_frame()),"overall_player_rating")
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

column_transformer=ColumnTransformer(
    transformers=[
        ("game_id",FunctionTransformer(lambda ser: ser.to_numpy().reshape(-1,1)),"game_id"),
        ("user_id",FunctionTransformer(lambda ser: ser.to_numpy().reshape(-1,1)),"user_id"),
        ("game details transform",game_detail_transform_3,["genres","release_year","number_of_reviews_from_purchased_people","game_avg_hrs_played","overall_player_rating"]),
        ("minmaxscaler",MaxAbsScaler(),["user_avg_hrs_played"]),
        ("target_transformer",FunctionTransformer(lambda ser: ser.apply( lambda x:["Not Recommended","Recommended"].index(x)  ).to_numpy().reshape(-1,1)),"recommendation")
    ],remainder="drop"
)
column_transformer


def read_data(filename)->pd.DataFrame:
    path=os.path.join("data",f"{filename}.csv")
    return pd.read_csv(path)[["genres","release_year","number_of_reviews_from_purchased_people","game_avg_hrs_played","overall_player_rating","game_id",
                                   "user_avg_hrs_played","user_id","recommendation"
                                   ]]
def main(file_name):
    df=read_data(file_name)
    df=column_transformer.fit_transform(df)
    df=df.toarray()
    index=np.where(df[:,-1]==0)
    np.save(f'data/{file_name}_class_0.npy',df[index])
    output=f"{file_name}, class_0_shape: {df[index].shape}"
    index=np.where(df[:,-1]==1)
    np.save(f'data/{file_name}_class_1.npy',df[index])
    output+=f", class_1_shape: {df[index].shape}, unique(user_id:{np.unique(df[:,1]).size},game_id:{np.unique(df[:,0]).size})"
    print(output)
    return output


if __name__=="__main__":
    with open("data/data_details.txt","w")as f:
        text=main("train")
        text+="\n"
        text+=main("test")
        text+="\n"
        text+=main("val")
        f.write(text)


