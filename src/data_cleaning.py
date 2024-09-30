import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import ast
# Load data function
def read_data(file_name:str,low_memory=True)->pd.DataFrame:
    if os.getcwd().split('/')[-1] == 'src':
        path = os.path.join("..","data",file_name)
    else:
        path = os.path.join("data",file_name)
    return pd.read_csv(path,low_memory=low_memory)

def steam_game_reviews_clean(df: pd.DataFrame) -> pd.DataFrame:
    df=df.assign(
        hours_played=df.hours_played.str.replace(r"[^.0-9]","",regex=True).astype(float).astype(int),
        helpful=df.helpful.str.replace(r"[^0-9]","",regex=True).astype(int),
        funny=df.funny.str.replace(r"[^0-9]","",regex=True).astype(int),
        username=df.username.str.split("\n").str.get(0).str.replace("!","").str.strip()

    ).drop(columns=["date"])
    df=df[~(df.username.isnull())&~(df.username=="")&~(df.username==".")]

    return df

def games_description_clean(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()

    df.loc[df[~df.number_of_reviews_from_purchased_people.str.replace(r"[^A-Za-z0-9]","",regex=True).str.isdigit()].index,["number_of_reviews_from_purchased_people"]]=df[~df.number_of_reviews_from_purchased_people.str.replace(r"[^A-Za-z0-9]","",regex=True).str.isdigit()].number_of_reviews_from_purchased_people.str.split(")").str.get(0).str.split("of ").str.get(1)



    df=df.assign(
        genres=df.genres.apply(lambda x: " ".join([y.lower().replace(" ","_").replace("'","").replace("'","") for y in ast.literal_eval(x)])),
        release_date=df.release_date.str.split(" ").str.get(-1),
        number_of_reviews_from_purchased_people=pd.to_numeric(df.number_of_reviews_from_purchased_people.str.replace(r"[^0-9]","",regex=True))
    )
    return df.rename(columns={"name":"game_name","release_date":"release_year"})
def add_features(df:pd.DataFrame):
    df=df.copy()
    return (
    df.merge(df.groupby(["game_id"])["hours_played"].mean(), on="game_id").rename(columns={"hours_played_y":"game_avg_hrs_played"})
    .merge(df.groupby(["user_id"])["hours_played"].mean(), on="user_id").rename(columns={"hours_played":"user_avg_hrs_played"}).drop(columns=["hours_played_x"])
    )

def save(file_name,df):
    if os.getcwd().split('/')[-1] == 'src':
        path=os.path.join("..","data",f"{file_name}.csv")
    else:
        path=os.path.join("data",f"{file_name}.csv")

    df.to_csv(path,index=False)
    return "Done!"



if __name__ == "__main__":
    games_description=read_data("games_description.csv")
    steam_game_reviews=read_data("steam_game_reviews.csv",low_memory=False)
    steam_game_reviews.drop_duplicates(inplace=True)
    games_description.loc[games_description[games_description.short_description.isnull()].index,"short_description"]=[
        "Cyberpunk 2077 contains strong language, intense violence, blood and gore, as well as nudity and sexual material",
        "This DLC may contain content not appropriate for all ages, or may not be appropriate for viewing at work: Frequent Violence or Gore, General Mature Content ",
        "Shadow of the Erdtree is an expansion to ELDEN RING, the 2022 Game of the Year. Dark and intense, Shadow of the Erdtree has players continue their quest with the freedom to explore and experience the adventure at their own pace.",
        "Gameplay consists of frequent combat scenarios with characters using their weapons to slash/stab enemies.",
        "Located in the south-central region of the United States, the state is a nature wonderland with three national forests, many State Parks, nearly 9,000 miles of pristine streams and rivers like the Arkansas and Mississippi, as well as the ever-stretching Ozark and Ouachita mountain ranges.",
        "Life is a journey of chivalric adventure, so why don’t you embrace yours, ruler? Create your very own epic story with the major expansion, Crusader Kings III: Tours & Tournaments. Be awed by the sights and events that await you, from bold, mock combat in the jousting area to adventuring perilous wilds with your royal entourage.",
        "The Shadows of Change pack introduces 3 new Legendary Lords for Tzeentch, Grand Cathay, and Kislev, usable in both the Realm of Chaos and Immortal Empires campaigns.",
        "",
        "The NBA 2K25 Hall of Fame Pass: Season 1 takes your MyCAREER and MyTEAM experience to the next level. The Season 1 Hall of Fame Pass includes everything in the Pro Season Pass (access to 40 levels of earnable premium rewards, and 4 additional Season Pass Rewards) a 15% XP Booster for the entire season, and 10 Level Skips immediately applied to the Season reward track, & 15,000 VC!",
        "The most anticipated moment for Assetto Corsa Competizione has finally come. Nürburgring 24 hours is in. Nordschleife is probably the most iconic circuit worldwide. Constantly attracting thousands of professional and weekend racers from across the globe, this is the place where driver aim to drive in their career. Surely being the most challenging road course to ever have existed, it consists of 25.3 km of tarmac with over 70 bends that challenge even the most experienced racers.",
        "",
        "Experience thrilling hockey action in VR! The new Hockey DLC for All In One Sports lets you enjoy the fun of air hockey in a completely new way.",
        "Get ready for the ultimate soccer experience in VR! With the new Soccer DLC for All In One Sports, you can now enjoy soccer like never before."
    ]
    games_description=games_description_clean(games_description)
    steam_game_reviews=steam_game_reviews_clean(steam_game_reviews)
    # merge
    final_data=games_description.merge(steam_game_reviews,on="game_name")
    label_encoder1 = LabelEncoder()
    final_data["user_id"]=label_encoder1.fit_transform(final_data["username"])
    label_encoder2 = LabelEncoder()
    final_data["game_id"]=label_encoder2.fit_transform(final_data["game_name"])
    final_data= add_features(final_data)
    print(final_data.columns)

    # spliting
    np.random.seed(42)
    repeated_user_index=np.array(final_data[final_data["user_id"].duplicated()].index)
    np.random.shuffle(repeated_user_index)
    val=final_data.iloc[repeated_user_index[:100000],:]
    test=final_data.iloc[repeated_user_index[100000:200000],:]
    train1=final_data.iloc[repeated_user_index[200000:],:]
    train2=final_data[~final_data["user_id"].duplicated()]
    train=pd.concat([train1,train2],ignore_index=True)
    del train1
    del train2
    save("train",train)
    save("val",val)
    save("test",test)



