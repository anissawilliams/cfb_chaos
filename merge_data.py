import requests
import pandas as pd
from tqdm import tqdm  # Optional: for progress bar
#import cfbd

import certifi
import ssl
import urllib3
CFBD_API_KEY = "mvtFD9Rg26SQFMDNkQL30s6nmfpDp8Vg7jgQRo0ec0frsxF/qMkbQ6vSpAWI/XlU"

ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



response = requests.get("https://api.collegefootballdata.com/games?id=401012257", headers={
    "Authorization": f"Bearer {CFBD_API_KEY}"
}, verify=certifi.where())


def fetch_game_metadata(game_id):
    url = f"https://api.collegefootballdata.com/games?id={game_id}"
    headers = {
        "Authorization": f"Bearer {CFBD_API_KEY}"
    }
    try:
        response = requests.get(url, headers=headers, verify=certifi.where())
        data = response.json()
        if data and isinstance(data, list):
            game = data[0]
            return {
                "game_id": game["id"],
                "start_date": game["startDate"],
                "week": game["week"],
                "home_team": game["homeTeam"],
                "away_team": game["awayTeam"],
                "home_conference": game["homeConference"],
                "away_conference": game["awayConference"],
                "excitement_index": game["excitementIndex"],
                "home_pregame_elo": game["homePregameElo"],
                "away_pregame_elo": game["awayPregameElo"],
                "attendance": game["attendance"],
                "venue": game["venue"]
            }
    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
        return None



def merge_metadata(df):
    metadata = []
    for game_id in tqdm(df["game_id"]):
        meta = fetch_game_metadata(game_id)
        if meta:
            metadata.append(meta)
    meta_df = pd.DataFrame(metadata)
    merged_df = df.merge(meta_df, on="game_id", how="left")
    return merged_df

original_data = pd.read_csv("game_features.csv")
updated_data = merge_metadata(original_data)
updated_data.to_csv("chaos_data.csv")


#print(temp)
