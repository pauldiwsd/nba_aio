import pandas as pd

df = pd.read_csv('season_2025_26_games.csv')
print(f'Clean data: {len(df)} rows, {df["GAME_ID"].nunique()} unique games')
print('Sample rows:')
print(df.head())