from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from nba_api.stats.static import players
from nba_api.stats.endpoints import scoreboardv2, commonteamroster, leaguedashteamstats, leaguegamefinder
import pandas as pd
import requests
import signal
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from prediction_engine import generate_predictions

app = Flask(__name__)
CORS(app)

def signal_handler(sig, frame):
    print('\nTerminating backend processes...')
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except (AttributeError, ProcessLookupError):
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# CSV path and in-memory cache so we never read from disk after first load/sync
SEASON_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'season_2025_26_games.csv')
SEASON_ID = '2025-26'
_SEASON_DF_CACHE = None

def _get_season_df():
    """Return season DataFrame from memory cache, or load from CSV once and cache. Never reads disk after first load/sync."""
    global _SEASON_DF_CACHE
    if _SEASON_DF_CACHE is not None:
        return _SEASON_DF_CACHE
    if not os.path.isfile(SEASON_CSV):
        return None
    try:
        _SEASON_DF_CACHE = pd.read_csv(SEASON_CSV)
        return _SEASON_DF_CACHE
    except Exception:
        return None

def _set_season_df_cache(df):
    """Update in-memory cache (call right after sync so data is hot for reports)."""
    global _SEASON_DF_CACHE
    _SEASON_DF_CACHE = df

def _load_season_csv():
    """Load season data: use in-memory cache when available, else load from CSV and cache."""
    df = _get_season_df()
    if df is not None:
        # Normalize GAME_ID format for consistency
        def normalize_game_id(game_id):
            game_id_str = str(game_id)
            if len(game_id_str) == 8:
                return '00' + game_id_str
            return game_id_str
        df['GAME_ID'] = df['GAME_ID'].apply(normalize_game_id)
    return df

def _unique_games_count(df):
    """Return number of unique games in a season log DataFrame."""
    if df is None or df.empty or 'GAME_ID' not in df.columns:
        return 0
    return int(df['GAME_ID'].nunique())

@app.route('/api/season-games-count', methods=['GET'])
def season_games_count():
    """Return how many games we have saved in the season CSV (for frontend display)."""
    try:
        df = _load_season_csv()
        count = _unique_games_count(df)
        total_rows = len(df) if df is not None else 0
        print(f"DEBUG: Games count endpoint - Unique games: {count}, Total rows: {total_rows}")
        if df is not None and len(df) > 0:
            print(f"DEBUG: Latest GAME_IDs sample: {df['GAME_ID'].tail(3).tolist()}")
        return jsonify({
            "games_saved": count,
            "debug_total_rows": total_rows,
            "debug_sample_games": df['GAME_ID'].head(5).tolist() if df is not None and not df.empty else []
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/sync-season-games', methods=['POST'])
def sync_season_games():
    """
    Fetch all 2025-26 NBA season games with player stats from NBA API.
    Merge into CSV: add new rows, update existing (GAME_ID + PLAYER_ID), no duplicates.
    Returns count of unique games, not player-game rows.
    """
    try:
        print("Fetching 2025-26 season from NBA API (LeagueGameFinder)...")
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable='2025-26',
            season_type_nullable='Regular Season',
            league_id_nullable='00',
            player_or_team_abbreviation='P'
        )
        new_df = finder.get_data_frames()[0]
        if new_df.empty:
            return jsonify({"error": "No data returned from API", "games_saved": 0}), 400

        # Normalize GAME_ID format to ensure consistent comparison
        # NBA API returns 10-digit IDs with '00' prefix, but existing data might not have it
        def normalize_game_id(game_id):
            game_id_str = str(game_id)
            if len(game_id_str) == 8:
                return '00' + game_id_str
            return game_id_str
            
        new_df['GAME_ID'] = new_df['GAME_ID'].apply(normalize_game_id)
        
        existing = _load_season_csv()
        print(f"DEBUG: Existing data - {len(existing) if existing is not None else 0} rows, {existing['GAME_ID'].nunique() if existing is not None else 0} unique games")
        print(f"DEBUG: New data from API - {len(new_df)} rows, {new_df['GAME_ID'].nunique()} unique games")
        
        if existing is not None and not existing.empty:
            # Use GAME_ID + PLAYER_ID as the unique key for merging
            existing_key = existing['GAME_ID'].astype(str) + '_' + existing['PLAYER_ID'].astype(str)
            new_key = new_df['GAME_ID'].astype(str) + '_' + new_df['PLAYER_ID'].astype(str)
            
            existing_keys_set = set(existing_key.tolist())
            new_keys_set = set(new_key.tolist())
            
            print(f"DEBUG: Existing unique keys: {len(existing_keys_set)}")
            print(f"DEBUG: New unique keys: {len(new_keys_set)}")
            print(f"DEBUG: Keys in both: {len(existing_keys_set.intersection(new_keys_set))}")
            
            # Only add truly new records (keys that don't exist)
            truly_new_mask = ~new_key.isin(existing_keys_set)
            new_records = new_df[truly_new_mask].copy()
            
            print(f"DEBUG: Truly new records: {len(new_records)} rows")
            
            if len(new_records) > 0:
                # Concatenate existing with new records
                combined = pd.concat([existing, new_records], ignore_index=True)
            else:
                # No new records, keep existing data
                combined = existing.copy()
                
            print(f"DEBUG: Combined result: {len(combined)} rows, {combined['GAME_ID'].nunique()} unique games")
        else:
            combined = new_df.copy()
            print(f"DEBUG: No existing data, using new data directly")
        
        # CRITICAL: Deduplicate the combined data to prevent inflation
        print(f"DEBUG: Before deduplication - {len(combined)} rows, {combined['GAME_ID'].nunique()} unique games")
        combined = combined.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID'])
        print(f"DEBUG: After deduplication - {len(combined)} rows, {combined['GAME_ID'].nunique()} unique games")
        
        # Ensure we're working with clean data
        unique_games_final = combined['GAME_ID'].nunique()
        total_rows_final = len(combined)
        print(f"DEBUG: Final clean data - {unique_games_final} unique games, {total_rows_final} rows")

        combined.to_csv(SEASON_CSV, index=False)
        # CLEAR the cache to force reload on next request
        global _SEASON_DF_CACHE
        _SEASON_DF_CACHE = None
        print("Season CSV cache cleared - will reload on next request")
        print(f"DEBUG: Data saved to CSV - rows: {len(combined)}, unique games: {combined['GAME_ID'].nunique()}")
        
        # Count unique games (not player-game rows)
        unique_games = combined['GAME_ID'].nunique()
        total_rows = len(combined)
        
        print(f"DEBUG: Sync endpoint - Unique games: {unique_games}, Total rows: {total_rows}")
        print(f"DEBUG: Sample GAME_IDs: {combined['GAME_ID'].head(10).tolist()}")
        
        # Verify no duplicates in GAME_ID column
        duplicate_check = combined.groupby('GAME_ID').size()
        max_players_per_game = duplicate_check.max()
        min_players_per_game = duplicate_check.min()
        print(f"DEBUG: Players per game - Min: {min_players_per_game}, Max: {max_players_per_game}")
        
        print(f"Season CSV updated: {unique_games_final} unique games, {total_rows_final} player-game rows.")
        return jsonify({
            "games_saved": unique_games_final,  # This is the unique game count
            "player_game_rows": total_rows_final,  # This is total rows
            "message": f"Season data synced: {unique_games_final} unique games, {total_rows_final} player records."
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

MILESTONES = {
    'PTS': [10, 15, 20, 22, 24, 25, 28, 30, 35, 40], 
    "REB": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "AST": [2, 3, 4, 5, 6, 7, 8], 
    "STL": [1, 2], 
    "BLK": [1, 2],
    "PRA": [20, 25, 30, 35, 40], 
    "3PM": [1, 2, 3, 4]
}

DEFENSE_STAT_MAP = {
    'PTS': 'OPP_PTS',
    'REB': 'OPP_REB',
    'AST': 'OPP_AST',
    'STL': 'OPP_STL',
    'BLK': 'OPP_BLK',
    '3PM': 'OPP_FG3M',
    'PRA': 'OPP_PTS'
}

def _player_on_roster(player_name, roster_names):
    """True if player_name matches any name in roster (exact, normalized, or substring)."""
    if not roster_names or not player_name:
        return False
    p = str(player_name).strip()
    for r in roster_names:
        r = str(r).strip()
        if p == r:
            return True
        if p.lower() == r.lower():
            return True
        if p in r or r in p:
            return True
    return False

def get_matchup_multipliers(opponent_team_id, opp_stats):
    """
    Standard matchup multiplier based only on pre-fetched opponent defensive rankings.
    """
    try:
        if opp_stats is None or opp_stats.empty: return {}

        multipliers = {}
        for cat, col in DEFENSE_STAT_MAP.items():
            if col not in opp_stats.columns: continue
            league_avg = opp_stats[col].mean()
            if league_avg == 0: continue
            team_row = opp_stats[opp_stats['TEAM_ID'] == opponent_team_id]
            if team_row.empty: continue
            team_val = float(team_row[col].iloc[0])
            multipliers[cat] = round(team_val / league_avg, 3)

        if 'PTS' in multipliers and 'REB' in multipliers and 'AST' in multipliers:
            multipliers['PRA'] = round((multipliers['PTS'] * 0.5 + multipliers['REB'] * 0.25 + multipliers['AST'] * 0.25), 3)
        return multipliers
    except Exception as e:
        print(f"Matchup error: {e}")
        return {}

# In-memory cache for games endpoint (cache for 60 seconds)
_GAMES_CACHE = {"data": None, "timestamp": 0}
_GAMES_CACHE_TTL = 60  # seconds

# Thread pool for parallel game fetching
_fetch_executor = ThreadPoolExecutor(max_workers=10)

def _fetch_games_for_date(d):
    """Fetch games for a single date - used for parallel execution"""
    from nba_api.stats.endpoints import scoreboardv2
    formatted_date = d.strftime('%Y-%m-%d')
    games_for_date = []
    try:
        frames = scoreboardv2.ScoreboardV2(game_date=formatted_date).get_data_frames()
        board = frames[0] if frames and len(frames) > 0 else None
        if board is not None and not board.empty:
            for _, row in board.iterrows():
                game_id = row.get('GAME_ID')
                gamecode = row.get('GAMECODE')
                if gamecode:
                    code_str = str(gamecode)
                    matchup = f"{code_str[8:]} @ {code_str[:3]}" if len(code_str) >= 11 else code_str
                else:
                    visitor_abbr = row.get('VISITOR_TEAM_ABBREVIATION') or row.get('VISITOR_TEAM_ABBREV')
                    home_abbr = row.get('HOME_TEAM_ABBREVIATION') or row.get('HOME_TEAM_ABBREV')
                    matchup = f"{visitor_abbr or ''} @ {home_abbr or ''}".strip()
                games_for_date.append({
                    'game_id': game_id,
                    'visitor_id': row.get('VISITOR_TEAM_ID'),
                    'home_id': row.get('HOME_TEAM_ID'),
                    'matchup': matchup,
                    'date': formatted_date
                })
            return games_for_date
        # Try CDN fallback
        cdn_date = d.strftime('%Y%m%d')
        url = f"https://cdn.nba.com/static/json/liveData/scoreboard/scoreboard_{cdn_date}.json"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}, timeout=10)
        if resp.ok:
            data = resp.json()
            for g in data.get('scoreboard', {}).get('games', []):
                away = g.get('awayTeam', {})
                home = g.get('homeTeam', {})
                matchup = f"{away.get('teamTricode', '')} @ {home.get('teamTricode', '')}"
                games_for_date.append({
                    'game_id': g.get('gameId'),
                    'visitor_id': away.get('teamId'),
                    'home_id': home.get('teamId'),
                    'matchup': matchup,
                    'date': formatted_date
                })
    except Exception:
        pass
    return games_for_date

@app.route('/api/games', methods=['GET'])
def get_games():
    try:
        # Check cache first
        now = datetime.now().timestamp()
        if _GAMES_CACHE["data"] and (now - _GAMES_CACHE["timestamp"]) < _GAMES_CACHE_TTL:
            return jsonify(_GAMES_CACHE["data"])
        
        base_now_nz = datetime.now(ZoneInfo("Pacific/Auckland")) if ZoneInfo else datetime.now()
        start_date = (base_now_nz - timedelta(days=1)).date()
        nz_dates = [start_date + timedelta(days=i) for i in range(0, 10)]
        if ZoneInfo:
            base_now_us = datetime.now(ZoneInfo("America/New_York"))
            us_start_date = (base_now_us - timedelta(days=1)).date()
            us_dates = [us_start_date + timedelta(days=i) for i in range(0, 10)]
            us_dates = [d for d in us_dates if d >= start_date]
        else:
            us_dates = []
        dates = sorted(set(nz_dates + us_dates))
        
        # Parallel fetch all dates at once - MUCH faster!
        results = list(_fetch_executor.map(_fetch_games_for_date, dates))
        
        # Collect all games
        games_list = []
        seen_game_ids = set()
        import math
        for games_for_date in results:
            for g in games_for_date:
                # Handle NaN values properly
                vid = g['visitor_id']
                hid = g['home_id']
                if vid is None or (isinstance(vid, float) and math.isnan(vid)):
                    continue
                if hid is None or (isinstance(hid, float) and math.isnan(hid)):
                    continue
                if not vid or not hid:
                    continue
                if g['game_id'] and g['game_id'] in seen_game_ids:
                    continue
                if g['game_id']:
                    seen_game_ids.add(g['game_id'])
                games_list.append({
                    "label": f"{g['date']} | {g['matchup']}",
                    "visitor_id": int(vid),
                    "home_id": int(hid)
                })
        
        end_date = (start_date + timedelta(days=9)).strftime('%Y-%m-%d')
        response_data = {"games": games_list, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date}
        
        # Cache the response
        _GAMES_CACHE["data"] = response_data
        _GAMES_CACHE["timestamp"] = datetime.now().timestamp()
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rosters', methods=['POST'])
def get_rosters():
    try:
        data = request.json
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(lambda: commonteamroster.CommonTeamRoster(team_id=data['visitor_id']).get_data_frames()[0]['PLAYER'].tolist())
            f2 = executor.submit(lambda: commonteamroster.CommonTeamRoster(team_id=data['home_id']).get_data_frames()[0]['PLAYER'].tolist())
            r1 = f1.result()
            r2 = f2.result()
        return jsonify({"players": sorted(list(set(r1 + r2)))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/player-stats', methods=['POST'])
def get_player_stats():
    try:
        data = request.json
        player_name = data['player_name']
        game_count = data.get('game_count', 10)
        v_id = data.get('visitor_id', None)
        h_id = data.get('home_id', None)
        
        search = players.find_players_by_full_name(player_name)
        if not search: return jsonify({"error": "Player not found"}), 404
        p_id = int(search[0]['id'])

        # Fast path: get player games from in-memory cache (no disk I/O)
        csv_df = _get_season_df()
        if csv_df is None or csv_df.empty or 'PLAYER_ID' not in csv_df.columns:
            return jsonify({"error": "Season cache empty. Use SYNC_2025-26_SEASON to download game data."}), 404
        player_mask = csv_df['PLAYER_ID'].astype(int) == p_id
        if not player_mask.any():
            return jsonify({"error": f"No 2025-26 game data for {player_name} in cache. Sync season or check spelling."}), 404
        df = csv_df.loc[player_mask].copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Historical games: home/away from CSV MATCHUP
        # "TEAM_A vs. TEAM_B" = A home, B away. "TEAM_A @ TEAM_B" = A away, B home. Player's team can be first or second.
        vs_mask = df['MATCHUP'].str.contains(' vs. ', na=False)
        at_mask = df['MATCHUP'].str.contains(' @ ', na=False)
        home_team_abbr = pd.Series(index=df.index, dtype=object)
        home_team_abbr[vs_mask] = df.loc[vs_mask, 'MATCHUP'].str.split(' vs. ', expand=True)[0].str.strip()
        home_team_abbr[at_mask] = df.loc[at_mask, 'MATCHUP'].str.split(' @ ', expand=True)[1].str.strip()
        df['IS_HOME'] = (df['TEAM_ABBREVIATION'].astype(str).str.strip() == home_team_abbr)
        
        # Debug: Print sample of home/away detection
        if len(df) > 0:
            sample_rows = df.head(3)
            for idx, row in sample_rows.iterrows():
                print(f"DEBUG: Player {row['PLAYER_NAME']} - Matchup: '{row['MATCHUP']}' - Team: {row['TEAM_ABBREVIATION']} - Is Home: {row['IS_HOME']}")

        # Upcoming game home/away: nba_api only (visitor + home rosters)
        v_id_int = int(v_id) if v_id is not None else None
        h_id_int = int(h_id) if h_id is not None else None
        is_home_game = True
        opp_id = None
        opp_stats = None

        if v_id_int is not None and h_id_int is not None:
            visitor_roster = []
            home_roster = []
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    f_opp = executor.submit(lambda: leaguedashteamstats.LeagueDashTeamStats(
                        measure_type_detailed_defense='Opponent', per_mode_detailed='PerGame',
                        season='2025-26', season_type_all_star='Regular Season'
                    ).get_data_frames()[0])
                    f_v = executor.submit(lambda: commonteamroster.CommonTeamRoster(team_id=v_id_int).get_data_frames()[0]['PLAYER'].tolist())
                    f_h = executor.submit(lambda: commonteamroster.CommonTeamRoster(team_id=h_id_int).get_data_frames()[0]['PLAYER'].tolist())
                    opp_stats = f_opp.result()
                    visitor_roster = f_v.result()
                    home_roster = f_h.result()
            except Exception as e:
                print(f"Roster/opp stats error: {e}")

            # Which roster is the player on? (try request name and CSV name for matching)
            csv_name = df['PLAYER_NAME'].iloc[0] if 'PLAYER_NAME' in df.columns and len(df) > 0 else player_name
            on_visitor = _player_on_roster(player_name, visitor_roster) or _player_on_roster(csv_name, visitor_roster)
            on_home = _player_on_roster(player_name, home_roster) or _player_on_roster(csv_name, home_roster)
            
            print(f"DEBUG: Player '{player_name}' - CSV name: '{csv_name}'")
            print(f"DEBUG: On visitor roster: {on_visitor}, On home roster: {on_home}")
            
            if on_visitor:
                is_home_game = False
                opp_id = h_id_int
                print(f"DEBUG: Setting as AWAY game (on visitor team)")
            elif on_home:
                is_home_game = True
                opp_id = v_id_int
                print(f"DEBUG: Setting as HOME game (on home team)")
            else:
                print(f"DEBUG: Could not determine team assignment - defaulting to HOME")
        
        last_n = df.sort_values('GAME_DATE', ascending=False).head(game_count).iloc[::-1].copy()
        last_n['PRA'] = last_n['PTS'] + last_n['REB'] + last_n['AST']
        
        # Also keep full season data for home/away splits
        full_season_df = df.copy()
        full_season_df['PRA'] = full_season_df['PTS'] + full_season_df['REB'] + full_season_df['AST']
        
        season_avg_pts = round(df['PTS'].mean(), 1)
        season_avg_reb = round(df['REB'].mean(), 1)
        season_avg_ast = round(df['AST'].mean(), 1)
        season_fg_pct = round((df['FGM'].sum() / df['FGA'].sum() * 100), 1) if df['FGA'].sum() > 0 else 0
        
        # Calculate home/away splits from filtered games (last_n)
        home_games = last_n[last_n['IS_HOME'] == True]
        away_games = last_n[last_n['IS_HOME'] == False]
        
        home_away_splits = {
            'home': {
                'games': len(home_games),
                'pts': round(home_games['PTS'].mean(), 1) if len(home_games) > 0 else 0,
                'reb': round(home_games['REB'].mean(), 1) if len(home_games) > 0 else 0,
                'ast': round(home_games['AST'].mean(), 1) if len(home_games) > 0 else 0,
            },
            'away': {
                'games': len(away_games),
                'pts': round(away_games['PTS'].mean(), 1) if len(away_games) > 0 else 0,
                'reb': round(away_games['REB'].mean(), 1) if len(away_games) > 0 else 0,
                'ast': round(away_games['AST'].mean(), 1) if len(away_games) > 0 else 0,
            }
        }
        
        stats_data = {}
        for cat, targets in MILESTONES.items():
            col = 'FG3M' if cat == "3PM" else cat
            stats = last_n[col].tolist()
            hits = [[t, int((sum(1 for s in stats if s >= t)/len(stats))*100)] for t in targets]
            stats_data[cat] = {"hits": hits, "raw": stats}

        # --- HIGH-RES SCAN FOR SUGGESTED PROPS (90%+ MAX VALUE) ---
        suggestions = []
        SCAN_RANGES = {
            'PTS': range(10, 51), 'REB': range(2, 21), 'AST': range(2, 16),
            'STL': range(1, 5), 'BLK': range(1, 5), 'PRA': range(15, 65), '3PM': range(1, 10)
        }
        for cat, r in SCAN_RANGES.items():
            col = 'FG3M' if cat == "3PM" else cat
            stats = last_n[col].tolist()
            best = None
            for t in sorted(list(r), reverse=True):
                rate = int((sum(1 for s in stats if s >= t)/len(stats))*100)
                if rate >= 90:
                    best = {"cat": cat, "threshold": t, "rate": rate}
                    break
            if best: suggestions.append(best)
        stats_data['suggestions'] = sorted(suggestions, key=lambda x: x['rate'], reverse=True)
        # --- END SUGGESTIONS ---
        
        stats_data['meta'] = {
            'player_id': p_id,
            'player_name': player_name,
            'season_stats': {
                'pts': season_avg_pts, 'reb': season_avg_reb, 'ast': season_avg_ast, 'fg_pct': season_fg_pct
            },
            'home_away_splits': home_away_splits,
            'upcoming_game_is_home': is_home_game
        }

        matchup_multipliers = {}
        if opp_id is not None and opp_stats is not None:
            try:
                matchup_multipliers = get_matchup_multipliers(opp_id, opp_stats)
                stats_data['meta']['opponent_team_id'] = opp_id
                stats_data['meta']['matchup_multipliers'] = matchup_multipliers
            except Exception as e:
                print(f"Opponent detection error: {e}")

        # Build game context for prediction engine
        game_context = {
            'is_home_game': is_home_game,
            'opponent_team_id': stats_data['meta'].get('opponent_team_id'),
        }

        try:
            stats_data['predictions'] = generate_predictions(
                last_n, 
                MILESTONES, 
                matchup_multipliers,
                game_context=game_context,
                full_season_df=full_season_df
            )
            print(f"Predictions generated for {player_name}: {list(stats_data['predictions'].keys())}")
        except Exception as e:
            import traceback
            print(f"Prediction error for {player_name}: {e}")
            traceback.print_exc()
            stats_data['predictions'] = {}
        
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live-boxscore/<game_id>', methods=['GET'])
def get_live_boxscore(game_id):
    url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================================================================
# INJURY REPORTS
# =============================================================================

_INJURY_CACHE = {"data": None, "timestamp": 0}
_INJURY_CACHE_TTL = 300  # 5 minutes

@app.route('/api/injuries', methods=['GET'])
def get_injuries():
    """Fetch current NBA injury reports"""
    try:
        # Check cache
        now = datetime.now().timestamp()
        if _INJURY_CACHE["data"] and (now - _INJURY_CACHE["timestamp"]) < _INJURY_CACHE_TTL:
            return jsonify(_INJURY_CACHE["data"])
        
        # Fetch from NBA injury endpoint
        url = "https://api-web.nba.com/injuries"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
        
        resp = requests.get(url, headers=headers, timeout=10)
        if not resp.ok:
            return jsonify({"error": "Failed to fetch injuries", "status": resp.status_code}), 500
        
        injury_data = resp.json()
        
        # Process and simplify
        injuries_list = []
        for league in injury_data.get('League', {}).get('Players', []):
            for player in league.get('Players', []):
                injuries_list.append({
                    'player_id': player.get('playerId'),
                    'player_name': player.get('playerName'),
                    'team_id': player.get('teamId'),
                    'team_name': player.get('teamName'),
                    'status': player.get('status'),
                    'description': player.get('description'),
                    'date_updated': player.get('dateUpdated')
                })
        
        response_data = {
            "injuries": injuries_list,
            "count": len(injuries_list),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache
        _INJURY_CACHE["data"] = response_data
        _INJURY_CACHE["timestamp"] = now
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/team-injuries/<team_id>', methods=['GET'])
def get_team_injuries(team_id):
    """Get injuries for a specific team"""
    try:
        resp = requests.get("https://api-web.nba.com/injuries", 
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if not resp.ok:
            return jsonify({"error": "Failed to fetch"}), 500
        
        injury_data = resp.json()
        team_injuries = []
        
        for league in injury_data.get('League', {}).get('Players', []):
            for player in league.get('Players', []):
                if str(player.get('teamId')) == str(team_id):
                    team_injuries.append({
                        'player_id': player.get('playerId'),
                        'player_name': player.get('playerName'),
                        'status': player.get('status'),
                        'description': player.get('description')
                    })
        
        return jsonify({"team_id": team_id, "injuries": team_injuries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================================================================
# PREDICTION JOURNAL - Track predictions and results
# =============================================================================

_PREDICTION_JOURNAL = []

@app.route('/api/journal/add', methods=['POST'])
def add_prediction():
    """Record a prediction for tracking"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        record = {
            "id": len(_PREDICTION_JOURNAL) + 1,
            "timestamp": datetime.now().isoformat(),
            "player_id": data.get("player_id"),
            "player_name": data.get("player_name"),
            "category": data.get("category"),
            "threshold": data.get("threshold"),
            "expected": data.get("expected"),
            "over_prob": data.get("over_prob"),
            "actual": None,  # To be filled later
            "result": None,  # 'win' or 'loss'
            "game_id": data.get("game_id"),
            "notes": data.get("notes", "")
        }
        _PREDICTION_JOURNAL.append(record)
        return jsonify({"status": "recorded", "id": record["id"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/journal/result', methods=['POST'])
def update_prediction_result():
    """Update prediction with actual result"""
    try:
        data = request.get_json()
        pred_id = data.get("id")
        actual = data.get("actual")
        
        for pred in _PREDICTION_JOURNAL:
            if pred["id"] == pred_id:
                pred["actual"] = actual
                threshold = pred["threshold"]
                over_prob = pred.get("over_prob", 0.5)
                
                # Determine win/loss
                if actual is not None:
                    if actual > threshold:
                        pred["result"] = "win"
                    else:
                        pred["result"] = "loss"
                
                return jsonify({"status": "updated", "prediction": pred})
        
        return jsonify({"error": "Prediction not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/journal', methods=['GET'])
def get_journal():
    """Get all predictions in journal"""
    try:
        return jsonify({
            "predictions": _PREDICTION_JOURNAL,
            "total": len(_PREDICTION_JOURNAL),
            "wins": sum(1 for p in _PREDICTION_JOURNAL if p.get("result") == "win"),
            "losses": sum(1 for p in _PREDICTION_JOURNAL if p.get("result") == "loss")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/journal/stats', methods=['GET'])
def get_journal_stats():
    """Get journal statistics"""
    try:
        completed = [p for p in _PREDICTION_JOURNAL if p.get("result")]
        total = len(completed)
        if total == 0:
            return jsonify({"accuracy": 0, "total": 0, "wins": 0, "losses": 0})
        
        wins = sum(1 for p in completed if p["result"] == "win")
        losses = total - wins
        
        # Per-category stats
        cat_stats = {}
        for p in completed:
            cat = p.get("category", "unknown")
            if cat not in cat_stats:
                cat_stats[cat] = {"wins": 0, "total": 0}
            cat_stats[cat]["total"] += 1
            if p["result"] == "win":
                cat_stats[cat]["wins"] += 1
        
        # Calculate accuracy per category
        for cat in cat_stats:
            cat_stats[cat]["accuracy"] = round(cat_stats[cat]["wins"] / cat_stats[cat]["total"] * 100, 1)
        
        return jsonify({
            "accuracy": round(wins / total * 100, 1),
            "total": total,
            "wins": wins,
            "losses": losses,
            "by_category": cat_stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/journal/export', methods=['GET'])
def export_journal_csv():
    """Export journal as CSV"""
    try:
        import csv
        from io import StringIO
        
        output = StringIO()
        if not _PREDICTION_JOURNAL:
            return jsonify({"error": "No predictions to export"}), 404
        
        fieldnames = ["id", "timestamp", "player_name", "category", "threshold", "expected", "over_prob", "actual", "result", "game_id"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in _PREDICTION_JOURNAL:
            row = {k: pred.get(k, "") for k in fieldnames}
            writer.writerow(row)
        
        output.seek(0)
        return output.getvalue(), 200, {"Content-Type": "text/csv"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Warm in-memory cache at startup so first report is fast
if os.path.isfile(SEASON_CSV):
    try:
        df = _get_season_df()
        if df is not None and not df.empty:
            print(f"Season cache loaded: {_unique_games_count(df)} games in memory.")
    except Exception:
        pass

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
