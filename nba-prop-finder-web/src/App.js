import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = process.env.REACT_APP_API_URL || '';
const API = API_BASE ? `${API_BASE}/api` : '/api';
console.log('API URL:', API);

// Animation Variants
const tabVariants = {
  initial: { opacity: 0, y: 10, filter: 'blur(10px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
  exit: { opacity: 0, y: -10, filter: 'blur(10px)' },
};

export default function MainApp() {
  const [activeTab, setActiveTab] = useState('finder');

  // --- STATE FOR LIVE TRACKER ---
  const [parlays, setParlays] = useState(() => {
    const saved = localStorage.getItem('uav_terminal_parlays');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse parlays", e);
      }
    }
    return [
      { id: Date.now(), gameId: '', input: '', isLocked: false, liveData: null, parsedProps: [] }
    ];
  });

  // --- STATE FOR PROP FINDER (LIFTED) ---
  const [finderState, setFinderState] = useState({
    games: [],
    selectedGame: null,
    players: [],
    selectedPlayer: '',
    statsData: null,
    loading: false,
    activeThresholds: {},
    gameCount: 10,
    reportGameCount: 10,
    gamesSaved: null,
    syncLoading: false,
    syncError: null,
    gamesError: null
  });

  useEffect(() => {
    localStorage.setItem('uav_terminal_parlays', JSON.stringify(parlays));
  }, [parlays]);

  const addParlay = () => {
    const newParlay = { id: Date.now(), gameId: '', input: '', isLocked: false, liveData: null, parsedProps: [] };
    setParlays(prev => [...prev, newParlay]);
  };

  const updateParlay = (id, updates) => {
    console.log(`Updating Parlay ${id}:`, updates);
    setParlays(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p));
  };

  const removeParlay = (id) => {
    setParlays(prev => {
      if (prev.length > 1) {
        return prev.filter(p => p.id !== id);
      }
      return prev.map(p => p.id === id ? { ...p, gameId: '', input: '', isLocked: false, liveData: null, parsedProps: [] } : p);
    });
  };

  return (
    <div className="main-bg">

      <nav className="glass-nav">
        <div className="nav-container">
          {/* UPDATED LOGO SECTION */}
          <div className="logo-wrapper">
            <img
              src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGhxZHA0ZTVkMXd5eWVzemV0eTMwdzJmZTg4cXAxbjQzMzd2ZWMwciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/e4muODbpQOmS7j3tV8/giphy.gif"
              alt="UAV Logo"
              className="nav-logo-gif"
            />
            <div className="logo">UAV<span className="logo-accent">TERMINAL</span></div>
            <div className="logo-feeds">
              <div
                className="feed-box"
                onMouseEnter={(e) => {
                  const vSmall = e.currentTarget.querySelector('.feed-video');
                  const vBig = e.currentTarget.querySelector('.feed-expanded');
                  if (vBig && vSmall) {
                    vBig.currentTime = vSmall.currentTime;
                    vBig.muted = false;
                    vBig.play().catch(() => { });
                  }
                }}
                onMouseLeave={(e) => {
                  const v = e.currentTarget.querySelector('.feed-expanded');
                  if (v) v.muted = true;
                }}
              >
                <video src="/uav_vid_1.mp4" autoPlay loop muted playsInline className="feed-video feed-crop-high" />
                <video src="/uav_vid_1.mp4" autoPlay loop muted playsInline className="feed-expanded feed-crop-high" aria-hidden="true" />
              </div>
              <div
                className="feed-box"
                onMouseEnter={(e) => {
                  const vSmall = e.currentTarget.querySelector('.feed-video');
                  const vBig = e.currentTarget.querySelector('.feed-expanded');
                  if (vBig && vSmall) {
                    vBig.currentTime = vSmall.currentTime;
                    vBig.muted = false;
                    vBig.play().catch(() => { });
                  }
                }}
                onMouseLeave={(e) => {
                  const v = e.currentTarget.querySelector('.feed-expanded');
                  if (v) v.muted = true;
                }}
              >
                <video src="/uav_vid_2.mp4" autoPlay loop muted playsInline className="feed-video feed-crop-high" />
                <video src="/uav_vid_2.mp4" autoPlay loop muted playsInline className="feed-expanded feed-crop-high" aria-hidden="true" />
              </div>
            </div>
          </div>

          <div className="nav-links">
            <button onClick={() => setActiveTab('finder')} className={`nav-btn ${activeTab === 'finder' ? 'active' : ''}`}>PROP FINDER</button>
            <button onClick={() => setActiveTab('parlay')} className={`nav-btn ${activeTab === 'parlay' ? 'active' : ''}`}>LIVE TRACKER</button>
          </div>
        </div>
      </nav>

      <main className="content">
        <AnimatePresence mode="wait">
          {activeTab === 'finder' ? (
            <motion.div
              key="finder"
              variants={tabVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              {/* Pass state and setter to component */}
              <PropFinderV3 state={finderState} setState={setFinderState} />
            </motion.div>
          ) : (
            <motion.div
              key="tracker"
              variants={tabVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="tracker-layout"
            >
              <header className="tracker-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <h1 style={{ margin: 0, fontSize: '24px' }}>SYSTEM TRACKING</h1>
                </div>
                <button onClick={addParlay} className="add-btn">
                  [ + INITIALIZE NEW PROBE ]
                </button>
              </header>

              <div className="tracker-grid">
                {parlays.map((parlay) => (
                  <ParlayTracker
                    key={parlay.id}
                    data={parlay}
                    onUpdate={(updates) => updateParlay(parlay.id, updates)}
                    onRemove={() => removeParlay(parlay.id)}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <style jsx global>{`
        :root {
          --cyan: #a0a0a0;  /* Cool metallic silver */
          --pink: #c0c0c0;  /* Platinum accent */
          --glass: rgba(40, 40, 45, 0.85);  /* Brushed metal gray */
          --glass-heavy: rgba(30, 30, 35, 0.95);  /* Deeper metal base */
          --border: rgba(160, 160, 160, 0.25);  /* Metallic border */
          --bg: #0a0a0c;  /* Deep charcoal base */
          --success: #6bc9a7;  /* Metallic teal-green */
          --danger: #ff6b6b;  /* Soft metallic red */
        }

        html, body {
          color-scheme: dark !important;
          background: var(--bg);
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        * { 
          box-sizing: border-box; 
          user-select: none; 
        }

        /* Global smooth transitions */
        a, button, select, input {
          transition: all 0.2s ease;
        }

        /* Performance optimizations for animated elements */
        .glass-card, .suggested-panel, .stat-category, .pred-row, .sug-row, .hit-chip {
          will-change: transform, box-shadow;
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        
        body { 
          margin: 0; 
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
          color: white; 
          overflow-x: hidden;
          /* Smooth scrolling optimization */
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          /* Fixed background to prevent repaints during scroll */
          background: #0a0a0c;
          background-attachment: fixed;
          /* GPU acceleration */
          transform: translateZ(0);
          will-change: scroll-position;
        }

        /* Main container with GPU acceleration */
        .main-bg { 
          min-height: 100vh; 
          position: relative; 
          transform: translateZ(0);
          backface-visibility: hidden;
        }

        /* Content area optimizations */
        .content {
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        .glow-orb { position: fixed; width: 500px; height: 500px; filter: blur(120px); border-radius: 50%; z-index: 0; pointer-events: none; opacity: 0.3; }
        
        .glass-nav { 
          position: sticky; 
          top: 0; 
          z-index: 100; 
          backdrop-filter: blur(20px) saturate(120%);
          -webkit-backdrop-filter: blur(20px) saturate(120%);
          background: var(--glass-heavy);
          border-bottom: 1px solid var(--border);
          box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          background-image: 
            linear-gradient(90deg, transparent 0%, rgba(160, 160, 160, 0.08) 50%, transparent 100%);
        }
        .nav-container { max-width: 98%; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; padding: 15px 40px; }
        
        /* LOGO WRAPPER STYLES */
        .logo-wrapper { display: flex; align-items: center; gap: 18px; }
        .nav-logo-gif { 
          height: 55px;
          width: 55px;
          border-radius: 6px; 
          mix-blend-mode: screen; 
          filter: drop-shadow(0 0 8px #c0c0c0);
          transition: transform 0.3s ease;
        }
        .nav-logo-gif:hover {
          transform: scale(1.1);
        }

        .logo { font-weight: 900; font-size: 24px; letter-spacing: 3px; font-family: monospace; }
        .logo-accent { color: #c0c0c0; text-shadow: 0 0 10px #c0c0c0; }
        
        .nav-links { display: flex; gap: 10px; }
        .nav-btn { 
          background: none; 
          border: none; 
          color: #888; 
          font-weight: 600; 
          cursor: pointer; 
          padding: 12px 24px; 
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); 
          font-size: 13px; 
          letter-spacing: 1.5px;
          border-radius: 8px;
          position: relative;
          overflow: hidden;
        }
        .nav-btn:hover { 
          color: #bbb; 
          background: rgba(255, 255, 255, 0.05);
          transform: translateY(-1px);
        }
        .nav-btn.active { 
          color: #c0c0c0; 
          background: rgba(255, 255, 255, 0.1);
          box-shadow: inset 0 -2px 0 #c0c0c0;
        }
        
        .content { 
          position: relative; 
          z-index: 1; 
          max-width: 98%; 
          margin: 0 auto; 
          padding: 25px 40px;
        }
        
        .terminal-input {
          width: 100%;
          background: rgba(40, 40, 45, 0.7) !important;
          border: 1px solid rgba(160, 160, 160, 0.3);
          border-radius: 10px;
          color: #e0e0e0 !important;
          padding: 12px 16px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          font-size: 14px;
          outline: none;
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          appearance: none;
          backdrop-filter: blur(10px);
          box-shadow: 
            inset 0 1px 2px rgba(0, 0, 0, 0.3),
            inset 0 -1px 0 rgba(255, 255, 255, 0.05);
        }

        .terminal-input option {
          background-color: #111 !important;
          color: #c0c0c0 !important;
        }

        .terminal-input:focus {
          border-color: rgba(255, 255, 255, 0.4);
          background: rgba(50, 50, 55, 0.8);
          box-shadow: 
            0 0 0 3px rgba(160, 160, 160, 0.2),
            inset 0 1px 2px rgba(0, 0, 0, 0.2),
            inset 0 -1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .terminal-input:hover:not(:disabled):not(:focus) {
          background: rgba(45, 45, 50, 0.8);
          border-color: rgba(160, 160, 160, 0.5);
        }
        
        select.terminal-input {
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2300f2ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
          background-repeat: no-repeat;
          background-position: right 12px center;
          background-size: 14px;
          cursor: pointer;
          padding-right: 40px;
        }

        .add-btn {
          background: transparent;
          color: #c0c0c0;
          border: 1px solid #c0c0c0;
          padding: 8px 16px;
          border-radius: 2px;
          font-weight: bold;
          font-family: 'Consolas', monospace;
          cursor: pointer;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          text-transform: uppercase;
          letter-spacing: 1px;
          font-size: 11px;
          box-shadow: inset 0 0 5px rgba(255, 255, 255, 0.2);
        }

        .engage-btn {
          background: linear-gradient(135deg, #a0a0a0, #808080);
          color: #000;
          border: none;
          padding: 14px 28px;
          border-radius: 12px;
          font-weight: 600;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          letter-spacing: 0.5px;
          cursor: pointer;
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          box-shadow: 
            0 4px 15px rgba(160, 160, 160, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          position: relative;
          overflow: hidden;
          text-shadow: 0 1px 1px rgba(255, 255, 255, 0.3);
        }
        .engage-btn::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
          transition: left 0.5s;
        }
        .engage-btn:hover:not(:disabled)::before {
          left: 100%;
        }
        .engage-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 
            0 8px 25px rgba(160, 160, 160, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.4),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
        }
        .engage-btn:disabled { 
          background: rgba(80, 80, 85, 0.6); 
          color: rgba(180, 180, 180, 0.6); 
          cursor: not-allowed; 
          box-shadow: none;
          transform: none;
          text-shadow: none;
        }

        .glass-card { 
          background: var(--glass); 
          backdrop-filter: blur(15px) saturate(120%);
          border: 1px solid var(--border); 
          border-radius: 16px; 
          padding: 25px; 
          position: relative;
          transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .glass-card:hover {
          transform: translateY(-4px);
          box-shadow: 
            0 12px 40px rgba(160, 160, 160, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.4);
        }
        
        .live-badge { font-size: 10px; background: #ff0000; color: white; padding: 2px 8px; border-radius: 4px; margin-left: 10px; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .tracker-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 30px; }

        /* LOGO FEED STYLES */
        .logo-feeds {
          display: flex;
          gap: 10px;
          margin-left: 20px;
          align-items: center;
        }
        .feed-box {
          position: relative;
          width: 98px;
          height: 55px;
          border-radius: 6px;
          border: 1px solid rgba(160, 160, 160, 0.3);
          box-shadow: 0 0 10px rgba(160, 160, 160, 0.15);
          cursor: pointer;
          z-index: 10;
          transition: border-color 0.3s ease;
        }
        .feed-box:hover {
          border-color: rgba(255, 255, 255, 0.4);
        }
        .feed-video {
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.95;
          border-radius: 4px;
        }
        .feed-crop-high {
          object-position: center 20%;
        }
        .feed-expanded {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
          pointer-events: none;
          opacity: 0;
          transform: scale(1);
          transform-origin: center top;
          transition: transform 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 0.1s ease;
          border-radius: 8px;
          box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.9),
            0 0 30px rgba(160, 160, 160, 0.5),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
          z-index: 1000;
        }
        .feed-box:hover .feed-expanded {
          opacity: 1;
          transform: scale(6) translateY(10px);
        }
      `}</style>
    </div>
  );
}

// --- COMPONENT: PARLAY TRACKER ---
function ParlayTracker({ data, onUpdate, onRemove }) {
  const { gameId, input, isLocked, liveData, parsedProps } = data;
  const [lastSync, setLastSync] = useState(null);

  useEffect(() => {
    if (!isLocked || !gameId) return;
    let isActive = true;
    let timeoutId;
    const update = () => {
      fetch(`${API}/live-boxscore/${gameId}`)
        .then(res => res.json())
        .then(resData => {
          if (isActive) {
            onUpdate({ liveData: resData.game });
            setLastSync(new Date().toLocaleTimeString());
            timeoutId = setTimeout(update, 12000);
          }
        })
        .catch(() => {
          if (isActive) timeoutId = setTimeout(update, 10000);
        });
    };
    update();
    return () => {
      isActive = false;
      clearTimeout(timeoutId);
    };
  }, [isLocked, gameId]);

  const parseInput = (text) => {
    return text.split('\n').filter(l => l.trim()).map(line => {
      const match = line.match(/^(.*?)\s+\(.*\)\s+\((.*)\)$/);
      if (!match) return null;
      const name = match[1].trim();
      const cond = match[2].toLowerCase();
      const target = parseInt(cond.match(/(\d+)/)?.[1] || 0);
      let type = "";
      if (cond.includes("three point") || cond.includes("3-pt")) type = "3pm";
      else if (cond.includes("rebounds")) type = "reb";
      else if (cond.includes("assists")) type = "ast";
      else if (cond.includes("steals")) type = "stl";
      else if (cond.includes("pra")) type = "pra";
      else if (cond.includes("points")) type = "pts";
      return type ? { player: name, type, target } : null;
    }).filter(p => p !== null);
  };

  const getPlayerLiveInfo = (playerName) => {
    if (!liveData) return { onCourt: false, jersey: '?', stats: null };
    const normalize = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\./g, "").toLowerCase().trim();
    const searchName = normalize(playerName);
    const allPlayers = [...liveData.awayTeam.players, ...liveData.homeTeam.players];
    const p = allPlayers.find(p => {
      const apiFull = normalize(`${p.firstName} ${p.familyName}`);
      return apiFull.includes(searchName) || normalize(p.familyName) === searchName || searchName.includes(normalize(p.familyName));
    });
    return p ? { onCourt: p.oncourt === "1", jersey: p.jerseyNum, stats: p.statistics } : { onCourt: false, jersey: '?', stats: null };
  };

  const calculateStat = (stats, type) => {
    if (!stats) return 0;
    switch (type) {
      case 'pts': return stats.points;
      case 'reb': return stats.reboundsTotal;
      case 'ast': return stats.assists;
      case 'stl': return stats.steals;
      case '3pm': return stats.threePointersMade;
      case 'pra': return (stats.points || 0) + (stats.reboundsTotal || 0) + (stats.assists || 0);
      default: return 0;
    }
  };

  const groupedProps = useMemo(() => {
    return parsedProps.reduce((acc, prop) => {
      if (!acc[prop.player]) acc[prop.player] = [];
      acc[prop.player].push(prop);
      return acc;
    }, {});
  }, [parsedProps]);

  return (
    <div className="glass-card">
      <div className="card-top">
        <button onClick={onRemove} className="icon-btn close">✕</button>
        {isLocked && <button onClick={() => onUpdate({ isLocked: false })} className="icon-btn settings">⚙ EDIT</button>}
      </div>

      {!isLocked ? (
        <div className="setup-container">
          <h2 className="setup-title">PROBE SETUP</h2>
          <div className="input-group" style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', fontSize: '10px', color: '#666', marginBottom: '5px' }}>TARGET GAME ID</label>
            <input
              className="terminal-input"
              value={gameId}
              onChange={e => onUpdate({ gameId: e.target.value })}
              onBlur={() => onUpdate({ gameId })}
              placeholder="00XXXXXXXX"
            />
          </div>
          <div className="input-group" style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', fontSize: '10px', color: '#666', marginBottom: '5px' }}>RAW DATA INPUT</label>
            <textarea className="terminal-input" style={{ height: '120px', resize: 'none' }} value={input} onChange={e => onUpdate({ input: e.target.value })} placeholder="Paste parlay text here..." />
          </div>
          <button className="engage-btn" style={{ width: '100%' }} onClick={() => onUpdate({ parsedProps: parseInput(input), isLocked: true })}>ENGAGE TRACKING</button>
        </div>
      ) : (
        <div className="live-container">
          <div className="score-display">
            <div className="team">
              <div className="tricode">{liveData?.awayTeam.teamTricode || '---'}</div>
              <div className="score">{liveData?.awayTeam.score || '0'}</div>
            </div>
            <div className="game-info">
              <div className="clock">{liveData ? (liveData.period > 4 ? `OT${liveData.period - 4}` : `Q${liveData.period}`) : '--'}</div>
              <div className="time">{liveData?.gameClock.replace('PT', '').replace('M', ':').replace('S', '') || '00:00'}</div>
            </div>
            <div className="team">
              <div className="tricode">{liveData?.homeTeam.teamTricode || '---'}</div>
              <div className="score">{liveData?.homeTeam.score || '0'}</div>
            </div>
          </div>

          <div className="player-list">
            {Object.keys(groupedProps).map((name) => {
              const info = getPlayerLiveInfo(name);
              return (
                <div key={name} className="player-row">
                  <div className="player-meta">
                    <span className="name">{name.toUpperCase()} <span className="jersey">#{info.jersey}</span></span>
                    <span className={`court-status ${info.onCourt ? 'active' : ''}`}>
                      {info.onCourt ? 'ON COURT' : 'BENCH'}
                    </span>
                  </div>
                  {groupedProps[name].map((prop, idx) => {
                    const val = calculateStat(info.stats, prop.type);
                    const hit = val >= prop.target;
                    const progress = Math.min((val / prop.target) * 100, 100);
                    return (
                      <div key={idx} className={`prop-stat ${hit ? 'hit' : 'miss'}`}>
                        <div className="prop-label">
                          <span>{prop.type.toUpperCase()}</span>
                          <span className="numerical"><strong>{val}</strong> / {prop.target}</span>
                        </div>
                        <div className="progress-bar">
                          <div className="fill" style={{ width: `${progress}%` }}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <div className="sync-footer">LAST DATA BURST: {lastSync || 'WAITING...'}</div>
        </div>
      )}

      <style jsx>{`
        .card-top { display: flex; justify-content: space-between; margin-bottom: 15px; }
        .icon-btn { background: none; border: none; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; }
        .icon-btn.close { color: #ff4444; }
        .icon-btn.settings { color: #c0c0c0; border: 1px solid #c0c0c0; padding: 2px 8px; border-radius: 4px; }
        .setup-title { font-size: 14px; letter-spacing: 2px; color: #c0c0c0; margin-top: 0; margin-bottom: 20px; }
        .score-display { display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid var(--border); }
        .tricode { font-size: 11px; color: #888; font-weight: bold; }
        .score { font-size: 32px; font-weight: 900; }
        .game-info { text-align: center; border-left: 1px solid #333; border-right: 1px solid #333; padding: 0 20px; }
        .clock { font-size: 12px; color: yellow; font-weight: bold; }
        .player-row { margin-bottom: 25px; }
        .player-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #222; }
        .name { font-weight: 800; font-size: 14px; color: #c0c0c0; }
        .court-status { font-size: 9px; padding: 2px 6px; border-radius: 4px; background: #222; color: #666; font-weight: bold; letter-spacing: 1px; }
        .court-status.active { background: rgba(0, 255, 0, 0.15); color: #00ff00; box-shadow: 0 0 8px rgba(0, 255, 0, 0.1); }
        .prop-stat { margin-bottom: 12px; }
        .prop-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; font-family: monospace; }
        .prop-stat.hit .numerical { color: var(--success); text-shadow: 0 0 10px rgba(0,255,0,0.3); }
        .prop-stat.miss .numerical { color: var(--danger); }
        .progress-bar { height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
        .fill { height: 100%; border-radius: 3px; transition: width 1s ease, background 0.5s ease; }
        .prop-stat.hit .fill { background: var(--success); box-shadow: 0 0 10px var(--success); }
        .prop-stat.miss .fill { background: var(--danger); box-shadow: 0 0 5px rgba(255,68,68,0.3); }
        .sync-footer { font-size: 9px; color: #444; text-align: center; margin-top: 10px; }
      `}</style>
    </div>
  );
}

// --- COMPONENT: PROP FINDER V3 (UPDATED) ---
function PropFinderV3({ state, setState }) {
  const { games, selectedGame, players, selectedPlayer, statsData, loading, activeThresholds, gameCount, gamesSaved, syncLoading, syncError, gamesError } = state;
  const [venueDropdownOpen, setVenueDropdownOpen] = useState(false);
  const memoizedSeasonStats = useMemo(() => statsData?.meta?.season_stats || {}, [statsData]);
  const memoizedPlayerImage = useMemo(() => {
    const playerId = statsData?.meta?.player_id;
    return playerId ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${playerId}.png` : "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png";
  }, [statsData]);

  const refreshGamesSaved = () => {
    fetch(`${API}/season-games-count`)
      .then(res => res.json())
      .then(data => {
        if (data.games_saved !== undefined) setState(prev => ({ ...prev, gamesSaved: data.games_saved }));
      })
      .catch(() => {});
  };

  // Track if games have been loaded initially
  const gamesLoadedRef = React.useRef(false);

  // Function to fetch and update games
  const fetchGames = useCallback(() => {
    console.log('Fetching games from:', `${API}/games`);
    fetch(`${API}/games`)
      .then(res => {
        console.log('Response status:', res.status, res.statusText);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data.games && data.games.length > 0) {
          setState(prev => {
            const isInitialLoad = !gamesLoadedRef.current;
            gamesLoadedRef.current = true;
            
            // Update games list
            const newState = { ...prev, games: data.games, gamesError: null };
            
            if (isInitialLoad) {
              // On initial load, trigger game selection after state update
              setTimeout(() => handleGameSelect(0, data.games[0]), 0);
            } else {
              // On refresh, check if current game still exists
              const currentGameIds = prev.games.map(g => `${g.visitor_id}-${g.home_id}`);
              const newGameIds = data.games.map(g => `${g.visitor_id}-${g.home_id}`);
              const currentStillExists = currentGameIds.includes(`${prev.selectedGame?.visitor_id}-${prev.selectedGame?.home_id}`);
              
              if (!currentStillExists && data.games.length > 0) {
                setTimeout(() => handleGameSelect(0, data.games[0]), 0);
              }
            }
            
            return newState;
          });
        } else {
          const range = data.start_date && data.end_date ? `No games found from ${data.start_date} to ${data.end_date}.` : 'No games found in the current search window.';
          setState(prev => ({ ...prev, games: [], selectedGame: null, players: [], selectedPlayer: '', gamesError: range }));
        }
      })
      .catch((err) => {
        console.error('Failed to fetch games:', err);
        setState(prev => ({ ...prev, gamesError: 'Unable to load games right now.' }));
      });
  }, [API]);

  // Initial load + auto-refresh every 120 seconds
  useEffect(() => {
    // Fetch immediately on mount
    fetchGames();
    
    // Set up interval for every 120 seconds
    const intervalId = setInterval(() => {
      fetchGames();
    }, 120000);
    
    // Cleanup interval on unmount
    return () => clearInterval(intervalId);
  }, [fetchGames]);

  useEffect(() => {
    refreshGamesSaved();
  }, []);

  const handleGameSelect = useCallback((idx, gameOverride = null) => {
    const game = gameOverride || games[idx];
    if (!game) return;

    setState(prev => ({ ...prev, selectedGame: game }));

    fetch(`${API}/rosters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ visitor_id: game.visitor_id, home_id: game.home_id })
    }).then(res => res.json()).then(data => {
      if (data.players && data.players.length > 0) {
        setState(prev => ({
          ...prev,
          players: data.players,
          selectedPlayer: data.players[0]
        }));
      }
    });
  }, [games, API]);

  const getSuggestedProps = useCallback(() => {
    return statsData?.suggestions || [];
  }, [statsData]);

  // Compute suggested props once per render (avoid calling twice)
  const suggestedProps = getSuggestedProps();

  const handleGenerate = useCallback(() => {
    setState(prev => ({ ...prev, loading: true, activeThresholds: {}, oddsData: null, reportGameCount: prev.gameCount }));

    // Determine opponent team: send both IDs, backend resolves via player's team
    const visitor_id = selectedGame?.visitor_id;
    const home_id = selectedGame?.home_id;

    fetch(`${API}/player-stats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        player_name: selectedPlayer,
        game_count: gameCount,
        visitor_id: visitor_id,
        home_id: home_id
      })
    })
      .then(res => res.json())
      .then(data => {
        setState(prev => ({ ...prev, statsData: data, loading: false }));
      });
  }, [selectedGame?.visitor_id, selectedGame?.home_id, selectedPlayer, gameCount, API]);

  const getOrderedStats = useCallback(() => {
    const order = ['PTS', 'REB', 'AST', 'PRA', 'STL', 'BLK', '3PM'];
    return statsData ? order.map(k => [k, statsData[k]]).filter(i => i[1]) : [];
  }, [statsData]);

  // Helper to safely get season stats (or return placeholders)
  const getSeasonStat = (key, isPct = false) => {
    // Use memoized season stats
    const val = memoizedSeasonStats[key] || 0;
    return isPct ? `${val}%` : val;
  };

  // Helper to get Player Image (Uses memoized calculation)
  const getPlayerImage = () => {
    return memoizedPlayerImage;
  };

  // --- ADDED: POISSON LOGIC FOR LIVE LINE EDITS ---
  const poissonCDF = (lambda, k) => {
    let sum = 0;
    let p = Math.exp(-lambda);
    let current = p;
    for (let i = 0; i <= k; i++) {
      sum += current;
      current *= lambda / (i + 1);
    }
    return sum;
  };

  // --- STAT ROW DISPLAY HELPER ---
  const renderStatRow = (cat) => {
    const pred = statsData.predictions[cat];
    if (!pred) return null;

    const matchupColor = pred.matchup_factor > 1.02 ? '#00ff88' : pred.matchup_factor < 0.98 ? '#ff4444' : '#888';
    const matchupArrow = pred.matchup_factor > 1.02 ? '↑' : pred.matchup_factor < 0.98 ? '↓' : '→';
    const adjustment = (pred.matchup_factor - 1) * 100;
    const matchupLabel = pred.matchup_applied
      ? (Math.abs(adjustment) < 0.5 ? `→ NEUTRAL` : `${matchupArrow} ${adjustment > 0 ? '+' : ''}${adjustment.toFixed(1)}%`)
      : '';

    return (
      <div key={cat} className="pred-row">
        <div className="pred-cat-header">
          <span className="pred-cat-name">{cat}
            {pred.matchup_applied && (
              <span className="matchup-tag" style={{ color: matchupColor }}> {matchupLabel}</span>
            )}
          </span>
          <span className="pred-expected">{pred.expected}</span>
        </div>

        <div className="pred-range">
          <span className="pred-floor">{pred.floor}</span>
          <div className="pred-bar-container">
            <div className="pred-bar-fill" style={{ width: `${pred.confidence}%` }}></div>
          </div>
          <span className="pred-ceiling">{pred.ceiling}</span>
        </div>
        <div className="pred-confidence">
          CONFIDENCE: {pred.confidence}%
          {pred.matchup_applied && <span className="raw-val-hint"> (RAW: {pred.raw_expected})</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="finder-wrapper">
      {/* --- TOP CONTROLS --- */}
      <div className="finder-controls glass-panel">
        <div className="f-group">
          <label className="terminal-label">MISSION_SELECT</label>
          <select
            className="terminal-input"
            value={games.findIndex(g => g.visitor_id === selectedGame?.visitor_id && g.home_id === selectedGame?.home_id)}
            onChange={(e) => handleGameSelect(e.target.value)}
            disabled={games.length === 0}
          >
            {games.map((g, i) => <option key={i} value={i}>{g.label}</option>)}
          </select>
        </div>

        <div className="f-group">
          <label className="terminal-label">OPERATIVE_SCAN</label>
          <select
            className="terminal-input"
            value={selectedPlayer}
            onChange={(e) => setState(prev => ({ ...prev, selectedPlayer: e.target.value }))}
            disabled={!selectedGame}
          >
            {players.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="f-group">
          <label className="terminal-label">GAME_LIMIT</label>
          <select
            className="terminal-input"
            value={gameCount}
            onChange={(e) => setState(prev => ({ ...prev, gameCount: parseInt(e.target.value) }))}
          >
            <option value={10}>LAST 10</option>
            <option value={15}>LAST 15</option>
            <option value={20}>LAST 20</option>
          </select>
        </div>

        <button className="engage-btn" onClick={handleGenerate} disabled={!selectedPlayer || loading}>
          {loading ? <><span className="loading-spinner"></span>SCANNING...</> : `GENERATE_L${gameCount}_REPORT`}
        </button>

        <div className="f-group season-sync-row">
          <div className="games-saved-wrapper">
            <span className="games-saved-label">GAMES_SAVED: {gamesSaved != null ? gamesSaved : '—'}</span>
            {gamesSaved != null && (
              <span className="games-tooltip">(Unique games in database)</span>
            )}
          </div>
          <button
            className="sync-season-btn"
            onClick={() => {
              setState(prev => ({ ...prev, syncLoading: true, syncError: null }));
              fetch(`${API}/sync-season-games`, { method: 'POST' })
                .then(res => {
                  return res.json().then(data => {
                    if (!res.ok) throw new Error(data.error || data.message || `Sync failed (${res.status})`);
                    return data;
                  });
                })
                .then(data => {
                  setState(prev => ({
                    ...prev,
                    syncLoading: false,
                    syncError: null,
                    gamesSaved: data.games_saved != null ? data.games_saved : prev.gamesSaved
                  }));
                  // No need to refresh - we already have the correct count from sync response
                })
                .catch(err => {
                  setState(prev => ({
                    ...prev,
                    syncLoading: false,
                    syncError: err.message || 'Sync failed'
                  }));
                });
            }}
            disabled={syncLoading}
          >
            {syncLoading ? <><span className="loading-spinner"></span>SYNCING...</> : 'SYNC_2025-26_SEASON'}
          </button>
          {syncError && <span className="sync-error">{syncError}</span>}
        </div>
      </div>

      {/* --- MAIN CONTENT AREA --- */}
      {gamesError && (
        <div className="no-games-panel glass-panel">
          <div className="no-games-title">NO_GAMES_FOUND</div>
          <div className="no-games-body">{gamesError}</div>
        </div>
      )}
      {statsData && (
        <div className="results-layout">

          <div className="left-column">
            {/* LEFT: PREDICTIONS */}
            {statsData?.predictions && Object.keys(statsData.predictions).length > 0 && (
              <div className="prediction-panel glass-panel">
                <div className="pred-header">
                  <span className="pred-title">PREDICTIVE_ANALYSIS</span>
                  <span className="pred-badge">AI_MODEL</span>
                </div>

                {/* HOME/AWAY GAME INDICATOR */}
                {statsData?.meta?.upcoming_game_is_home !== undefined && (
                  <div className="venue-indicator">
                    <div className="venue-main-row">
                      <div className={`venue-badge ${statsData.meta.upcoming_game_is_home ? 'home' : 'away'}`}>
                        {statsData.meta.upcoming_game_is_home ? '🏠 HOME GAME' : '✈️ AWAY GAME'}
                      </div>
                      {statsData.meta.home_away_splits && (
                        <div className={`venue-performance ${
                          statsData.meta.home_away_splits.home.pts > statsData.meta.home_away_splits.away.pts 
                            ? 'home-better' 
                            : statsData.meta.home_away_splits.home.pts < statsData.meta.home_away_splits.away.pts
                            ? 'away-better'
                            : 'neutral'
                        }`}>
                          {statsData.meta.home_away_splits.home.pts > statsData.meta.home_away_splits.away.pts 
                            ? '📈 Better at HOME' 
                            : statsData.meta.home_away_splits.home.pts < statsData.meta.home_away_splits.away.pts
                            ? '📈 Better on ROAD'
                            : '➡️ Same performance'}
                        </div>
                      )}
                    </div>
                    
                    {statsData.meta.home_away_splits && (
                      <>
                        <div 
                          className="venue-dropdown-toggle" 
                          onClick={() => setVenueDropdownOpen(!venueDropdownOpen)}
                        >
                          <span>View L{state.reportGameCount} Venue Stats</span>
                          <span className={`dropdown-arrow ${venueDropdownOpen ? 'open' : ''}`}>▼</span>
                        </div>
                        
                        <AnimatePresence>
                          {venueDropdownOpen && (
                            <motion.div
                              initial={{ opacity: 0, y: -5, maxHeight: 0 }}
                              animate={{ opacity: 1, y: 0, maxHeight: 500 }}
                              exit={{ opacity: 0, y: -5, maxHeight: 0 }}
                              transition={{ 
                                duration: 0.5, 
                                ease: [0.25, 0.46, 0.45, 0.94],
                                opacity: { duration: 0.4 }
                              }}
                            >
                              <div className="venue-splits">
                                <div className="splits-header">LAST {state.reportGameCount} GAMES SPLIT</div>
                                <div className="split-row">
                                  <div className="split-section">
                                    <div className="split-item">
                                      <span className="split-emoji">🏠</span>
                                      <span className="split-label">HOME</span>
                                    </div>
                                    <div className="split-games-below">({statsData.meta.home_away_splits.home.games} games)</div>
                                  </div>
                                  <div className="split-stats">
                                    <span>{statsData.meta.home_away_splits.home.pts} PTS</span>
                                    <span>{statsData.meta.home_away_splits.home.reb} REB</span>
                                    <span>{statsData.meta.home_away_splits.home.ast} AST</span>
                                  </div>
                                </div>
                                <div className="split-row">
                                  <div className="split-section">
                                    <div className="split-item">
                                      <span className="split-emoji">✈️</span>
                                      <span className="split-label">AWAY</span>
                                    </div>
                                    <div className="split-games-below">({statsData.meta.home_away_splits.away.games} games)</div>
                                  </div>
                                  <div className="split-stats">
                                    <span>{statsData.meta.home_away_splits.away.pts} PTS</span>
                                    <span>{statsData.meta.home_away_splits.away.reb} REB</span>
                                    <span>{statsData.meta.home_away_splits.away.ast} AST</span>
                                  </div>
                                </div>
                                <div className="split-diff-section">
                                  <div className="diff-header">VENUE DIFFERENTIALS</div>
                                  <div className="diff-grid">
                                    <div className={`diff-item ${(statsData.meta.home_away_splits.home.pts - statsData.meta.home_away_splits.away.pts) >= 0 ? 'positive' : 'negative'}`}>
                                      <span className="diff-cat">PTS</span>
                                      <span className="diff-val">
                                        {(statsData.meta.home_away_splits.home.pts - statsData.meta.home_away_splits.away.pts) >= 0 ? '+' : ''}
                                        {(statsData.meta.home_away_splits.home.pts - statsData.meta.home_away_splits.away.pts).toFixed(1)}
                                      </span>
                                      <span className="diff-loc">
                                        {statsData.meta.upcoming_game_is_home ? 'at home' : 'on road'}
                                      </span>
                                    </div>
                                    <div className={`diff-item ${(statsData.meta.home_away_splits.home.reb - statsData.meta.home_away_splits.away.reb) >= 0 ? 'positive' : 'negative'}`}>
                                      <span className="diff-cat">REB</span>
                                      <span className="diff-val">
                                        {(statsData.meta.home_away_splits.home.reb - statsData.meta.home_away_splits.away.reb) >= 0 ? '+' : ''}
                                        {(statsData.meta.home_away_splits.home.reb - statsData.meta.home_away_splits.away.reb).toFixed(1)}
                                      </span>
                                      <span className="diff-loc">
                                        {statsData.meta.upcoming_game_is_home ? 'at home' : 'on road'}
                                      </span>
                                    </div>
                                    <div className={`diff-item ${(statsData.meta.home_away_splits.home.ast - statsData.meta.home_away_splits.away.ast) >= 0 ? 'positive' : 'negative'}`}>
                                      <span className="diff-cat">AST</span>
                                      <span className="diff-val">
                                        {(statsData.meta.home_away_splits.home.ast - statsData.meta.home_away_splits.away.ast) >= 0 ? '+' : ''}
                                        {(statsData.meta.home_away_splits.home.ast - statsData.meta.home_away_splits.away.ast).toFixed(1)}
                                      </span>
                                      <span className="diff-loc">
                                        {statsData.meta.upcoming_game_is_home ? 'at home' : 'on road'}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </>
                    )}
                  </div>
                )}

                {/* CONTEXT INSIGHTS - B2B, Pace, Clutch, Minutes */}
                {statsData?.predictions?.PTS?.context_insights && (
                  <div className="context-insights">
                    <div className="context-header">GAME CONTEXT</div>
                    <div className="context-chips">
                      {Object.entries(statsData.predictions.PTS.context_insights).map(([key, val]) => {
                        if (key === 'b2b' && val.impact === 'negative') {
                          return <span key={key} className="context-chip negative">💤 {val.reason}</span>;
                        }
                        if (key === 'rest' && val.factor !== 1.0) {
                          return <span key={key} className={val.factor > 1 ? 'context-chip positive' : 'context-chip negative'}>⚡ {val.reason}</span>;
                        }
                        if (key === 'pace' && val.factor !== 1.0) {
                          return <span key={key} className={val.factor > 1 ? 'context-chip positive' : 'context-chip negative'}>🚀 {val.reason}</span>;
                        }
                        if (key === 'clutch' && val.factor !== 1.0) {
                          return <span key={key} className={val.factor > 1 ? 'context-chip positive' : 'context-chip negative'}>🔥 {val.reason}</span>;
                        }
                        if (key === 'minutes' && val.factor !== 1.0) {
                          return <span key={key} className={val.factor > 1 ? 'context-chip positive' : 'context-chip negative'}>⏱️ {val.reason}</span>;
                        }
                        if (key === 'opponent' && val.factor !== 1.0) {
                          return <span key={key} className={val.factor > 1 ? 'context-chip positive' : 'context-chip negative'}>🎯 {val.reason}</span>;
                        }
                        return null;
                      })}
                    </div>
                  </div>
                )}

                {['PTS', 'REB', 'AST', 'PRA', '3PM', 'STL', 'BLK'].map(cat => renderStatRow(cat))}
              </div>
            )}
          </div>

          <div className="middle-column">
            {/* MIDDLE: PLAYER PROFILE CARD */}
            <div className="profile-card glass-panel">
              <div className="profile-glow"></div>
              <div className="img-hex-container">
                <img src={getPlayerImage()} alt={selectedPlayer} className="player-img"
                  onError={(e) => e.target.src = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"} />
              </div>

              <h2 className="profile-name">{statsData?.meta?.player_name || selectedPlayer}</h2>
              <div className="profile-team">2025-2026 SEASON</div>

              <div className="season-stats-grid">
                <div className="s-stat">
                  <div className="s-label">AVG PTS</div>
                  <div className="s-val">{getSeasonStat('pts')}</div>
                </div>
                <div className="s-stat">
                  <div className="s-label">AVG REB</div>
                  <div className="s-val">{getSeasonStat('reb')}</div>
                </div>
                <div className="s-stat">
                  <div className="s-label">AVG AST</div>
                  <div className="s-val">{getSeasonStat('ast')}</div>
                </div>
                <div className="s-stat">
                  <div className="s-label">FG %</div>
                  <div className="s-val">{getSeasonStat('fg_pct', true)}</div>
                </div>
              </div>
            </div>

            {/* SUGGESTED PROPS */}
            <div className="suggested-panel glass-panel">
              <div className="suggested-header">
                <span className="sug-title">SUGGESTED PROPS</span>
                <span className="sug-badge">90% HIT</span>
              </div>
              <div className="suggestions-list">
                {suggestedProps.length > 0 ? suggestedProps.map((sug, i) => (
                  <div key={i} className="sug-row">
                    <span className="sug-cat">{sug.cat}</span>
                    <span className="sug-val">{sug.threshold}+</span>
                    <span className="sug-rate">{sug.rate}%</span>
                  </div>
                )) : (
                  <div className="sug-none">NO HIGH-PROBABILITY TARGETS DETECTED</div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT: STATS GRID */}
          <div className="stats-grid-container">
            {getOrderedStats().map(([cat, data]) => (
              <div key={cat} className="stat-category">
                <div className="cat-title">{cat} <span className="l10">L{data.raw.length} RAW</span></div>
                <div className="raw-values">
                  {data.raw.map((v, i) => (
                    <span key={i} className="v-pill" style={{
                      color: activeThresholds[cat] ? (v >= activeThresholds[cat] ? '#00FF00' : '#ff4444') : '#fff'
                    }}>
                      {v}
                    </span>
                  ))}
                </div>
                <div className="hits-grid">
                  {data.hits.map(([t, rate]) => (
                    <div
                      key={t}
                      className="hit-chip"
                      onClick={() => {
                        setState(prev => ({
                          ...prev,
                          activeThresholds: {
                            ...prev.activeThresholds,
                            [cat]: prev.activeThresholds[cat] === t ? null : t
                          }
                        }));
                      }}
                      style={{
                        borderColor: activeThresholds[cat] === t ? 'rgba(160, 160, 160, 0.4)' : 'rgba(160, 160, 160, 0.15)',
                        background: activeThresholds[cat] === t 
                          ? 'linear-gradient(135deg, rgba(160, 160, 160, 0.2), rgba(160, 160, 160, 0.1))' 
                          : 'var(--glass)',
                        boxShadow: activeThresholds[cat] === t 
                          ? '0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)' 
                          : '0 2px 6px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
                      }}
                    >
                      <span className="t-val">{t}</span>
                      <span className="t-plus">+</span>
                      <span className="t-rate" style={{ color: rate >= 70 ? '#00ff00' : rate >= 50 ? 'orange' : '#666' }}>
                        {rate}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      <style jsx>{`
        .finder-wrapper { width: 100%; }
        .glass-panel { 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border: 1px solid var(--border);
          border-radius: 16px;
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-panel:hover {
          box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          border-color: rgba(160, 160, 160, 0.3);
        }
        
        /* CONTROLS */
        .finder-controls { 
          padding: 25px; 
          display: grid; 
          grid-template-columns: 1fr 1fr 120px auto; 
          gap: 20px; 
          align-items: end; 
          margin-bottom: 25px;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .finder-controls:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        }
        .terminal-label { 
          display: block; 
          font-size: 10px; 
          letter-spacing: 2px; 
          color: #666; 
          margin-bottom: 10px; 
          font-weight: bold;
          transition: all 0.2s ease;
        }
        .terminal-input {
          width: 100%;
          padding: 10px 14px;
          background: rgba(20, 20, 25, 0.8);
          border: 1px solid rgba(160, 160, 160, 0.2);
          color: #fff;
          border-radius: 8px;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          backdrop-filter: blur(10px);
        }
        .terminal-input:focus {
          outline: none;
          border-color: rgba(255, 255, 255, 0.4);
          box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.15), 0 4px 12px rgba(0, 0, 0, 0.3);
          transform: translateY(-1px);
          background: rgba(20, 20, 25, 0.95);
        }
        .terminal-input:hover {
          border-color: rgba(160, 160, 160, 0.4);
          background: rgba(20, 20, 25, 0.9);
        }
        .engage-btn {
          width: 100%;
          padding: 14px 20px;
          background: linear-gradient(135deg, #c0c0c0, #4a9e9e);
          color: #ffffff;
          border: none;
          border-radius: 8px;
          font-weight: 700;
          letter-spacing: 1.5px;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          text-shadow: 0 1px 2px rgba(0,0,0,0.2);
          box-shadow: 0 4px 15px rgba(255, 255, 255, 0.25);
        }
        .engage-btn:hover:not(:disabled) {
          transform: translateY(-2px) scale(1.01);
          box-shadow: 0 8px 25px rgba(255, 255, 255, 0.4);
        }
        .engage-btn:active:not(:disabled) {
          transform: translateY(0) scale(0.99);
        }
        .engage-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        .season-sync-row {
          grid-column: 1 / -1;
          display: flex;
          align-items: center;
          gap: 16px;
          flex-wrap: wrap;
        }
        .games-saved-wrapper {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .games-saved-label {
          font-size: 11px;
          letter-spacing: 1px;
          color: #c0c0c0;
          font-weight: bold;
        }
        .games-tooltip {
          font-size: 9px;
          color: #888;
          font-style: italic;
        }
        .sync-season-btn {
          padding: 8px 16px;
          background: rgba(160, 160, 160, 0.25);
          border: 1px solid rgba(160, 160, 160, 0.4);
          color: #c0c0c0;
          border-radius: 4px;
          font-size: 11px;
          letter-spacing: 1px;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          text-shadow: 0 1px 1px rgba(0, 0, 0, 0.3);
        }
        .sync-season-btn:hover:not(:disabled) {
          background: rgba(160, 160, 160, 0.4);
          border-color: #c0c0c0;
          box-shadow: 
            0 0 12px rgba(160, 160, 160, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        .sync-season-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }
        .sync-error {
          color: #f66;
          font-size: 11px;
          margin-left: 8px;
        }
        .no-games-panel {
          padding: 24px;
          text-align: center;
          margin-bottom: 25px;
        }
        .no-games-title {
          font-size: 12px;
          letter-spacing: 2px;
          color: #c0c0c0;
          font-weight: bold;
          margin-bottom: 10px;
        }
        .no-games-body {
          font-size: 12px;
          color: #aaa;
        }

        /* LAYOUT SPLIT */
        .results-layout { display: grid; grid-template-columns: 320px 320px 1fr; gap: 25px; align-items: start; }
        .left-column, .middle-column { width: 100%; display: flex; flex-direction: column; gap: 25px; }

        /* SUGGESTED PANEL */
        .suggested-panel { 
          padding: 28px;
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 16px;
          border: 1px solid var(--border);
          transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .suggested-panel:hover {
          transform: translateY(-4px);
          box-shadow: 
            0 12px 40px rgba(160, 160, 160, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.4);
        }
        .suggested-header { 
          display: flex; 
          justify-content: space-between; 
          align-items: center; 
          margin-bottom: 10px; 
          border-bottom: 1px solid rgba(255, 255, 255, 0.1); 
          padding-bottom: 6px;
          gap: 8px;
        }
        .sug-title { 
          font-size: 11px; 
          letter-spacing: 1px; 
          color: #c0c0c0; 
          font-weight: 800;
          text-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          white-space: nowrap;
        }
        .sug-badge { 
          font-size: 9px; 
          background: linear-gradient(135deg, rgba(50, 215, 75, 0.2), rgba(100, 200, 100, 0.1)); 
          color: #6bc9a7; 
          padding: 2px 6px; 
          border-radius: 10px; 
          border: 1px solid rgba(50, 215, 75, 0.3);
          font-weight: 600;
          letter-spacing: 0.5px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          white-space: nowrap;
        }
        .sug-row { 
          display: flex; 
          justify-content: space-between; 
          align-items: center;
          padding: 6px 10px; 
          border-bottom: 1px solid rgba(255,255,255,0.08); 
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          border-radius: 8px;
          margin-bottom: 4px;
          background: rgba(255,255,255,0.05);
          box-shadow: 
            0 2px 8px rgba(0,0,0,0.1),
            inset 0 0 0 1px rgba(255, 255, 255, 0.03);
        }
        .sug-row:hover {
          background: linear-gradient(135deg, rgba(160, 160, 160, 0.15), rgba(107, 201, 167, 0.1));
          transform: scale(1.02);
          border-bottom: 1px solid rgba(160, 160, 160, 0.3);
          box-shadow: 
            0 4px 15px rgba(160, 160, 160, 0.2),
            inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        .sug-cat { 
          color: rgba(255, 255, 255, 0.8); 
          font-weight: 600; 
          font-size: 11px;
          letter-spacing: 0.5px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .sug-val { 
          color: #fff; 
          font-weight: 800; 
          font-size: 13px;
          text-shadow: 0 0 10px rgba(255,255,255,0.4);
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .sug-rate { 
          color: var(--success); 
          text-shadow: 0 0 10px rgba(50, 215, 75, 0.4);
          font-weight: 800;
          font-size: 11px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .sug-none { 
          font-size: 10px; 
          color: #555; 
          text-align: center; 
          padding: 30px 20px;
          background: rgba(255,255,255,0.02);
          border-radius: 8px;
          border: 1px dashed rgba(255,255,255,0.1);
        }

        /* PROFILE CARD STYLES */
        .profile-card { 
          padding: 35px 25px; 
          text-align: center; 
          position: relative; 
          overflow: hidden; 
          height: fit-content;
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border: 1px solid var(--border);
          border-radius: 16px;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .profile-card:hover {
          transform: translateY(-4px);
          box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.4);
        }
        .profile-glow { 
          position: absolute; 
          top: -50px; 
          left: -50px; 
          width: 200px; 
          height: 200px; 
          background: rgba(160, 160, 160, 0.1); 
          filter: blur(60px); 
          z-index: 0;
          transition: all 0.3s ease;
        }
        .profile-card:hover .profile-glow {
          transform: scale(1.2);
          opacity: 0.3;
        }
        
        .img-hex-container {
          position: relative;
          width: 180px;
          height: 180px;
          margin: 0 auto 20px;
          border-radius: 50%;
          border: 3px solid var(--border);
          overflow: hidden;
          background: radial-gradient(circle, #2a2a2f, #1a1a1f);
          box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.3),
            inset 0 0 0 1px rgba(255, 255, 255, 0.1);
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .img-hex-container:hover {
          border-color: rgba(255, 255, 255, 0.4);
          box-shadow: 
            0 12px 35px rgba(160, 160, 160, 0.25),
            inset 0 0 0 1px rgba(255, 255, 255, 0.15);
        }
        .player-img { width: 100%; height: 100%; object-fit: cover; object-position: top; }

        .profile-name { 
          font-size: 22px; 
          font-weight: 800; 
          color: #fff; 
          margin: 0 0 8px; 
          text-transform: none; 
          letter-spacing: 0.5px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .profile-team { 
          font-size: 12px; 
          font-weight: 600; 
          color: #c0c0c0; 
          margin-bottom: 30px; 
          letter-spacing: 1.5px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .season-stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .s-stat { 
          background: var(--glass);
          padding: 18px; 
          border-radius: 16px; 
          border: 1px solid var(--border);
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          box-shadow: 
            0 6px 20px rgba(0,0,0,0.15),
            inset 0 0 0 1px rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
        }
        .s-stat:hover {
          background: linear-gradient(135deg, rgba(160, 160, 160, 0.2), rgba(192, 192, 192, 0.1));
          transform: translateY(-4px) scale(1.05);
          border-color: rgba(255, 255, 255, 0.4);
          box-shadow: 
            0 10px 30px rgba(160, 160, 160, 0.25),
            inset 0 0 0 1px rgba(160, 160, 160, 0.1);
        }
        .s-label { 
          font-size: 10px; 
          color: rgba(255, 255, 255, 0.7); 
          font-weight: 600; 
          margin-bottom: 6px;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          letter-spacing: 0.8px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          text-transform: uppercase;
        }
        .s-stat:hover .s-label {
          color: #c0c0c0;
        }
        .s-val { 
          font-size: 22px; 
          font-weight: 800; 
          color: #fff;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .s-stat:hover .s-val {
          color: #c0c0c0;
          text-shadow: 0 0 12px rgba(160, 160, 160, 0.5);
          transform: scale(1.1);
        }

        /* RIGHT: STATS GRID STYLES */
        .stats-grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; width: 100%; }
        
        .stat-category { 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 16px;
          padding: 25px;
          border: 1px solid var(--border);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .stat-category:hover {
          transform: translateY(-4px);
          background: rgba(45, 45, 50, 0.9);
          box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.4);
        }
        .cat-title { 
          color: #c0c0c0; 
          font-weight: 800; 
          font-size: 14px; 
          margin-bottom: 15px; 
          border-bottom: 1px solid rgba(255, 255, 255, 0.1); 
          padding-bottom: 8px;
          transition: all 0.2s ease;
          letter-spacing: 0.5px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stat-category:hover .cat-title {
          color: #e0e0e0;
          text-shadow: 0 0 12px rgba(255, 255, 255, 0.3);
        }
        .raw-values { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
        .v-pill { font-family: monospace; font-weight: bold; font-size: 14px; width: 25px; text-align: center; }
        .hits-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 8px; }
        .hit-chip { 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          padding: 12px 14px; 
          border-radius: 12px; 
          display: flex; 
          justify-content: center; 
          align-items: center;
          cursor: pointer; 
          border: 1px solid var(--border); 
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 
            0 4px 15px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.08) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.04) 100%);
          position: relative;
          text-align: center;
        }
        .hit-chip:hover { 
          border-color: rgba(255, 255, 255, 0.3); 
          background: rgba(50, 50, 55, 0.9);
          transform: translateY(-2px);
          box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          z-index: 10;
        }
        .hit-chip:active {
          transform: translateY(0);
          transition: all 0.1s ease;
        }
        
        /* Fix percentage alignment */
        .t-val { 
          font-size: 15px; 
          font-weight: 800; 
          color: #fff;
          transition: all 0.2s ease;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          white-space: nowrap;
          position: relative;
          top: 0px;
          display: inline-block;
          text-align: center;
          vertical-align: middle;
        }
        .hit-chip:hover .t-val {
          color: #e0e0e0;
          text-shadow: 0 0 8px rgba(255, 255, 255, 0.3);
        }
        .t-plus {
          font-size: 15px;
          font-weight: 800;
          color: #fff;
          margin: 0 2px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          white-space: nowrap;
          display: inline-block;
          text-align: center;
          vertical-align: middle;
          position: relative;
          top: 0px;
        }
        .t-rate { 
          font-size: 8px; 
          font-weight: 800;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          white-space: nowrap;
          position: relative;
          top: 0px;
          left: 4px;
          display: inline-block;
          text-align: center;
          vertical-align: middle;
        }
        .hit-chip:hover .t-rate {
          transform: scale(1.1);
          color: #c0c0c0;
        }

        /* GLOBAL SMOOTH TRANSITIONS */
        * {
          transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        
        /* PREDICTION PANEL */
        .prediction-panel { 
          padding: 20px; 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 16px;
          border: 1px solid var(--border);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .prediction-panel:hover {
          transform: translateY(-4px);
          box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.4);
        }
        .pred-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; }
        .pred-title { color: #c0c0c0; font-weight: 900; font-size: 13px; letter-spacing: 2px; }
        .context-insights {
          margin: 15px 0;
          padding: 14px;
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 12px;
          border: 1px solid var(--border);
          box-shadow: 
            0 4px 15px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.06),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.06) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.03) 100%);
        }
        .context-header {
          color: #999;
          font-size: 10px;
          letter-spacing: 1.5px;
          margin-bottom: 12px;
          font-weight: 700;
        }
        .context-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .context-chip {
          font-size: 10px;
          padding: 6px 12px;
          border-radius: 10px;
          font-weight: 700;
          letter-spacing: 0.5px;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }
        .context-chip.positive {
          background: linear-gradient(135deg, rgba(50, 215, 75, 0.25), rgba(50, 215, 75, 0.12));
          color: #4ade80;
          border: 1px solid rgba(50, 215, 75, 0.35);
        }
        .context-chip.negative {
          background: linear-gradient(135deg, rgba(255, 68, 68, 0.25), rgba(255, 68, 68, 0.12));
          color: #f87171;
          border: 1px solid rgba(255, 68, 68, 0.35);
        }
        .pred-badge { 
          font-size: 9px; 
          background: linear-gradient(135deg, rgba(160, 160, 160, 0.2), rgba(160, 160, 160, 0.1)); 
          color: #c0c0c0; 
          padding: 4px 10px; 
          border-radius: 8px; 
          font-weight: 700; 
          letter-spacing: 1px; 
          border: 1px solid rgba(160, 160, 160, 0.25);
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }
        .pred-row { 
          margin-bottom: 16px; 
          padding: 12px 14px; 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 12px; 
          border: 1px solid var(--border);
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 
            0 4px 15px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.06) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.03) 100%);
        }
        .pred-row:hover { 
          transform: translateY(-2px); 
          border-color: rgba(255, 255, 255, 0.3); 
          background: rgba(50, 50, 55, 0.9);
          box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
        }
        .pred-cat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .pred-cat-name { font-size: 11px; font-weight: bold; color: #888; letter-spacing: 1px; }
        .pred-expected { 
          font-size: 22px; 
          font-weight: 900; 
          color: #e0e0e0; 
          text-shadow: 0 0 10px rgba(255, 255, 255, 0.2); 
        }
        .pred-range { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
        .pred-floor, .pred-ceiling { font-size: 10px; color: #666; font-weight: bold; min-width: 25px; }
        .pred-bar-container { flex: 1; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; }
        .pred-bar-fill { 
          height: 100%; 
          background: linear-gradient(90deg, #6bc9a7, #8dd8b8); 
          border-radius: 2px; 
          transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); 
        }
        .pred-confidence { font-size: 9px; color: #888; font-weight: bold; letter-spacing: 1px; }

        .raw-val-hint { color: #999; font-style: italic; margin-left: 5px; }

        /* VENUE INDICATOR */
        .venue-indicator { 
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 16px;
          padding: 16px;
          margin-bottom: 15px;
          border: 1px solid var(--border);
          box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.3);
          background-image: 
            linear-gradient(135deg, rgba(160, 160, 160, 0.1) 0%, transparent 30%),
            linear-gradient(45deg, transparent 70%, rgba(192, 192, 192, 0.05) 100%);
        }
        .venue-main-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
        }
        .venue-badge { 
          font-size: 11px; 
          font-weight: 800; 
          padding: 8px 14px; 
          border-radius: 10px;
          letter-spacing: 1px;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .venue-badge.home { 
          background: linear-gradient(135deg, rgba(50, 180, 120, 0.2), rgba(50, 180, 120, 0.1)); 
          color: #6bc9a7; 
          border: 1px solid rgba(107, 201, 167, 0.3);
          box-shadow: 0 2px 8px rgba(50, 180, 120, 0.15);
        }
        .venue-badge.away { 
          background: linear-gradient(135deg, rgba(255, 170, 0, 0.2), rgba(255, 170, 0, 0.1)); 
          color: #ffc14d; 
          border: 1px solid rgba(255, 170, 0, 0.3);
          box-shadow: 0 2px 8px rgba(255, 170, 0, 0.15);
        }
        .venue-performance {
          font-size: 10px;
          font-weight: 800;
          padding: 8px 12px;
          border-radius: 10px;
          letter-spacing: 0.5px;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .venue-performance.home-better {
          background: linear-gradient(135deg, rgba(50, 180, 120, 0.15), rgba(50, 180, 120, 0.08));
          color: #6bc9a7;
          border: 1px solid rgba(107, 201, 167, 0.25);
          box-shadow: 0 2px 6px rgba(50, 180, 120, 0.1);
        }
        .venue-performance.away-better {
          background: linear-gradient(135deg, rgba(255, 170, 0, 0.15), rgba(255, 170, 0, 0.08));
          color: #ffc14d;
          border: 1px solid rgba(255, 170, 0, 0.25);
          box-shadow: 0 2px 6px rgba(255, 170, 0, 0.1);
        }
        .venue-performance.neutral {
          background: linear-gradient(135deg, rgba(136, 136, 136, 0.15), rgba(136, 136, 136, 0.08));
          color: #999;
          border: 1px solid rgba(136, 136, 136, 0.2);
          box-shadow: 0 2px 6px rgba(136, 136, 136, 0.1);
        }
        .venue-dropdown-toggle {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          margin-top: 12px;
          background: var(--glass);
          border-radius: 10px;
          cursor: pointer;
          font-size: 11px;
          color: #999;
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          border: 1px solid var(--border);
          box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }
        .venue-dropdown-toggle:hover {
          background: rgba(50, 50, 55, 0.9);
          color: #ccc;
          border-color: rgba(160, 160, 160, 0.3);
          transform: translateY(-2px);
          box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }
        .dropdown-arrow {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          font-size: 10px;
          color: #a0a0a0;
        }
        .dropdown-arrow.open {
          transform: rotate(180deg) scale(1.1);
        }
        .venue-splits { 
          margin-top: 8px;
          padding: 14px;
          background: var(--glass);
          backdrop-filter: blur(15px) saturate(120%);
          -webkit-backdrop-filter: blur(15px) saturate(120%);
          border-radius: 12px;
          border: 1px solid var(--border);
          box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2);
          animation: dropdownSlideDown 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        @keyframes dropdownSlideDown {
          0% {
            opacity: 0;
            transform: translateY(-5px);
            max-height: 0;
          }
          100% {
            opacity: 1;
            transform: translateY(0);
            max-height: 500px;
          }
        }
        
        @keyframes dropdownSlideUp {
          0% {
            opacity: 1;
            transform: translateY(0);
            max-height: 500px;
          }
          100% {
            opacity: 0;
            transform: translateY(-5px);
            max-height: 0;
          }
        }
        .splits-header {
          font-size: 10px;
          color: #c0c0c0;
          font-weight: 900;
          letter-spacing: 1.5px;
          margin-bottom: 16px;
          text-align: center;
          text-transform: uppercase;
        }
        .split-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 12px;
          background: rgba(255,255,255,0.03);
          border-radius: 6px;
          margin-bottom: 8px;
          margin-top: -4px;
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          border: 1px solid transparent;
          min-width: 220px;
          max-width: 260px;
        }
        .split-row:hover {
          background: linear-gradient(90deg, rgba(160, 160, 160, 0.1), rgba(160, 160, 160, 0.05));
          border-color: rgba(160, 160, 160, 0.3);
          transform: scale(1.02);
          box-shadow: 0 4px 12px rgba(160, 160, 160, 0.15);
        }
        .split-item { 
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          flex-shrink: 0;
        }
        .split-emoji {
          font-size: 14px;
          line-height: 1;
        }
        .split-label { 
          color: #ccc; 
          font-weight: 800; 
          font-size: 11px;
          letter-spacing: 0.5px;
        }
        .split-games {
          color: #666;
          font-size: 10px;
          font-weight: bold;
          background: rgba(255,255,255,0.1);
          padding: 2px 8px;
          border-radius: 10px;
        }
        .split-section {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          justify-content: center;
        }
        .split-games-below {
          color: #666;
          font-size: 8px;
          font-weight: 600;
          background: rgba(255,255,255,0.05);
          padding: 1px 6px;
          border-radius: 8px;
          text-align: center;
        }
        /* Compact column alignment with minimal spacing */
        .split-stats {
          display: flex;
          gap: 2px;
          font-size: 9px;
          font-weight: 800;
          color: #fff;
          min-width: 70px;
          flex-shrink: 0;
          align-self: center;
          justify-content: flex-start;
          align-items: center;
        }
        
        .split-stats span {
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          padding: 1px 3px;
          border-radius: 4px;
          text-align: right;
          min-width: 50px;
        }
        
        /* Reset all transforms for clean column alignment */
        .split-row:nth-child(1) .split-stats,
        .split-row:nth-child(2) .split-stats {
          transform: none;
        }
        .split-stats span {
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          padding: 2px 6px;
          border-radius: 4px;
        }
        .split-stats span:hover {
          background: rgba(160, 160, 160, 0.2);
          transform: scale(1.1);
          box-shadow: 0 0 8px rgba(160, 160, 160, 0.3);
        }
        /* SPLIT DIFF SECTION - APPLE LIQUID METAL THEME */
        .split-diff-section {
          margin-top: 10px;
          padding: 12px;
          background: linear-gradient(135deg, rgba(160, 160, 160, 0.1), rgba(192, 192, 192, 0.05));
          border-radius: 10px;
          border: 1px solid rgba(160, 160, 160, 0.2);
          box-shadow: 
            0 4px 12px rgba(0,0,0,0.2),
            inset 0 0 0 1px rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(15px) saturate(180%);
          -webkit-backdrop-filter: blur(15px) saturate(180%);
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }
        .split-diff-section:hover {
          transform: translateY(-3px) scale(1.03);
          background: linear-gradient(135deg, rgba(160, 160, 160, 0.15), rgba(192, 192, 192, 0.1));
          border-color: rgba(255, 255, 255, 0.4);
          box-shadow: 
            0 8px 20px rgba(160, 160, 160, 0.25),
            inset 0 0 0 1px rgba(160, 160, 160, 0.1);
        }
        .diff-header {
          font-size: 9px;
          color: #c0c0c0;
          font-weight: 800;
          letter-spacing: 1px;
          text-align: center;
          margin-bottom: 10px;
          text-transform: uppercase;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .diff-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .diff-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 8px 6px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          border: 1px solid rgba(160, 160, 160, 0.15);
          transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }
        .diff-item:hover {
          transform: translateY(-2px) scale(1.03);
          background: rgba(160, 160, 160, 0.1);
          border-color: rgba(255, 255, 255, 0.4);
          box-shadow: 0 4px 12px rgba(160, 160, 160, 0.2);
        }
        .diff-cat {
          font-size: 9px;
          color: #ccc;
          font-weight: 700;
          letter-spacing: 0.5px;
          margin-bottom: 3px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .diff-val {
          font-size: 11px;
          font-weight: 900;
          margin-bottom: 2px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .diff-item.positive .diff-val {
          color: #6bc9a7;
          text-shadow: 0 0 8px rgba(107, 201, 167, 0.4);
        }
        .diff-item.negative .diff-val {
          color: #ff6b6b;
          text-shadow: 0 0 8px rgba(255, 107, 107, 0.4);
        }
        .diff-loc {
          font-size: 7px;
          color: rgba(255, 255, 255, 0.6);
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.3px;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
      `}</style>
    </div>
  );
}
