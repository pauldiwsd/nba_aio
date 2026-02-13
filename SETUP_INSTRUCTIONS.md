# NBA Prop Finder - Setup Instructions

## Backend Setup (Python)

1. **Install Python dependencies:**
```bash
pip install flask flask-cors nba-api pandas
```

2. **Run the backend:**
```bash
python backend_api.py
```

Backend will run on `http://localhost:5000`

## Frontend Setup (React)

1. **Create React app:**
```bash
npx create-react-app nba-prop-finder-web
cd nba-prop-finder-web
```

2. **Replace `src/App.js` with the `frontend_connected.jsx` code**

3. **Install dependencies (if needed):**
```bash
npm install
```

4. **Run the frontend:**
```bash
npm start
```

Frontend will run on `http://localhost:3000`

## Usage

1. Start the Python backend first (port 5000)
2. Start the React frontend (port 3000)
3. Open browser to `http://localhost:3000`
4. Upload your background image
5. Select a game
6. Search for a player
7. Click "GENERATE LATEST 10 GAMES"

## File Structure
```
project/
├── backend_api.py          # Python Flask server
├── requirements.txt        # Python dependencies
└── nba-prop-finder-web/    # React app folder
    └── src/
        └── App.js          # Frontend code
```

## Deploying Online

### Backend (Free Options):
- **Render.com** (easiest)
- **Railway.app**
- **PythonAnywhere**

### Frontend:
- **Vercel** (recommended)
- **Netlify**

When deployed, update the `API_BASE` in frontend from `http://localhost:5000/api` to your deployed backend URL.
