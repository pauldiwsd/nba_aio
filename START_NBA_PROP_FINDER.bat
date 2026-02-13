@echo off
TITLE NBA Prop Finder Launcher

:: 1. Force kill any old Python/Node processes before starting
echo Cleaning up old processes...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul
taskkill /F /IM node.exe /T 2>nul

echo.
echo Starting Flask Backend...
:: Using 'python' instead of 'pythonw' ensures the process is tied to this window
start /B python backend_api.py

echo Starting React Frontend...
cd nba-prop-finder-web
start /B npm start

echo.
echo ==================================================
echo        NBA PROP FINDER IS NOW RUNNING
echo ==================================================
echo.
echo  1. Your browser should open automatically.
echo  2. Keep THIS window open while using the app.
echo  3. When finished, press ANY KEY here to exit.
echo.
echo ==================================================
pause

:: 2. Cleanup: This kills the processes when you press a key
echo.
echo Shutting down and clearing Task Manager...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul
taskkill /F /IM node.exe /T 2>nul
echo Done!
timeout /t 2 >nul
exit