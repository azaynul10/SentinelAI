@echo off
cd /d "%~dp0"
echo Starting Fall Detection Backend...
echo.
call .venv\Scripts\activate.bat
python app.py
pause
