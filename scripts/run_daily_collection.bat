@echo off
REM Daily training data collection for Polycast ML
REM Schedule this with Windows Task Scheduler to run daily

cd /d C:\Users\Admin\Documents\polycast
python scripts/daily_training_session.py 3 --interval 15

REM After collection, try to retrain if enough data
python scripts/retrain_with_real_data.py

pause
