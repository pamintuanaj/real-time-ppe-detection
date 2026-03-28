@echo off

echo ====================================
echo PPE Detection Project Runner
echo ====================================

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Training YOLOv8 model...
REM python scripts/train.py

echo.
echo Running Baseline ML...
python scripts/baseline_ml.py

echo.
echo Running NLP Component...
python scripts/nlp_component.py

echo.
echo Running RL Component...
python scripts/rl_qlearning.py

echo.
echo Starting Live Camera Detection...
echo Press 'q' to exit camera
python scripts/live_cam.py

echo.
echo ====================================
echo All tasks completed.
echo ====================================

pause