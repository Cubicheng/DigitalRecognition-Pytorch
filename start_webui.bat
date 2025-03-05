@echo off
echo Serve is starting...
python -u serve.py
start http://127.0.0.1:5000
pause