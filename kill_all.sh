kill -9 $(ps -aux | grep main.py | awk '{print $2}' | xargs)