#!/bin/bash
python run.py &
RUN_PID=$!
echo "Started run.py with PID $RUN_PID"
sleep 3
ps aux | grep $RUN_PID
kill -INT $RUN_PID
wait $RUN_PID
