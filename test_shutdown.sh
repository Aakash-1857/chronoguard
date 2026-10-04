#!/bin/bash
sudo python run.py &
RUN_PID=$!
echo "Started run.py with PID $RUN_PID"
# Wait for it to start
sleep 5

# Replay some traffic
sudo tcpreplay --intf1=lo0 --pps=500 tests/fixtures/throughput_fixture.pcap > /dev/null 2>&1 &
REPLAY_PID=$!
sleep 2

# Send SIGINT to run.py
echo "Sending SIGINT to run.py"
sudo kill -INT $RUN_PID

# Wait for it to exit
wait $RUN_PID
EXIT_CODE=$?
echo "run.py exited with code $EXIT_CODE"

# Check for orphans
echo "Checking for orphans:"
ps aux | grep -iE 'main\.py|ui/app\.py|streamlit' | grep -v grep || echo "No orphans found."
