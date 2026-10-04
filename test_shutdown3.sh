#!/bin/bash
python run.py &
RUN_PID=$!
echo "Started run.py with PID $RUN_PID"
sleep 5

tcpreplay --intf1=lo0 --pps=500 tests/fixtures/throughput_fixture.pcap > /dev/null 2>&1 &
REPLAY_PID=$!
echo "Started tcpreplay with PID $REPLAY_PID"
sleep 15

echo "Sending SIGINT to run.py (PID $RUN_PID)"
kill -INT $RUN_PID

wait $RUN_PID
echo "run.py exited"

echo "Checking for orphans:"
ps aux | grep -iE 'main\.py|ui/app\.py|streamlit' | grep -v grep || echo "No orphans found."
