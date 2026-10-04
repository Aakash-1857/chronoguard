import asyncio
import time
import subprocess
import urllib.request

async def make_noise():
    for _ in range(150):
        try:
            urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=0.1)
        except Exception:
            pass
        await asyncio.sleep(0.1)

async def main():
    print("Starting run.py...")
    proc = subprocess.Popen(["python", "run.py"])
    time.sleep(5) # wait for startup

    print("Generating Streamlit HTTP noise and running tcpreplay...")
    # Run tcpreplay in background
    replay = subprocess.Popen(["tcpreplay", "--intf1=lo0", "--pps=500", "tests/fixtures/throughput_fixture.pcap"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    await make_noise()
    
    print("Sending SIGINT to run.py")
    proc.send_signal(2)
    proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
