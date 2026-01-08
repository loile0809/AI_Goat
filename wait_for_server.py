
import time
import requests
import sys

def wait_for_server():
    print("Waiting for server at http://localhost:8000/health ...")
    start = time.time()
    while True:
        try:
            r = requests.get("http://localhost:8000/health", timeout=1)
            if r.status_code == 200:
                print(f"Server is UP! (took {time.time()-start:.1f}s)")
                return
        except Exception:
            pass
        if time.time() - start > 600:
            print("Timeout waiting for server.")
            sys.exit(1)
        time.sleep(5)

if __name__ == "__main__":
    wait_for_server()
