import requests
import pandas as pd
import time
import os

# Configuration
API_KEY = "579b464db66ec23bdd00000199b6ed9f40064b0d7dafcd2d46a54982"
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
TARGET = 21000
FILENAME = "india_aqi_21000.csv"
BATCH_SIZE = 500  

def get_total_saved():
    if not os.path.exists(FILENAME): return 0
    try:
        return len(pd.read_csv(FILENAME))
    except: return 0

print(f"--- JOB START: Target {TARGET} records ---")

try:
    while True:
        current_count = get_total_saved()
        if current_count >= TARGET:
            print(f"Goal reached! {current_count} records saved.")
            break
            
        print(f"\nStatus: {current_count}/{TARGET} records. Starting new pull...")
        offset = 0  # We reset offset to get the latest live snapshot
        
        while True:
            url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&limit={BATCH_SIZE}&offset={offset}"
            
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 502:
                    print("Server 502 (Busy). Waiting 30s...")
                    time.sleep(30)
                    continue
                
                data = response.json()
                records = data.get('records', [])
                
                if not records:
                    print("End of current live buffer reached.")
                    break
                
                # Append to CSV immediately
                df = pd.DataFrame(records)
                file_exists = os.path.isfile(FILENAME)
                df.to_csv(FILENAME, mode='a', index=False, header=not file_exists)
                
                offset += len(records)
                print(f"Saved +{len(records)} records (Total: {get_total_saved()})")
                time.sleep(2)
                
            except Exception as e:
                print(f"Error: {e}. Retrying...")
                time.sleep(10)

        if get_total_saved() < TARGET:
            print("Waiting 1 hour for the next government data refresh...")
            time.sleep(3600)

except KeyboardInterrupt:
    print(f"\nStopped by user. {get_total_saved()} records are safely stored.")
