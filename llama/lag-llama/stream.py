import requests
import zipfile
import io
import json
import pandas as pd
from datetime import datetime

# Direct URL to the 30GB ZIP content
URL = "https://zenodo.org/api/records/10608607/files/CESNET-TLS-Year22.zip/content"
OUTPUT_FILE = "network_metrics_data.csv"

class RemoteStream(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.len = int(requests.head(url, allow_redirects=True).headers['Content-Length'])
    def read(self, size=-1):
        if size == -1: size = self.len - self.pos
        r = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+size-1}'})
        self.pos += len(r.content)
        return r.content
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_END: self.pos = self.len + offset
        return self.pos
    def tell(self): return self.pos

def extract_stats_week():
    print("Connecting to Zenodo... Extracting weekly stats (Univariate).")
    stream = RemoteStream(URL)
    
    with zipfile.ZipFile(stream) as z:
        # 1. Identify JSON stats files for the first week
        target_files = [f for f in z.namelist() if "WEEK-2022-01" in f and f.endswith(".json")]
        
        if not target_files:
            print("No stats files found. Check folder naming in ZIP.")
            return

        all_records = []
        for json_file in sorted(target_files):
            print(f"Reading stats: {json_file}")
            with z.open(json_file) as f:
                data = json.load(f)
                
                # CESNET stats usually contain time-series bins
                # We extract the 'traffic_volume' (bytes) and timestamps
                for entry in data.get('traffic_stats', []):
                    dt = datetime.fromtimestamp(entry['timestamp'])
                    
                    all_records.append({
                        'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
                        'traffic_volume': entry.get('bytes', 0),
                        'source_as': entry.get('src_asn', '0'),
                        'destination_as': entry.get('dst_asn', '0'),
                        'handover_as': '0',
                        'nexthop_as': '0',
                        'flow_key_id': f"{entry.get('src_asn', '0')}_{entry.get('dst_asn', '0')}"
                    })

        # 2. Create DataFrame and Adhere to Schema
        df = pd.DataFrame(all_records)
        
        # 3. Handle Anomaly Labeling (Top 5% as baseline)
        q = df['traffic_volume'].quantile(0.95)
        df['is_anomaly'] = (df['traffic_volume'] > q).astype(int)

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! {OUTPUT_FILE} created using backbone stats.")

if __name__ == "__main__":
    extract_stats_week()
