import time
import re
import psycopg2
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not found."
import os
from datetime import datetime, timedelta
import traceback

LOG_FILE = "/var/log/auth.log"
# OUTPUT_LOG = "/var/log/ssh_accepted.log"

def store_accepted_ip_address(line: str, send_mail: bool=True) -> bool:
    try:
        with psycopg2.connect(user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              dbname=os.environ['postgres_dbname'], host=os.environ['postgres_server_ip'], 
                              port=os.environ['postgres_server_port']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''INSERT INTO ssh_accepted_log (line, ip_address, timestamp_utc, send_mail) VALUES (%s, %s, %s, %s)''', (line, (match := re.search(r"(?:\d{1,3}\.){3}(?:\d{1,3})", line)) and match.group() or "N/A", datetime.utcnow(), send_mail))
    except Exception as exp:
        with open("store_accepted_ip_address_exceptions", "a+") as f:
            print("Exception at {datetime}:\n\n{exception}\n\n{line}".format(datetime=datetime.utcnow(), exception="\n".join(traceback.format_exception(exp)), line=line), end=f"\n\n{'-'*100}\n\n", file=f)
        return False
    else:
        return True

def follow(file_path):
    with open(file_path, 'r') as f:
        f.seek(0, 2)  # Move to EOF
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line.strip()

def monitor():
    for line in follow(LOG_FILE):
        if re.search(r"\bsshd\[\d+\]\: Accepted publickey for\b", line):
            if store_accepted_ip_address(line):
                print(f"Line: {line} stored successfully.")
            else:
                print(f"Store Failed.")

if __name__ == "__main__":
    monitor()
