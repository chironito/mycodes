import psycopg2
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment file not found."
import os
from datetime import datetime, timedelta
import traceback
import requests
from time import sleep
import argparse
from typing import List

args = argparse.ArgumentParser()
args.add_argument("--minutes", default=10, type=int)
args = args.parse_args()

system_prompt = """You are a Chiri - an LLM powered VTuber who maintains a persistent memory of each user."""

memory_prompt = """Below is:
- The user's existing memory summary
- A recent batch of messages from the user and your responses to those messages.
Your task is to generate an updated memory summary for this user that reflects their interests, personality, emotional state, preferences, and any relevant changes or new information.

The memory should be:
- Concise, factual, and emotionally intelligent
- Written in the third person ("User likes...", "User seems interested in...")
- Avoid speculation or assumptions without evidence
- Avoid repeating unchanged facts unless they’re central to identity
- Focus on persistent traits or habits more than one-time events

---

- Existing memory:
{previous_memory}

- Recent messages from the user and Chiri's responses:
{recent_user_messages}

---

Write a single updated memory summary below:"""

def store_exception(exception_type: str, exception_info: str, mail_sent: bool=None) -> bool:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''INSERT INTO exceptions_table (exception_type, exception_info, timestamp_utc, mail_sent) VALUES (%s, %s, %s, %s)''', (exception_type, exception_info, datetime.now(), mail_sent))
    except Exception as exp:
        with open("store_exception_exception", "w") as f:
            f.write("\n".join(traceback.format_exception(exp)))
        return False
    else:
        return True

def get_previous_memory(user_id: str) -> str:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT memory FROM memory_table WHERE user_id = %s''', (str(user_id),))
                try:
                    memory = cursor.fetchone()[0] or "N/A"
                except TypeError:
                    memory = "N/A"
    except Exception as exp:
        store_exception("get_previous_memory", "\n".join(traceback.format_exception(exp)))
        return "N/A"
    else:
        return memory

def get_recent_user_ids(minutes: int=10) -> List[str]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT user_id FROM chat_table WHERE user_message_utc >= NOW() - INTERVAL '%s minutes' """ % (minutes,))
                try:
                    recent_user_ids = [entry[0] for entry in cursor.fetchall()]
                except TypeError:
                    recent_user_ids = []
    except Exception as exp:
        store_exception("get_recent_user_ids", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return recent_user_ids

def get_recent_user_messages(user_id: str) -> List[List[str]]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT user_message, chiri_response FROM chat_table WHERE user_id = %s AND memorized = false''', (str(user_id),))
                conversation_history = cursor.fetchall()
    except Exception as exp:
        store_exception("get_recent_user_messages", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return conversation_history

def store_memory(user_id: str, memory: str) -> bool:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''INSERT INTO memory_table (user_id, memory, timestamp_utc) VALUES (%s, %s, %s)''', (str(user_id), memory, datetime.now()))
    except Exception as exp:
        store_exception("store_memory", "\n".join(traceback.format_exception(exp)))
        return False
    else:
        return True

def set_memorized(user_id: str) -> bool:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''UPDATE chat_table SET memorized = true WHERE user_id = %s AND memorized = false''', (str(user_id),))
    except Exception as exp:
        store_exception("set_memorized", "\n".join(traceback.format_exception(exp)))
        return False
    else:
        return True

def llm_openai(system_prompt: str, user_prompt: str, temperature=0.7, model_name='gpt-4.1-nano', depth=0) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['openai_api_key']}"}
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    payload = {"model": model_name, "temperature": temperature, "input": messages}
    try:
        r = requests.post(os.environ['openai_api_endpoint'], json=payload, headers=headers)
        return r.json()["output"][0]["content"][0]["text"]
    except Exception as exp:
        store_exception(f"llm_openai (Retry attempt: {depth+1})", "\n".join(traceback.format_exception(exp)))
        if depth >= 3:
            raise RecursionError(f"Max retries (3) reached. {r.text}")
        sleep(1)
        return llm_openai(system_prompt, user_prompt, temperature, model_name, depth+1)

if __name__ == "__main__":
    recent_user_ids = get_recent_user_ids(args.minutes)
    for user_id in recent_user_ids:
        print(f"Creating memory for: {user_id}")
        recent_user_messages = get_recent_user_messages(user_id)
        previous_user_memory = get_previous_memory(user_id)
        user_prompt = memory_prompt.format(recent_user_messages="\n\n".join(f"User Message: {user_message}\nChiri Response: {chiri_response or '<no_response>'}" for user_message, chiri_response in recent_user_messages), previous_memory=previous_user_memory)
        new_memory = llm_openai(system_prompt, user_prompt)
        if store_memory(user_id, new_memory):
            print(f"Memory created for: {user_id}")
        else:
            print(f"Memory create failed for: {user_id}")
        set_memorized(user_id)