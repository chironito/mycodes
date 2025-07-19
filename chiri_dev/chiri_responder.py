import psycopg2
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not Found."
import os
from batched_async_requests import batched_requests
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union
import argparse
import traceback
import uuid
from time import sleep
import re

args = argparse.ArgumentParser()
args.add_argument("--minutes", default=60, type=int)
args.add_argument("--threshold", default=5, type=int)
args = args.parse_args()

system_prompt = """You are a VTuber named Chiri. You're a cheerful, nerdy anime girl who loves games and tech."""

vtuber_prompt = """Respond to the following message from a viewer in an entertaining, non-repetitive way.

Chat context (recent messages): {chat_history}

User's conversation history with Chiri: {conversation_history}"

User's past preferences/Chiri's memory related to the User: {memory} 

Viewer: {display_name}
Message: "{user_message}" """

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

def get_pending_user_messages(minutes: int=60):
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT user_id, display_name, user_message, user_message_utc, user_type, message_id FROM chat_table WHERE user_message_utc > NOW() - INTERVAL '%s minutes' AND chiri_response IS null""" % (minutes,))
                pending_user_messages = cursor.fetchall()
    except Exception as exp:
        store_exception("get_pending_user_messages", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return pending_user_messages

def get_tokens_count(text: str) -> int:
    return round(len(text.split()) * 1.5)

def get_conversation_history(user_id: uuid.UUID, max_tokens=4096, n=25) -> List[List[str]]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT user_id, display_name, user_message, user_message_utc, chiri_response, chiri_response_utc, message_id FROM chat_table WHERE user_id = %s ORDER BY id DESC LIMIT %s""", (str(user_id), n))
                *conversation_history, = reversed(cursor.fetchall())
        conversation_history = []
        for user_id, display_name, user_message, user_message_utc, chiri_response, chiri_response_utc, message_id in conversation_history:
            entry = []
            entry.append(f"{display_name or '<no_display_name>'} (UTC: {user_message_utc}): {user_message}")
            if chiri_response:
                entry.append(f"Chiri Response (UTC: {chiri_response_utc}): {chiri_response}")
            if sum(map(get_tokens_count, conversation_history)) + sum(map(get_tokens_count, entry)) > max_tokens:
                break
            conversation_history.append("\n".join(entry))
    except Exception as exp:
        store_exception("get_conversation_history", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return conversation_history

def get_chat_history(max_tokens=4096, n=10) -> List[List[str]]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT user_id, display_name, user_message, user_message_utc, chiri_response, chiri_response_utc, message_id FROM chat_table ORDER BY id DESC LIMIT %s""", (n,))
                *conversation_history, = reversed(cursor.fetchall())
        conversation_history = []
        for user_id, display_name, user_message, user_message_utc, chiri_response, chiri_response_utc, message_id in conversation_history:
            entry = []
            entry.append(f"{display_name or '<no_display_name>'} (UTC: {user_message_utc}): {user_message}")
            if chiri_response:
                entry.append(f"Chiri Response (UTC: {chiri_response_utc}): {chiri_response}")
            if sum(map(get_tokens_count, conversation_history)) + sum(map(get_tokens_count, entry)) > max_tokens:
                break
            conversation_history.append("\n".join(entry))
    except Exception as exp:
        store_exception("get_chat_history", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return conversation_history

def get_memory(user_id: str) -> str:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT memory FROM memory_table WHERE user_id = %s ORDER BY id DESC LIMIT 1''', (user_id,))
                try:
                    memory = cursor.fetchone()[0]
                except TypeError:
                    memory = ""
    except Exception as exp:
        store_exception("get_memory", "\n".join(traceback.format_exception(exp)))
        return ""
    else:
        return memory

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

def store_chiri_response(chiri_response: str, message_id: uuid.UUID) -> bool:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""UPDATE chat_table SET chiri_response = %s, chiri_response_utc = %s WHERE message_id = %s""", (chiri_response, datetime.now(), str(message_id)))
    except Exception as exp:
        store_exception("store_chiri_response", "\n".join(traceback.format_exception(exp)))
        return False
    else:
        return True

async def requests_batch(endpoint, headers, method, payload):
    return await batched_requests(endpoint=endpoints, headers=headers, method=methods, payload=payloads)

def touch_app(filename: str="chiri_app.py"):
    with open(filename) as f:
        file_content = f.read()
    with open(filename, "w") as f:
        f.write(file_content.strip() if re.search(r"\n$", file_content) else file_content + "\n")
        
if __name__ == "__main__":
    pending_user_messages = get_pending_user_messages(args.minutes)
    # if len(pending_user_messages) > args.threshold:
    pending_payloads = []
    pending_message_ids = []
    for user_id, display_name, user_message, user_message_utc, user_type, message_id in pending_user_messages:
        # print(f"Responding to {display_name}")
        conversation_history = get_conversation_history(user_id)
        chat_history = get_chat_history()
        memory = get_memory(user_id)
        user_prompt = vtuber_prompt.format(display_name=display_name or "<no_display_name>", 
                                           user_message=user_message, 
                                           chat_history="\n\n".join(chat_history), 
                                           conversation_history="\n\n".join(conversation_history), 
                                           memory=memory)
        if len(pending_user_messages) > args.threshold:
            pending_payloads.append({"endpoint": os.environ['deepseek_api_endpoint'], "method": 'post', "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['deepseek_api_key']}"}, "payload": {"temperature": 0.7, "model": "deepseek-chat", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}})
            pending_message_ids.append(str(message_id))
        else:
            chiri_response = llm_openai(system_prompt, user_prompt)
            if not store_chiri_response(chiri_response, message_id):
                print(f"Response to {message_id} Failed.")
            else:
                print(f"Response to {message_id} Successful.")
    if pending_payloads:
        endpoints = []
        payloads = []
        headers = []
        methods = []
        for pending_payload in pending_payloads:
            endpoints.append(pending_payload['endpoint'])
            payloads.append(pending_payload['payload'])
            headers.append(pending_payload['headers'])
            methods.append(pending_payload['method'])
        responses = requests_batch(endpoint=endpoints, headers=headers, method=methods, payload=payloads)
        for response, message_id in zip(responses, pending_message_ids):
            try:
                if not store_chiri_response(response["choices"][0]["message"]["content"], message_id):
                    print(f"Response to {message_id} Failed.")
                else:
                    print(f"Response to {message_id} Successful.")
                    # subprocess.getoutput("touch chiri_app.py")
                    touch_app()
            except Exception as exp:
                print(f"Response to {message_id} Failed.")