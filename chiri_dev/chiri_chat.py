import falcon
import psycopg2
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not found."
import os
import re
from datetime import datetime, timedelta
import traceback
import hashlib
from typing import Union, Dict, List, Tuple
import uuid

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

def store_user_message(user_id: str, user_message: str, display_name: str=None, user_type: int=0) -> Union[uuid.UUID, None]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            message_id = uuid.uuid4()
            with connection.cursor() as cursor:
                cursor.execute('''INSERT INTO chat_table (user_id, display_name, user_message, user_message_utc, user_type, message_id) VALUES (%s, %s, %s, %s, %s, %s)''', (user_id, display_name, user_message, datetime.now(), user_type, str(message_id)))
    except Exception as exp:
        store_exception("store_user_message", "\n".join(traceback.format_exception(exp)))
        return
    else:
        return message_id

def get_chiri_response(message_id: uuid.UUID) -> Union[str, None]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], user=os.environ['postgres_user'], password=os.environ['postgres_password'], 
                              host=os.environ['postgres_server_ip'], port=int(os.environ['postgres_server_port'])) as connection:
            message_id = uuid.uuid4()
            with connection.cursor() as cursor:
                cursor.execute('''SELECT chiri_response FROM chat_table WHERE message_id = %s''', (str(message_id),))
                chiri_response = cursor.fetchone()
    except Exception as exp:
        store_exception("get_chiri_response", "\n".join(traceback.format_exception(exp)))
        return "-"
    else:
        try:
            return chiri_response[0]
        except TypeError:
            return ""
    
class UserChatResource:
    def on_post(self, req, resp):
        password = req.get_header("api-key")
        if not password:
            resp.text = "Api Key not found."
            resp.status = falcon.HTTP_401
            return
        if hashlib.sha256(password.encode()).hexdigest() != os.environ['chiri_password_sha256']:
            resp.text = "Invalid Api Key."
            resp.status = falcon.HTTP_401
            return
        request_dict = req.media
        try:
            user_message = request_dict["user_message"]
        except (KeyError, TypeError):
            resp.text = "No user message."
            resp.status = falcon.HTTP_422
            return
        try:
            user_id = request_dict["user_id"]
        except KeyError:
            resp.text = "No user id."
            resp.status = falcon.HTTP_422
            return
        user_type = request_dict.get("user_type", 0)
        display_name = request_dict.get("display_name")
        message_id = store_user_message(user_id, user_message, display_name, user_type)
        if message_id is not None:
            resp.status = falcon.HTTP_201
            resp.media = {"response": "User message queued.", "message_id": str(message_id)}
        else:
            resp.status = falcon.HTTP_424
            resp.text = {"response": "User message queueing failed.", "message_id": None}
        return
        
    def on_get(self, req, resp):
        password = req.get_header("api-key")
        if not password:
            resp.text = "Api Key not found."
            resp.status = falcon.HTTP_401
            return
        if hashlib.sha256(password.encode()).hexdigest() != os.environ['chiri_password_sha256']:
            resp.text = "Invalid Api Key."
            resp.status = falcon.HTTP_401
            return
        message_id = req.get_param("message_id")
        chiri_response = get_chiri_response(message_id)
        if chiri_response is None:
            resp.status = falcon.HTTP_202
            resp.media = {"response": None, "remarks": "Awaiting Chiri Response."}
        elif chiri_response == "":
            resp.status = falcon.HTTP_404
            resp.media = {"response": "", "remarks": "Message ID not found."}
        elif chiri_response == "-":
            resp.status = falcon.HTTP_424
            resp.media = {"response": None, "remarks": "DB error."}
        else:
            resp.status = falcon.HTTP_200
            resp.media = {"response": chiri_response, "remarks": None}
        return

class PresentResource:
    def on_get(self, req, resp):
        resp.text = "Present!"
        resp.status = falcon.HTTP_200
        return
        
app = falcon.App()
app.add_route("/user_chat", UserChatResource())
app.add_route("/", PresentResource())
