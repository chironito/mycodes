import streamlit as st
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not found."
import os
import hashlib
import re
import uuid
import psycopg2
from datetime import datetime, timedelta
import traceback
import requests
from time import sleep
from typing import Union, Tuple, List, Dict

def store_exception(exception_type: str, exception_info: str, mail_sent=None) -> bool:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''INSERT INTO exceptions_table (exception_type, exception_info, timestamp_utc, mail_sent) VALUES (%s, %s, %s, %s)''', (exception_type, exception_info, datetime.now(), mail_sent))
    except Exception as exp:
        with open("store_exception_exception", "w") as f:
            f.write("\n".join(traceback.format_exception(exp)))
        return False
    else:
        return True

def create_user(email_id: str, password: str, display_name: str=None, description: str=None, user_type: int=None) -> Union[uuid.UUID, None]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                user_id = str(uuid.uuid4())
                cursor.execute('''INSERT INTO user_table (email_id, user_id, display_name, description, user_type, created_utc) VALUES (%s, %s, %s, %s, %s, %s)''', (email_id, user_id, display_name, description, user_type or 0, datetime.now()))
                cursor.execute('''INSERT INTO password_table (user_id, password_sha256, created_utc) VALUES (%s, %s, %s)''',\
                               (user_id, hashlib.sha256(password.encode()).hexdigest(), datetime.now()))
    except Exception as exp:
        store_exception("create_user_exception", "\n".join(traceback.format_exception(exp)))
        return
    else:
        return user_id

def update_user_info(email_id: str, password: str, display_name: str=None, description: str=None, user_type: int=None, email_id_modified=None) -> Union[bool, None]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT user_table.email_id, user_table.user_id, password_table.password_sha256
FROM user_table
INNER JOIN password_table
ON password_table.user_id = user_table.user_id
WHERE user_table.email_id = %s AND password_table.password_sha256 = %s''', \
                               (email_id, hashlib.sha256(password.encode()).hexdigest()))
                try:
                    user_id = cursor.fetchone()[1]
                except TypeError:
                    return False
                else:
                    if not user_id:
                        return False
                    updated_utc = datetime.now()
                    set_fields = [f"{field_name} = %s" for field_name, field_value in zip(("email_id", "display_name", "description", "user_type", "updated_utc"), (email_id_modified, display_name, description, user_type, updated_utc)) if field_value]
                    if set_fields:
                        cursor.execute("""UPDATE user_table SET {set_fields} WHERE user_id = {user_id}""".format(set_fields=", ".join(set_fields), user_id=f"'{user_id}'"), tuple(filter(bool, (email_id_modified, display_name, description, user_type, updated_utc))))
                    # with open("update_query", "w") as f:
                    #     f.write("""UPDATE user_table SET {set_fields} WHERE user_id = {user_id}""".format(set_fields=", ".join(set_fields), user_id=f"'{user_id}'"))
                    return True
    except Exception as exp:
        store_exception("update_user_exception", "\n".join(traceback.format_exception(exp)))
        return

def user_login(email_id: str, password_sha256: str) -> Union[Tuple[str], None, bool]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT user_table.user_id, password_table.password_sha256, user_table.display_name
FROM user_table
INNER JOIN password_table
ON user_table.user_id = password_table.user_id
WHERE user_table.email_id = %s AND password_table.password_sha256 = %s''', (email_id, password_sha256))
                try:
                    user_id, _, display_name = cursor.fetchone()
                except TypeError:
                    return False
    except Exception as exp:
        store_exception("user_login_exception", "\n".join(traceback.format_exception(exp)))
        return
    else:
        return (user_id, display_name)

def get_user_info(user_id: str) -> Union[List[Union[str, int]], None]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT email_id, display_name, description, user_type FROM user_table WHERE user_id = %s''', (user_id,))
                return cursor.fetchone()
    except Exception as exp:
        store_exception("get_user_info", "\n".join(traceback.format_exception(exp)))
        return

def get_recent_chiri_interactions(n=10) -> List[Dict[str, str]]:
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], 
                              host=os.environ['postgres_server_ip'], 
                              port=int(os.environ['postgres_server_port']), 
                              user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT display_name, user_message, chiri_response FROM chat_table ORDER BY id DESC LIMIT {n}'''.format(n=n))
                recent_chats = cursor.fetchall()
        chiri_interactions = []
        for display_name, user_message, chiri_response in reversed(recent_chats):
            chiri_interactions.append({"role": "user", "content": f"{display_name or 'nanashi'}: {user_message}"})
            if chiri_response:
                chiri_interactions.append({"role": "assistant", "content": f"Chiri: {chiri_response}"})
    except Exception as exp:
        store_exception("get_recent_chiri_interactions", "\n".join(traceback.format_exception(exp)))
        return []
    else:
        return chiri_interactions
           
st.set_page_config(layout="wide")

if st.session_state.get("retry_attempts") is None:
    st.session_state['retry_attempts'] = 0

if st.session_state['retry_attempts'] >= 3:
    st.error("Too many login attempts")
    raise Exception("Too many login attempts")

if not st.session_state.get("logged_in"):
    login_id = st.text_input("Login ID")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    if login_button:
        if login_id and password:
            if (login_id == os.environ['chiri_login']) and (hashlib.sha256(password.encode()).hexdigest() == os.environ['chiri_password_sha256']):
                st.session_state['logged_in'] = True
                st.write("Logged in Successfully!")
            else:
                st.write("Wrong login/password.")
                st.session_state['retry_attempts'] += 1
            sleep(1)
            st.rerun()
        else:
            st.write("Enter login id and password")
            st.session_state['retry_attempts'] += 1
else:
    if st.session_state.get("state_id") is None:
        st.session_state['state_id'] = 0
    if st.session_state['state_id'] == 0:
        email_id = st.text_input("Email ID")
        user_password = st.text_input("Password", type="password")
        user_login_button = st.button("Login")
        create_new_user = st.button("Create New User")
        if create_new_user:
            st.session_state['state_id'] = -1
            st.rerun()
        if user_login_button:
            if email_id and user_password:
                user_password_sha256 = hashlib.sha256(user_password.encode()).hexdigest()
                try:
                    st.session_state['user_id'], st.session_state['display_name'] = user_login(email_id, user_password_sha256)
                    st.write("Successfully Logged in!")
                    st.session_state['state_id'] = 1
                except TypeError:
                    st.error("Username/Password incorrect.")
                sleep(1)
                st.rerun()
    elif st.session_state['state_id'] == 1:
        st.header("Home Page")
        _, col = st.columns([9, 1])
        with col:
            update_profile = st.button("Update Profile")
        chiri_interactions = get_recent_chiri_interactions()
        for entry in chiri_interactions:
            with st.chat_message(entry['role']):
                st.write(entry['content'])
        user_message = st.chat_input("Message Chiri")
        if user_message:
            r = requests.post("http://localhost:8000/user_chat", 
                              headers={"api-key": 'nlblg7h57opgoz275x80bunr64on5rzod25rk9sletnudtb25q'}, 
                              json={'user_message': user_message, 
                                    'user_id': st.session_state['user_id'], 
                                    'display_name': st.session_state['display_name']})
            r.raise_for_status()
            st.rerun()
        if update_profile:
            st.session_state['state_id'] = -2
            st.rerun()
    elif st.session_state['state_id'] == -1:
        email_id_new = st.text_input("Email ID")
        user_password_new = st.text_input("Password", type="password")
        display_name_new = st.text_input("Display Name")
        description_new = st.text_area("Description", height=400)
        create_user_button = st.button("Create User")
        if create_user_button:
            if create_user(email_id_new, user_password_new, display_name_new, description_new):
                st.write("User Created!")
            else:
                st.write("User creation failed..")
            sleep(1)
            st.session_state['state_id'] = 0
            st.rerun()
    elif st.session_state['state_id'] == -2:
        _, col_2 = st.columns([9, 1])
        with col_2:
            screen_minus_2_back_button = st.button("Back")
            if screen_minus_2_back_button:
                st.session_state['state_id'] = 1
                st.rerun()
        try:
            email_id, display_name, description, user_type = get_user_info(st.session_state['user_id'])
        except TypeError:
            st.error("User not found!")
            raise Exception("User not found!")
        email_id_modified = st.text_input("Email ID", value=email_id)
        display_name_modified = st.text_input("Display Name", value=display_name)
        description_modified = st.text_input("Description", value=description)
        user_password_ = st.text_input("Password", type="password")
        update_profile_button = st.button("Update")
        if update_profile_button:
            if user_password_:
                update_status = update_user_info(email_id, user_password_, display_name_modified, description_modified, None, email_id_modified)
                if update_status:
                    st.write("Update Successful!")
                    st.session_state['state_id'] = 1
                    sleep(1)
                    st.rerun()
                elif update_status is None:
                    st.write("Updated Failed..")
                else:
                    st.write("Wrong User Password")
            else:
                st.write("Enter password to update profile")
