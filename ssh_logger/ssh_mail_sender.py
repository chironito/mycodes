import psycopg2
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not Found."
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List
import traceback

def get_app_password(email_id: str=os.environ['mail_email_id']) -> str:
    try:
        with psycopg2.connect(dbname=os.environ['mail_postgres_dbname'], host=os.environ['postgres_server_ip'], 
                              port=os.environ['postgres_server_port'], user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute('''SELECT app_password FROM app_passwords WHERE email_id = %s ORDER BY id DESC LIMIT 1''', (email_id,))
                try:
                    return cursor.fetchone()[0] or ''
                except TypeError:
                    return ''
    except Exception as exp:
        return ''

def send_mail_alert(subject: str, body: str):
    sender_email = os.environ['mail_email_id']
    receiver_email = "sriguhanramesh@gmail.com"
    app_password = get_app_password(os.environ['mail_email_id'])  # Store securely!

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as exp:
        with open("send_mail_alert", "a+") as f:
            f.write("Failed to send mail: {exception}".format(exception="\n".join(traceback.format_exception(exp))))
        raise

def send_pending_alerts(days_delta: int=1):
    try:
        with psycopg2.connect(dbname=os.environ['postgres_dbname'], host=os.environ['postgres_server_ip'], 
                              port=os.environ['postgres_server_port'], user=os.environ['postgres_user'], 
                              password=os.environ['postgres_password']) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT line, ip_address, timestamp_utc FROM ssh_accepted_log WHERE timestamp_utc >= NOW() - INTERVAL '%s days' AND send_mail = true""", (days_delta,))
                connection_entries = cursor.fetchall()
                if connection_entries:
                    body = "\n".join(f"SSH Login from: {ip_address}\n{line}\nAt: {timestamp_utc}" for line, ip_address, timestamp_utc in connection_entries)
                    subject = f"SSH Alert at Chiro-Jump from: {connection_entries[0][1]}"
                    try:
                        send_mail_alert(subject, body)
                    finally:
                        cursor.execute('''UPDATE ssh_accepted_log SET send_mail = null WHERE send_mail = true''')
    except Exception as exp:
        with open("Alert Exception", "w") as f:
            f.write("\n".join(traceback.format_exception(exp)))

if __name__ == "__main__":
    send_pending_alerts()