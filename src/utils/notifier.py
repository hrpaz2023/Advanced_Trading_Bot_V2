# src/utils/notifier.py
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Notifier:
    def __init__(self, config_path='configs/global_config.json'):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.config = config['email_notifications']
            self.enabled = self.config.get('enabled', False)
            
            if self.enabled:
                self.smtp_server = self.config['smtp_server']
                self.smtp_port = self.config['smtp_port']
                self.sender_email = self.config['sender_email']
                self.sender_password = self.config['sender_password']
                self.receiver_email = self.config['receiver_email']

        except (FileNotFoundError, KeyError) as e:
            print(f"Advertencia: No se pudo cargar la configuración de notificaciones. Error: {e}")
            self.enabled = False

    def send_notification(self, subject, body):
        if not self.enabled:
            return

        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            print("✓ Notificación por correo enviada.")
        except Exception as e:
            print(f"✗ Error al enviar la notificación por correo: {e}")