"""Alert channels. Add a new channel by adding a Notifier class (Open/Closed)."""
from __future__ import annotations

import json
import smtplib
import urllib.parse
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from typing import List

from .interfaces import Alert, Notifier

ICON = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[HIGH]"}


def alert_text(a: Alert) -> str:
    when = datetime.fromtimestamp(a.ts).strftime("%Y-%m-%d %H:%M:%S")
    return (f"EnerGuard alert {ICON.get(a.severity, '')} on {a.device_id} at {when}\n"
            f"Likely cause: {a.cause}\n"
            f"Why: {a.diagnosis}\n"
            f"Next step: {a.action}\n"
            f"Alert ID: {a.alert_id}")


class ConsoleNotifier(Notifier):
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.sent: List[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)
        if not self.quiet:
            print("\n" + "=" * 70 + "\n" + alert_text(alert) + "\n" + "=" * 70)


class TelegramNotifier(Notifier):
    """Sends the alert with two buttons: Real fault / False alarm.
    Run `python -m energuard telegram-feedback` to save the button presses."""

    def __init__(self, bot_token: str, chat_id: str):
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id

    def send(self, alert: Alert) -> None:
        keyboard = {"inline_keyboard": [[
            {"text": "Real fault", "callback_data": f"fb|{alert.alert_id}|{alert.device_id}|fault"},
            {"text": "False alarm", "callback_data": f"fb|{alert.alert_id}|{alert.device_id}|false_alarm"}]]}
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": alert_text(alert),
                                       "reply_markup": json.dumps(keyboard)}).encode()
        try:
            urllib.request.urlopen(self.url, data=data, timeout=10).read()
        except Exception as e:  # never let a network problem stop monitoring
            print(f"[telegram] could not send alert: {e}")


class EmailNotifier(Notifier):
    def __init__(self, smtp_host: str, smtp_port: int, user: str, password: str, to: str):
        self.host, self.port, self.user, self.password, self.to = smtp_host, smtp_port, user, password, to

    def send(self, alert: Alert) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"EnerGuard {alert.severity.upper()} alert: {alert.cause} ({alert.device_id})"
        msg["From"], msg["To"] = self.user, self.to
        msg.set_content(alert_text(alert))
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as s:
                s.starttls()
                s.login(self.user, self.password)
                s.send_message(msg)
        except Exception as e:
            print(f"[email] could not send alert: {e}")


class MultiNotifier(Notifier):
    def __init__(self, notifiers: List[Notifier]):
        self.notifiers = notifiers

    def send(self, alert: Alert) -> None:
        for n in self.notifiers:
            n.send(alert)
