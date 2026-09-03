import uuid
import time
import sys
import os

# Local import
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from src.utils.logger import logger
from src.utils.common import check_inbox, send_email

EMAIL_ENV_VARS = {
    "sender_email": "MAPLEBOT_EMAIL_SENDER",
    "password": "MAPLEBOT_EMAIL_PASSWORD",
    "receiver_email": "MAPLEBOT_EMAIL_RECEIVER",
}


def load_email_settings():
    """Load email credentials without storing secrets in source control."""
    settings = {
        key: os.environ.get(env_name, "").strip()
        for key, env_name in EMAIL_ENV_VARS.items()
    }
    missing = [
        EMAIL_ENV_VARS[key]
        for key, value in settings.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required email environment variables: " + ", ".join(missing)
        )
    return settings


def wait_for_reply(email_addr, password, token,
                   timeout_sec=90, search_interval=10):
    '''
    Wait for user reply for a while
    Checks inbox for a reply containing the token.
    Times out after `timeout_sec` seconds.
    '''
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        reply = check_inbox(email_addr, password, token)
        logger.info(f"User replied: {reply}")
        if reply and reply[0] in {"1", "2", "3", "4"}:
            return int(reply[0])

        logger.info(f"[{int(time.time() - start_time)}s] No valid reply yet. Waiting...")
        time.sleep(search_interval)

    logger.error(f"[wait_for_reply] Timeout: "
                 f"No valid reply received in {time.time() - start_time} seconds.")
    return None

if __name__ == "__main__":
    email_settings = load_email_settings()

    token = uuid.uuid4().hex[:8]  # Generate 8-character token
    send_email(
        email_settings["sender_email"],
        email_settings["password"],
        email_settings["receiver_email"],
        f"[MS Bot] Help me pass the test({token})",
        "Please directly reply this email\nType '1', '2', '3', or '4'",
        "screenshot/rune_detected_2025-06-23_03-47-05.png",
    )

    user_reply = wait_for_reply(
        email_settings["sender_email"], email_settings["password"], token
    )
    if user_reply:
        logger.info(f"User selected: {user_reply}", )
    else:
        logger.info("Proceeding without user response.")
