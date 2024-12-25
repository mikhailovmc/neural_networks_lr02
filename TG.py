from telethon.sync import TelegramClient
from telethon.tl.functions.messages import *
from telethon.tl.types import InputPeerEmpty
from dateutil.parser import parse

last_date = None
size_chats = 300

api_id = 0
api_hash = ''
phone = ''


def get_connection():
    client = TelegramClient(phone, api_id, api_hash, system_version="4.16.30-vxCUSTOM")
    client.connect()
    if not client.is_user_authorized():
        client.send_code_request(phone)
        me = client.sign_in(phone, input('Enter code: '))

    return client


def get_posts(client, channel_id, date, count, offset_id):

    result = client(GetDialogsRequest(
        offset_date=None,
        offset_id=0,
        offset_peer=InputPeerEmpty(),
        limit=size_chats,
        hash=0
    ))

    channel_entity = None
    chats = []

    chats.extend(result.chats)
    for chat in chats:
        if chat.id == int(channel_id):
            channel_entity = chat
            break

    if (int(count) < 0) | (channel_entity is None):
        return None

    posts = client(GetHistoryRequest(
        peer=channel_entity,
        limit=int(count),
        offset_date=parse(date),
        offset_id=offset_id,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0))

    return posts


def close_connection(client):
    client.disconnect()