import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from my_project.rag.core.config import settings


def get_checkpointer():
    # ensure checkpoint folder exists
    settings.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        database=settings.CHECKPOINT_PATH,
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn=conn)
    return checkpointer