import sqlite3
import io
import numpy as np
from Encoding.encoder import encode

"""
This section creates an array datatype for use in sqlite3
"""


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)

    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)

    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


"""
This section has the most common functions to interact with the database
"""


def create_database():
    with sqlite3.connect("chatbot.db", detect_types=sqlite3.PARSE_DECLTYPES) as connection:
        c = connection.cursor()
        sql = """create table if not exists Training
        (id integer primary key autoincrement, context text, response text, context_emb array, response_emb array)"""
        c.execute(sql)
        connection.commit()


create_database()


def insert(values):
    with sqlite3.connect("chatbot.db", detect_types=sqlite3.PARSE_DECLTYPES) as connection:
        c = connection.cursor()
        sql = "insert into Training (context, response, context_emb, response_emb) values (?, ?, ?, ?)"
        c.executemany(sql, values)
        connection.commit()


def retrieve(start, end):
    num_rows = end - start
    with sqlite3.connect("chatbot.db", detect_types=sqlite3.PARSE_DECLTYPES) as connection:
        c = connection.cursor()
        sql = f"select context_emb, response_emb from Training limit {num_rows} offset {start}"
        c.execute(sql)
        candidates = c.fetchall()

        return candidates


def get_length():
    """
    Function to get number of utterances in db and return value
    :return: integer number of utterances
    """
    with sqlite3.connect("chatbot.db", detect_types=sqlite3.PARSE_DECLTYPES) as connection:
        c = connection.cursor()
        sql = f"select count(*) from Training"
        c.execute(sql)
        count = c.fetchone()[0]

        return count


def get_lowest_id():
    """
    Function to get the lowest id from the Training dataset (in case autoincrement has not started at 0)
    :return: integer of min(ids)
    """
    with sqlite3.connect("chatbot.db", detect_types=sqlite3.PARSE_DECLTYPES) as connection:
        c = connection.cursor()
        sql = f"select min(id) from Training"
        c.execute(sql)
        minimum = c.fetchone()[0]

        return minimum


def preprocess(contexts, responses):
    try:
        array_contexts = encode(contexts)
        array_responses = encode(responses)
        values = []
        for i in range(len(contexts)):
            values.append((contexts[i], responses[i], array_contexts[i], array_responses[i]))

        return values

    except:
        return None
