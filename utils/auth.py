import sqlite3

def create_users_table():

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
    username TEXT,
    password TEXT,
    email TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_user(username,password,email):

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO users(username,password,email) VALUES(?,?,?)",
        (username,password,email)
    )

    conn.commit()
    conn.close()


def login_user(username,password):

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute(
    "SELECT * FROM users WHERE username=? AND password=?",
    (username,password)
    )

    data = c.fetchone()

    conn.close()

    return data