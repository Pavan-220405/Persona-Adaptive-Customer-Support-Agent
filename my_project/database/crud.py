from mysql.connector.connection import MySQLConnection


def create_user(conn: MySQLConnection, gmail: str, password: str, name: str):
    cursor = conn.cursor()

    query = """
    INSERT INTO users (gmail, password, name)
    VALUES (%s, %s, %s)
    """

    try:
        cursor.execute(query, (gmail, password, name))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()

    return {"message": "User created successfully"}


def get_user(conn: MySQLConnection, gmail: str) -> dict | None:
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT gmail, password, name
    FROM users
    WHERE gmail = %s
    """


    try:
        cursor.execute(query, (gmail,))
        user = cursor.fetchone()
        return user 
    finally:
        cursor.close()