from my_project.database.database import get_connection

def create_user(gmail, password, name):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO users (gmail, password, name)
    VALUES (%s, %s, %s)
    """

    cursor.execute(query, (gmail, password, name))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message": "User created successfully"}


def get_user(gmail):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT gmail, password, name
    FROM users
    WHERE gmail = %s
    """

    cursor.execute(query, (gmail,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    return user