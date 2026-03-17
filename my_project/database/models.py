from mysql.connector.connection import MySQLConnection


def create_users_table(conn : MySQLConnection):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            gmail VARCHAR(255) PRIMARY KEY,
            password VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL
        )
    """)

    conn.commit()

    cursor.close()
    print("Table Created")


if __name__ == "__main__":
    from my_project.database.database import get_connection
    conn = get_connection()
    create_users_table(conn)
    conn.close()