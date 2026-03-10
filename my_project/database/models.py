from my_project.database.database import get_connection


def create_users_table():
    conn = get_connection()
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
    conn.close()
    print("Table Created")


if __name__ == "__main__":
    create_users_table()