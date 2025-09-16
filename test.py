# import sqlite3

# # Replace with path to your murder_mystery.db file
# db_path = "sql-murder-mystery.db"

# # Connect
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # Run test query
# cursor.execute("""
# SELECT name, address_street_name, address_number 
# FROM person 
# LIMIT 10;
# """)

# tables = cursor.fetchall()

# for t in tables:
#     print(t)


from src.utils.sql_handler import SqlHandler

sql=SqlHandler()
query="SELECT * FROM person WHERE name LIKE 'J%';"
print(sql.execute_command(query))