import mysql.connector
from mysql.connector import Error
#import logging

# connection  string: root@127.0.0.1:3306 db: recorder

def fetch_records():
    try:
        connection = None
        dbconfig =  {"host":'127.0.0.1',       # Replace with your host, e.g., '127.0.0.1'
                "port":3306,            # Replace with your port, e.g., '3306'
                "database":'recorder',  # Replace with your database name
                "user":'root',    # Replace with your username
                "password":'password' # Replace with your password
            }   
        # Establish connection to the MySQL database
        try:
            print("Entered in try block")
            connection = mysql.connector.connect(pool_name='mypool',pool_size = 3, **dbconfig               
            )
            print("Connection established, leaving try block")
        except Error as e:
            print("Error in connection",e)
        print (connection)
        if connection.is_connected():
            print("Connected to the database")
            
            # Create a cursor to execute SQL queries
            cursor = connection.cursor()

            # Write your SELECT query
            query = "SELECT count(*) FROM cvs"  # Replace 'your_table' with your table name

            # Execute the query
            cursor.execute(query)

            # Fetch all rows from the executed query
            records = cursor.fetchall()

            print("Records retrieved:")
            for row in records:
                print(row)

    except Error as e:
        print(f"Error: {e}")

    finally:
        # Close the cursor and connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

# Call the function to fetch records
fetch_records()