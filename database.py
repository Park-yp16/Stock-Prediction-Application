import sqlite3 as sql

conn = sql.connect('database.db')
print("Opened database successfully")

cursor = conn.cursor()

sqlQuery1 = """ CREATE TABLE IF NOT EXISTS users (
    userID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)"""
cursor.execute(sqlQuery1)

sqlQuery2 = """ CREATE TABLE IF NOT EXISTS models (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	lookup_step INTEGER NOT NULL,
	stock TEXT NOT NULL,
	n_steps INTEGER NOT NULL,
	updateDate TEXT NOT NULL, 
	modelName TEXT NOT NULL)
"""
cursor.execute(sqlQuery2)

print("Table created successfully")
conn.close()