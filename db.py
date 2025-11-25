import psycopg2
from config import DB_URL

conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()
