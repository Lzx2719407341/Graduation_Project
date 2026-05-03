# core/history_manager.py
# 历史记录管理模块，负责存储和查询检测历史记录，使用SQLite数据库

import sqlite3
import pandas as pd
from datetime import datetime
import os

# HistoryManager类，负责管理检测历史记录，使用SQLite数据库存储
class HistoryManager:
    def __init__(self, db_path="data/history.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # 初始化数据库，创建表格
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS records 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 time TEXT, filename TEXT, pole_count INTEGER, results TEXT)''')

    # 添加记录，包含文件名、杆数和结果字符串
    def add_record(self, filename, pole_count, results_str):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO records (time, filename, pole_count, results) VALUES (?, ?, ?, ?)",
                         (time_str, filename, pole_count, results_str))

    # 获取所有记录，返回一个DataFrame
    def get_all(self):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM records ORDER BY id DESC", conn)
