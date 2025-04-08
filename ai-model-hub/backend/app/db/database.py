import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# 獲取數據庫URL，優先使用環境變量，否則默認為SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# 創建SQLAlchemy引擎
# connect_args 僅在SQLite時需要
engine_args = {}
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_args)

# 創建SessionLocal類
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 創建Base類，所有數據模型將繼承此類
Base = declarative_base()

# 數據庫會話依賴項
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """初始化數據庫，創建所有表"""
    # 在生產環境中，建議使用 Alembic 等遷移工具
    Base.metadata.create_all(bind=engine) 