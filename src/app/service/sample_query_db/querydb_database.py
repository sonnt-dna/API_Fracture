from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import local library
from .querydb_config import get_password, username, driver

username = username

driver = driver


def connect_DB(server: str, database: str):
    password = get_password()
    params = 'Driver=' + driver + ';Server=' + server + ',1433;Database=' + database + ';Uid={' + username + '};Pwd={' + \
             password + '};Encrypt=yes;TrustServerCertificate=no;Connection ' \
                        'Timeout=30;Authentication=ActiveDirectoryPassword'

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal
