
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Include các endpoints từ Service Predict vào app
from .service.sample_predict.predict_routes import predict_router
app.include_router(predict_router)

# Include các endpoints từ Service Query DB vào app
from .service.sample_query_db.querydb_routes import querydb_router
app.include_router(querydb_router)

# Thiết lập các giá trị cho CORS
origins = ["*"]
# Ví dụ:
# origins = ["https://abc.com",
#           "113.112.0.1"]

# Áp dụng middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test to check if server is running or not
@app.get("/")
def read_root():
    return {"Hello, your project is running - test action all, expose to outside"}