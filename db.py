import os, pymongo
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "indian_house_db"

def get_db():
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    return client[DB_NAME]

def check_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        return True, f"MongoDB v{client.server_info().get('version', '')} Connected"
    except Exception as e:
        return False, str(e)

check_mongo_connection = check_connection

def init_house_data(df, force_reset=False):
    col = get_db()["house_data"]
    count = col.count_documents({})
    if count > 0 and not force_reset:
        return count, f"Already initialized with {count} records. Duplicate import prevented."
    if force_reset:
        col.delete_many({})
    records = df.head(1000).to_dict(orient="records")
    for i, r in enumerate(records, 1):
        r.setdefault("ID", i)
    if records:
        col.insert_many(records)
    return col.count_documents({}), f"Successfully imported exactly {len(records)} records using insert_many()."

def get_total_records():
    return get_db()["house_data"].count_documents({})

get_total_house_records = get_total_records

def get_paginated_houses(page=1, page_size=20, search="", sort_by="ID", sort_asc=True):
    col = get_db()["house_data"]
    q = {}
    if search:
        reg = {"$regex": search, "$options": "i"}
        q["$or"] = [{"City": reg}, {"Locality": reg}, {"Property_Type": reg}, {"Furnished_Status": reg}]
    total = col.count_documents(q)
    order = pymongo.ASCENDING if sort_asc else pymongo.DESCENDING
    docs = list(col.find(q, {"_id": 0}).sort(sort_by, order).skip((page - 1) * page_size).limit(page_size))
    return docs, total

def insert_house_document(doc):
    return str(get_db()["house_data"].insert_one(doc).inserted_id)

def read_house_by_id(prop_id):
    return get_db()["house_data"].find_one({"ID": int(prop_id)}, {"_id": 0})

def update_house_document(prop_id, update_dict):
    res = get_db()["house_data"].update_one({"ID": int(prop_id)}, {"$set": update_dict})
    return res.matched_count, res.modified_count

def delete_house_document(prop_id):
    return get_db()["house_data"].delete_one({"ID": int(prop_id)}).deleted_count

def filter_properties(city="All", prop_type="All", min_price=0, max_price=600):
    q = {}
    if city and city != "All": q["City"] = city
    if prop_type and prop_type != "All": q["Property_Type"] = prop_type
    q["Price_in_Lakhs"] = {"$gte": float(min_price), "$lte": float(max_price)}
    return list(get_db()["house_data"].find(q, {"_id": 0}).limit(30)), q

def run_city_aggregation(min_bhk=1):
    pipe = [
        {"$match": {"BHK": {"$gte": int(min_bhk)}}},
        {"$group": {"_id": "$City", "avg_price": {"$avg": "$Price_in_Lakhs"}, "avg_sqft": {"$avg": "$Size_in_SqFt"}, "count": {"$sum": 1}}},
        {"$sort": {"avg_price": -1}}
    ]
    res = [{"City": d["_id"], "Avg Price (Lakhs)": round(d["avg_price"], 2), "Avg SqFt": round(d["avg_sqft"], 1), "Count": d["count"]} for d in get_db()["house_data"].aggregate(pipe)]
    return res, pipe

def run_type_aggregation():
    pipe = [
        {"$group": {"_id": "$Property_Type", "avg_price": {"$avg": "$Price_in_Lakhs"}, "count": {"$sum": 1}}},
        {"$sort": {"avg_price": -1}}
    ]
    res = [{"Property Type": d["_id"], "Avg Price (Lakhs)": round(d["avg_price"], 2), "Count": d["count"]} for d in get_db()["house_data"].aggregate(pipe)]
    return res, pipe

def save_prediction(input_data, p_lakhs, p_inr, model_name="Random Forest Regressor"):
    doc = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input": input_data, "input_features": input_data, "price_lakhs": p_lakhs, "price_inr": p_inr, "model": model_name}
    return str(get_db()["predictions"].insert_one(doc).inserted_id)

def get_predictions(limit=50):
    return list(get_db()["predictions"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))

def clear_predictions():
    return get_db()["predictions"].delete_many({}).deleted_count

def save_model_metrics(metrics):
    col = get_db()["model_metrics"]
    col.delete_many({})
    col.insert_one(dict(metrics, recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

def get_model_metrics():
    return list(get_db()["model_metrics"].find({}, {"_id": 0}))
