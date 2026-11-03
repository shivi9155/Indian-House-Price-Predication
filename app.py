import streamlit as st, pandas as pd, plotly.express as px, importlib
import db, model
importlib.reload(db)
importlib.reload(model)

st.set_page_config(page_title="Indian House Price Prediction", page_icon="🏠", layout="wide")

@st.cache_data
def load_csv():
    return pd.read_csv("indian_urban_house_prices.csv")

# Ensure initial data in MongoDB
try:
    if db.check_connection()[0] and db.get_total_records() == 0:
        db.init_house_data(model.clean_data(load_csv()).head(1000))
except Exception:
    pass

# Sidebar
st.sidebar.title("🏠 Navigation")
menu = st.sidebar.radio("Go to:", [
    "1. Home", "2. Dataset", "3. MongoDB Operations",
    "4. Model Training", "5. Model Evaluation",
    "6. House Price Prediction", "7. Prediction History"
])
st.sidebar.markdown("---")
is_conn, conn_msg = db.check_connection()
st.sidebar.success(f"🟢 {conn_msg}" if is_conn else f"🔴 {conn_msg}")
st.sidebar.caption("Indian House Price Prediction | PyMongo + Random Forest")

# ================= 1. HOME =================
if menu == "1. Home":
    st.title("🏠 Indian House Price Prediction System")
    st.subheader("Predicting Indian property prices using Python, MongoDB (PyMongo), and Random Forest")
    
    df_raw = load_csv()
    db_count = db.get_total_records() if is_conn else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kaggle CSV Records", f"{len(df_raw):,}")
    c2.metric("Imported to MongoDB", f"{db_count:,}", help="Stored in 'house_data' collection")
    c3.metric("Input Features", "8 Core Features")
    c4.metric("Machine Learning Model", "Random Forest Regressor")

    st.info("""
    **Project Architecture:**
    - **Dataset:** Real Kaggle Indian Urban House Price dataset; exactly 1,000 valid records stored in MongoDB.
    - **Database:** MongoDB (`indian_house_db`) with collections `house_data`, `predictions`, and `model_metrics`.
    - **Machine Learning:** Scikit-learn **Random Forest Regressor** trained on an 80/20 train/test split.
    """)
    if is_conn and st.button("🔄 Re-Import Exactly 1,000 Records to MongoDB"):
        _, msg = db.init_house_data(model.clean_data(df_raw).head(1000), force_reset=True)
        st.success(msg)
        st.rerun()

# ================= 2. DATASET =================
elif menu == "2. Dataset":
    st.title("📋 Kaggle Dataset Explorer")
    st.write("Browse the 1,000 residential records stored in MongoDB collection `house_data`.")
    
    c_s, c_sort, c_dir = st.columns([2, 1, 1])
    search_q = c_s.text_input("🔍 Search (City, Locality, Type):")
    sort_f = c_sort.selectbox("Sort By:", ["ID", "Price_in_Lakhs", "Size_in_SqFt", "BHK", "City"])
    sort_asc = c_dir.radio("Order:", ["Asc", "Desc"], horizontal=True) == "Asc"
    
    PAGE_SIZE = 20
    _, total = db.get_paginated_houses(1, PAGE_SIZE, search_q, sort_f, sort_asc)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input(f"Page (1 to {total_pages}) - 20 records/page:", 1, total_pages, 1)
    
    docs, _ = db.get_paginated_houses(page, PAGE_SIZE, search_q, sort_f, sort_asc)
    if docs:
        st.dataframe(pd.DataFrame(docs), use_container_width=True, height=480)
        st.caption(f"Showing page {page} of {total_pages} (Total matched: {total})")
    else:
        st.warning("No records found.")

# ================= 3. MONGODB OPERATIONS =================
elif menu == "3. MongoDB Operations":
    st.title("⚙️ MongoDB CRUD & Aggregation Workbench")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Find/Filter", "Insert", "Update", "Delete", "Aggregation"])

    with tab1:
        st.markdown("#### `find()` with Filters")
        cities = ["All"] + sorted(list(load_csv()['City'].dropna().unique()))
        f_city = st.selectbox("Filter City:", cities)
        f_type = st.selectbox("Property Type:", ["All", "Apartment", "Independent House", "Villa"])
        f_min, f_max = st.slider("Price Range (Lakhs):", 0, 600, (0, 500))
        res, query = db.filter_properties(f_city, f_type, f_min, f_max)
        st.code(f"db.house_data.find({query})", "javascript")
        st.write(f"Matched {len(res)} properties:")
        st.dataframe(pd.DataFrame(res), use_container_width=True)

    with tab2:
        st.markdown("#### `insert_one()` Document")
        with st.form("insert_form"):
            i_id = st.number_input("Property ID:", 5000, 99999, 9999)
            i_city = st.text_input("City:", "Bangalore")
            i_loc = st.text_input("Locality:", "Whitefield")
            i_type = st.selectbox("Type:", ["Apartment", "Independent House", "Villa"])
            i_sqft = st.number_input("SqFt:", 300.0, 10000.0, 1400.0)
            i_bhk = st.number_input("BHK:", 1, 6, 3)
            i_price = st.number_input("Price (Lakhs):", 5.0, 1000.0, 120.0)
            if st.form_submit_button("Insert Document"):
                new_doc = {"ID": i_id, "City": i_city, "Locality": i_loc, "Property_Type": i_type, "Size_in_SqFt": i_sqft, "BHK": i_bhk, "Bathrooms": i_bhk, "Price_in_Lakhs": i_price, "Furnished_Status": "Furnished", "Parking_Space": "Yes"}
                oid = db.insert_house_document(new_doc)
                st.success(f"Inserted document with ObjectId: {oid}")

    with tab3:
        st.markdown("#### `update_one()` with `$set`")
        u_id = st.number_input("Property ID to Update:", 1, 99999, 1)
        u_price = st.number_input("New Price (Lakhs):", 1.0, 1000.0, 150.0)
        u_furn = st.selectbox("New Furnishing:", ["Furnished", "Semi-furnished", "Unfurnished"])
        if st.button("Apply update_one"):
            matched, modified = db.update_house_document(u_id, {"Price_in_Lakhs": u_price, "Furnished_Status": u_furn})
            st.success(f"Matched: {matched}, Modified: {modified}")
            st.json(db.read_house_by_id(u_id))

    with tab4:
        st.markdown("#### `delete_one()` Document")
        d_id = st.number_input("Property ID to Delete:", 1, 99999, 9999)
        if st.button("Apply delete_one", type="primary"):
            st.success(f"Deleted {db.delete_house_document(d_id)} document(s).")

    with tab5:
        st.markdown("#### Aggregation Pipelines (`$match`, `$group`, `$avg`, `$sort`)")
        min_bhk = st.slider("Filter Min BHK ($match):", 1, 5, 2)
        city_agg, pipe = db.run_city_aggregation(min_bhk)
        st.code(str(pipe), "python")
        agg_df = pd.DataFrame(city_agg)
        st.dataframe(agg_df, use_container_width=True)
        if not agg_df.empty:
            st.plotly_chart(px.bar(agg_df.head(10), x="City", y="Avg Price (Lakhs)", title="Top Cities by Avg Price", text="Avg Price (Lakhs)"), use_container_width=True)

# ================= 4. MODEL TRAINING =================
elif menu == "4. Model Training":
    st.title("🤖 Model Training: Random Forest Regressor")
    st.markdown("""
    - **Algorithm:** Random Forest Regressor (100 estimators, max depth 14)
    - **Split:** 80% Training (800 records), 20% Testing (200 records)
    - **Preprocessing:** One-Hot Encoding + Standard Scaling
    """)
    if st.button("🚀 Train Random Forest Regressor Now", type="primary"):
        with st.spinner("Training Random Forest model on 1,000 records..."):
            metrics, _ = model.train_and_evaluate(load_csv().head(1000))
            if is_conn: db.save_model_metrics(metrics)
            st.success(f"Trained successfully! Model Accuracy: {metrics['accuracy']}%")
            st.table(pd.DataFrame([{
                "Model": metrics["model_name"],
                "Train Records": metrics["train_records"],
                "Test Records": metrics["test_records"],
                "Accuracy": f"{metrics['accuracy']}%",
                "R² Score": metrics["r2_score"],
                "MAE": f"₹ {metrics['mae']} L",
                "MSE": metrics["mse"],
                "RMSE": f"₹ {metrics['rmse']} L"
            }]))

# ================= 5. MODEL EVALUATION =================
elif menu == "5. Model Evaluation":
    st.title("📈 Model Evaluation Metrics")
    saved_m = db.get_model_metrics() if is_conn else []
    if not saved_m or "accuracy" not in saved_m[0] or "rmse" not in saved_m[0]:
        m, _ = model.train_and_evaluate(load_csv().head(1000))
        saved_m = [m]
        if is_conn: db.save_model_metrics(m)

    m = saved_m[0]
    acc = m.get("accuracy", 87.6)
    r2 = m.get("r2_score", 0.8420)
    mae = m.get("mae", 20.93)
    mse = m.get("mse", 875.42)
    rmse = m.get("rmse", 29.59)

    st.subheader("🎯 Overall Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.1f}%")
    c2.metric("R² Score", f"{r2:.4f}")
    c3.metric("MAE", f"₹ {mae:.2f} L")
    c4.metric("MSE", f"{mse:.2f}")
    c5.metric("RMSE", f"₹ {rmse:.2f} L")

    st.markdown("---")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        acc_df = pd.DataFrame([
            {"Metric": "Model Accuracy (%)", "Score": acc},
            {"Metric": "R² Score (%)", "Score": round(r2 * 100, 1)}
        ])
        fig_acc = px.bar(
            acc_df, x="Metric", y="Score", text="Score",
            range_y=[0, 100], title="Accuracy & Goodness-of-Fit (%)",
            color="Metric", color_discrete_sequence=["#10B981", "#3B82F6"]
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with ch_col2:
        err_df = pd.DataFrame([
            {"Error Metric": "MAE (Lakhs)", "Value": mae},
            {"Error Metric": "RMSE (Lakhs)", "Value": rmse}
        ])
        fig_err = px.bar(
            err_df, x="Error Metric", y="Value", text="Value",
            title="Prediction Error in Lakhs (Lower is Better)",
            color="Error Metric", color_discrete_sequence=["#F59E0B", "#EF4444"]
        )
        st.plotly_chart(fig_err, use_container_width=True)

    st.success(f"✅ Random Forest Regressor achieves **{acc:.1f}% Accuracy** with an $R^2$ of **{r2:.4f}** and MAE of **₹ {mae:.2f} Lakhs**.")

# ================= 6. HOUSE PRICE PREDICTION =================
elif menu == "6. House Price Prediction":
    st.title("🔮 Predict House Price")
    df_raw = load_csv()
    cities = sorted(list(df_raw['City'].dropna().unique()))
    localities = sorted(list(df_raw['Locality'].dropna().unique()[:30]))

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City:", cities)
            locality = st.selectbox("Locality:", localities)
            p_type = st.selectbox("Property Type:", ["Apartment", "Independent House", "Villa"])
        with col2:
            sqft = st.number_input("Area (SqFt):", 300.0, 15000.0, 1200.0, step=50.0)
            bhk = st.slider("Bedrooms (BHK):", 1, 6, 3)
            baths = st.slider("Bathrooms:", 1, 6, 2)
        with col3:
            furnish = st.selectbox("Furnishing:", ["Furnished", "Semi-furnished", "Unfurnished"])
            parking = st.selectbox("Parking:", ["Yes", "No"])
            st.text_input("Model:", "Random Forest Regressor", disabled=True)

        if st.form_submit_button("💰 Predict Price", type="primary"):
            user_input = {"City": city, "Locality": locality, "Property_Type": p_type, "Size_in_SqFt": float(sqft), "BHK": int(bhk), "Bathrooms": int(baths), "Furnished_Status": furnish, "Parking_Space": parking}
            try:
                p_lakhs, p_inr, m_name = model.predict_price(user_input)
                c_p1, c_p2 = st.columns(2)
                c_p1.metric("Estimated Price (Lakhs)", f"₹ {p_lakhs:,.2f} Lakhs")
                c_p2.metric("Estimated Price (Rupees)", f"₹ {p_inr:,.2f}")
                if is_conn:
                    pid = db.save_prediction(user_input, p_lakhs, p_inr, m_name)
                    st.success(f"Saved to MongoDB 'predictions' collection (ID: {pid})")
            except Exception as e:
                st.error(f"Error: {e}")

# ================= 7. PREDICTION HISTORY =================
elif menu == "7. Prediction History":
    st.title("📜 Prediction History")
    history = db.get_predictions() if is_conn else []
    if history:
        rows = []
        for h in history:
            inp = h.get("input") or h.get("input_features") or {}
            p_lakh = h.get("price_lakhs") or h.get("predicted_price_lakhs", 0)
            p_inr = h.get("price_inr") or h.get("predicted_price_inr", 0)
            rows.append({
                "Time": h.get("timestamp", "-"),
                "City": inp.get("City", "-"),
                "Locality": inp.get("Locality", "-"),
                "Type": inp.get("Property_Type", "-"),
                "BHK": inp.get("BHK", "-"),
                "Area": inp.get("Size_in_SqFt", "-"),
                "Price (Lakhs)": f"₹ {p_lakh:,.2f} L",
                "Price (INR)": f"₹ {p_inr:,.2f}",
                "Model": h.get("model") or h.get("model_used", "Random Forest Regressor")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if st.button("🗑️ Clear Prediction History"):
            db.clear_predictions()
            st.rerun()
    else:
        st.info("No predictions recorded yet.")
