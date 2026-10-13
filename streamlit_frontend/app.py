"""
Streamlit Frontend — Vehicle Management and Price Prediction System

Interactive UI for managing vehicles and making ML predictions.
All operations communicate with the FastAPI backend via HTTP.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import streamlit as st
import requests
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10  # seconds

# ── Page Configuration ────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Management & Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-price {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    .category-budget { background-color: #48BB78; color: white; }
    .category-midrange { background-color: #4299E1; color: white; }
    .category-premium { background-color: #9F7AEA; color: white; }
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────

def api_request(method, endpoint, json_data=None, show_error=True):
    """Make an API request to the backend."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json_data,
            timeout=REQUEST_TIMEOUT,
        )
        return response
    except requests.ConnectionError:
        if show_error:
            st.error(f"❌ Cannot connect to backend at {API_BASE_URL}. Please ensure the FastAPI server is running.")
        return None
    except requests.Timeout:
        if show_error:
            st.error("⏱️ Request timed out. The backend may be overloaded.")
        return None
    except requests.RequestException as e:
        if show_error:
            st.error(f"❌ Request failed: {str(e)}")
        return None


def check_api_health():
    """Check if the API backend is reachable."""
    response = api_request("GET", "/", show_error=False)
    if response and response.status_code == 200:
        return True
    return False


def format_inr(amount):
    """Format a number as Indian Rupees."""
    if amount is None:
        return "N/A"
    amount = float(amount)
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric"]
TRANSMISSION_TYPES = ["Manual", "Automatic"]

COMMON_MAKES = [
    "Maruti Suzuki", "Hyundai", "Tata", "Honda", "Toyota",
    "Mahindra", "Kia", "Ford", "Volkswagen", "Skoda",
    "BMW", "Mercedes-Benz", "Audi"
]


# ── Sidebar Navigation ───────────────────────────────────────────

st.sidebar.markdown("## 🚗 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📋 Vehicle Management", "💰 Price Prediction",
     "🏷️ Category Classification", "ℹ️ About"],
    label_visibility="collapsed"
)

# API health indicator
api_healthy = check_api_health()
if api_healthy:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.caption(f"Backend: {API_BASE_URL}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sahishnakumaar S S**")
st.sidebar.markdown("Roll No: 24ADR134")


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">🚗 Vehicle Management & Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dashboard — Overview of your vehicle records</p>', unsafe_allow_html=True)

    if not api_healthy:
        st.warning("⚠️ Backend API is not reachable. Dashboard data is unavailable.")
        st.info(f"Please start the FastAPI backend and ensure it's running at: **{API_BASE_URL}**")
    else:
        response = api_request("GET", "/vehicles")
        if response and response.status_code == 200:
            data = response.json()
            vehicles = data.get("vehicles", [])
            total = data.get("total", 0)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🚗 Total Vehicles", total)

            if vehicles:
                df = pd.DataFrame(vehicles)

                fuel_counts = df["fuel_type"].value_counts() if "fuel_type" in df.columns else pd.Series(dtype=int)
                trans_counts = df["transmission"].value_counts() if "transmission" in df.columns else pd.Series(dtype=int)

                with col2:
                    most_fuel = fuel_counts.index[0] if len(fuel_counts) > 0 else "N/A"
                    st.metric("⛽ Most Common Fuel", most_fuel)

                with col3:
                    most_trans = trans_counts.index[0] if len(trans_counts) > 0 else "N/A"
                    st.metric("⚙️ Most Common Trans.", most_trans)

                with col4:
                    avg_year = int(df["year"].mean()) if "year" in df.columns else "N/A"
                    st.metric("📅 Avg. Year", avg_year)

                st.markdown("---")

                # Charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.subheader("⛽ Fuel Type Distribution")
                    st.bar_chart(fuel_counts)

                with chart_col2:
                    st.subheader("⚙️ Transmission Distribution")
                    st.bar_chart(trans_counts)

                chart_col3, chart_col4 = st.columns(2)

                with chart_col3:
                    st.subheader("📅 Manufacturing Year Distribution")
                    if "year" in df.columns:
                        year_counts = df["year"].value_counts().sort_index()
                        st.bar_chart(year_counts)

                with chart_col4:
                    st.subheader("🏭 Make Distribution")
                    if "make" in df.columns:
                        make_counts = df["make"].value_counts().head(10)
                        st.bar_chart(make_counts)

                st.markdown("---")
                st.subheader("📊 Recent Vehicles")
                display_df = df[["id", "make", "model", "year", "km_driven", "fuel_type", "transmission"]].tail(10)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            else:
                with col2:
                    st.metric("⛽ Fuel Types", "N/A")
                with col3:
                    st.metric("⚙️ Transmissions", "N/A")
                with col4:
                    st.metric("📅 Avg. Year", "N/A")

                st.info("📭 No vehicles found. Add some vehicles using the Vehicle Management section!")

        else:
            st.error("Failed to fetch vehicle data from the API.")


# ══════════════════════════════════════════════════════════════════
# VEHICLE MANAGEMENT
# ══════════════════════════════════════════════════════════════════

elif page == "📋 Vehicle Management":
    st.markdown('<p class="main-header">📋 Vehicle Management</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Add, view, update, and delete vehicle records</p>', unsafe_allow_html=True)

    if not api_healthy:
        st.warning("⚠️ Backend API is not reachable.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "➕ Add Vehicle", "📋 View All", "🔍 Search by ID",
        "✏️ Update Vehicle", "🔧 Partial Update", "🗑️ Delete Vehicle"
    ])

    # ── Add Vehicle ──
    with tab1:
        st.subheader("Add a New Vehicle")
        with st.form("add_vehicle_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                make = st.selectbox("Make / Brand", COMMON_MAKES, index=0)
                model_name = st.text_input("Model", placeholder="e.g., Swift, Creta")
                year = st.number_input("Manufacturing Year", min_value=1900, max_value=2030, value=2020)
                km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=2000000, value=30000)
            with col2:
                fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
                transmission = st.selectbox("Transmission", TRANSMISSION_TYPES)
                engine_cc = st.number_input("Engine Capacity (CC)", min_value=50, max_value=10000, value=1200)
                previous_owners = st.number_input("Previous Owners", min_value=0, max_value=20, value=0)

            submitted = st.form_submit_button("🚗 Add Vehicle", use_container_width=True)
            if submitted:
                if not model_name or not model_name.strip():
                    st.error("Please enter a vehicle model name.")
                else:
                    payload = {
                        "make": make,
                        "model": model_name.strip(),
                        "year": int(year),
                        "km_driven": int(km_driven),
                        "fuel_type": fuel_type,
                        "transmission": transmission,
                        "engine_cc": int(engine_cc),
                        "previous_owners": int(previous_owners),
                    }
                    with st.spinner("Adding vehicle..."):
                        resp = api_request("POST", "/vehicles", json_data=payload)
                    if resp and resp.status_code == 201:
                        result = resp.json()
                        st.success(f"✅ Vehicle added successfully! ID: **{result.get('id', 'N/A')}**")
                        st.json(result)
                    elif resp:
                        st.error(f"Failed to add vehicle: {resp.json().get('detail', resp.text)}")

    # ── View All ──
    with tab2:
        st.subheader("All Vehicle Records")
        if st.button("🔄 Refresh", key="refresh_all"):
            st.rerun()

        with st.spinner("Loading vehicles..."):
            resp = api_request("GET", "/vehicles")
        if resp and resp.status_code == 200:
            data = resp.json()
            vehicles = data.get("vehicles", [])
            st.info(f"Total vehicles: **{data.get('total', 0)}**")
            if vehicles:
                df = pd.DataFrame(vehicles)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No vehicles found. Add some using the 'Add Vehicle' tab.")
        elif resp:
            st.error(f"Failed to fetch vehicles: {resp.text}")

    # ── Search by ID ──
    with tab3:
        st.subheader("Search Vehicle by ID")
        search_id = st.text_input("Enter Vehicle ID", placeholder="e.g., VH-A1B2C3D4")
        if st.button("🔍 Search", key="search_btn"):
            if not search_id.strip():
                st.warning("Please enter a vehicle ID.")
            else:
                with st.spinner("Searching..."):
                    resp = api_request("GET", f"/vehicles/{search_id.strip()}")
                if resp and resp.status_code == 200:
                    vehicle = resp.json()
                    st.success("Vehicle found!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**ID:** {vehicle.get('id')}")
                        st.markdown(f"**Make:** {vehicle.get('make')}")
                        st.markdown(f"**Model:** {vehicle.get('model')}")
                        st.markdown(f"**Year:** {vehicle.get('year')}")
                        st.markdown(f"**KM Driven:** {vehicle.get('km_driven'):,}")
                    with col2:
                        st.markdown(f"**Fuel Type:** {vehicle.get('fuel_type')}")
                        st.markdown(f"**Transmission:** {vehicle.get('transmission')}")
                        st.markdown(f"**Engine CC:** {vehicle.get('engine_cc')}")
                        st.markdown(f"**Previous Owners:** {vehicle.get('previous_owners')}")
                        st.markdown(f"**Created:** {vehicle.get('created_at', 'N/A')}")
                elif resp and resp.status_code == 404:
                    st.warning(f"Vehicle with ID '{search_id.strip()}' not found.")
                elif resp:
                    st.error(f"Error: {resp.text}")

    # ── Update (PUT) ──
    with tab4:
        st.subheader("Update Vehicle (Replace All Fields)")
        update_id = st.text_input("Vehicle ID to Update", placeholder="e.g., VH-A1B2C3D4", key="update_id")

        if update_id.strip():
            # Fetch current data
            resp = api_request("GET", f"/vehicles/{update_id.strip()}", show_error=False)
            current = resp.json() if resp and resp.status_code == 200 else None

            if current:
                st.info(f"Editing vehicle: **{current.get('id')}** — {current.get('make')} {current.get('model')}")

            with st.form("update_vehicle_form"):
                col1, col2 = st.columns(2)
                with col1:
                    u_make_idx = COMMON_MAKES.index(current["make"]) if current and current.get("make") in COMMON_MAKES else 0
                    u_make = st.selectbox("Make", COMMON_MAKES, index=u_make_idx, key="u_make")
                    u_model = st.text_input("Model", value=current.get("model", "") if current else "", key="u_model")
                    u_year = st.number_input("Year", 1900, 2030, value=current.get("year", 2020) if current else 2020, key="u_year")
                    u_km = st.number_input("KM Driven", 0, 2000000, value=current.get("km_driven", 30000) if current else 30000, key="u_km")
                with col2:
                    u_fuel_idx = FUEL_TYPES.index(current["fuel_type"]) if current and current.get("fuel_type") in FUEL_TYPES else 0
                    u_fuel = st.selectbox("Fuel Type", FUEL_TYPES, index=u_fuel_idx, key="u_fuel")
                    u_trans_idx = TRANSMISSION_TYPES.index(current["transmission"]) if current and current.get("transmission") in TRANSMISSION_TYPES else 0
                    u_trans = st.selectbox("Transmission", TRANSMISSION_TYPES, index=u_trans_idx, key="u_trans")
                    u_cc = st.number_input("Engine CC", 50, 10000, value=current.get("engine_cc", 1200) if current else 1200, key="u_cc")
                    u_owners = st.number_input("Previous Owners", 0, 20, value=current.get("previous_owners", 0) if current else 0, key="u_owners")

                if st.form_submit_button("💾 Update Vehicle", use_container_width=True):
                    if not u_model or not u_model.strip():
                        st.error("Model name is required.")
                    else:
                        payload = {
                            "make": u_make, "model": u_model.strip(),
                            "year": int(u_year), "km_driven": int(u_km),
                            "fuel_type": u_fuel, "transmission": u_trans,
                            "engine_cc": int(u_cc), "previous_owners": int(u_owners),
                        }
                        with st.spinner("Updating..."):
                            resp = api_request("PUT", f"/vehicles/{update_id.strip()}", json_data=payload)
                        if resp and resp.status_code == 200:
                            st.success("✅ Vehicle updated successfully!")
                            st.json(resp.json())
                        elif resp and resp.status_code == 404:
                            st.warning(f"Vehicle '{update_id.strip()}' not found.")
                        elif resp:
                            st.error(f"Update failed: {resp.json().get('detail', resp.text)}")

    # ── Partial Update (PATCH) ──
    with tab5:
        st.subheader("Partial Update (PATCH)")
        patch_id = st.text_input("Vehicle ID", placeholder="e.g., VH-A1B2C3D4", key="patch_id")

        if patch_id.strip():
            st.markdown("*Leave fields empty/unchanged to skip them*")
            with st.form("patch_form"):
                col1, col2 = st.columns(2)
                with col1:
                    p_km = st.number_input("KM Driven (0 to skip)", min_value=0, max_value=2000000, value=0, key="p_km")
                    p_year = st.number_input("Year (0 to skip)", min_value=0, max_value=2030, value=0, key="p_year")
                    p_owners = st.number_input("Previous Owners (-1 to skip)", min_value=-1, max_value=20, value=-1, key="p_owners")
                with col2:
                    p_fuel = st.selectbox("Fuel Type", ["-- No Change --"] + FUEL_TYPES, key="p_fuel")
                    p_trans = st.selectbox("Transmission", ["-- No Change --"] + TRANSMISSION_TYPES, key="p_trans")
                    p_cc = st.number_input("Engine CC (0 to skip)", min_value=0, max_value=10000, value=0, key="p_cc")

                if st.form_submit_button("🔧 Apply Patch", use_container_width=True):
                    patch_data = {}
                    if p_km > 0:
                        patch_data["km_driven"] = int(p_km)
                    if p_year > 0:
                        patch_data["year"] = int(p_year)
                    if p_owners >= 0:
                        patch_data["previous_owners"] = int(p_owners)
                    if p_fuel != "-- No Change --":
                        patch_data["fuel_type"] = p_fuel
                    if p_trans != "-- No Change --":
                        patch_data["transmission"] = p_trans
                    if p_cc > 0:
                        patch_data["engine_cc"] = int(p_cc)

                    if not patch_data:
                        st.warning("No fields selected for update.")
                    else:
                        with st.spinner("Patching..."):
                            resp = api_request("PATCH", f"/vehicles/{patch_id.strip()}", json_data=patch_data)
                        if resp and resp.status_code == 200:
                            st.success("✅ Vehicle patched successfully!")
                            st.json(resp.json())
                        elif resp and resp.status_code == 404:
                            st.warning(f"Vehicle '{patch_id.strip()}' not found.")
                        elif resp:
                            st.error(f"Patch failed: {resp.json().get('detail', resp.text)}")

    # ── Delete ──
    with tab6:
        st.subheader("Delete a Vehicle")
        del_id = st.text_input("Vehicle ID to Delete", placeholder="e.g., VH-A1B2C3D4", key="del_id")

        if del_id.strip():
            # Show current vehicle info
            resp = api_request("GET", f"/vehicles/{del_id.strip()}", show_error=False)
            if resp and resp.status_code == 200:
                v = resp.json()
                st.warning(f"⚠️ You are about to delete: **{v.get('make')} {v.get('model')} ({v.get('year')})** — ID: {v.get('id')}")

                confirm = st.checkbox("I confirm I want to delete this vehicle", key="del_confirm")
                if confirm:
                    if st.button("🗑️ Delete Vehicle", type="primary", key="del_btn"):
                        with st.spinner("Deleting..."):
                            resp = api_request("DELETE", f"/vehicles/{del_id.strip()}")
                        if resp and resp.status_code == 200:
                            st.success("✅ Vehicle deleted successfully!")
                        elif resp and resp.status_code == 404:
                            st.warning("Vehicle not found.")
                        elif resp:
                            st.error(f"Delete failed: {resp.text}")
            elif resp and resp.status_code == 404:
                st.info(f"No vehicle found with ID: {del_id.strip()}")


# ══════════════════════════════════════════════════════════════════
# PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════

elif page == "💰 Price Prediction":
    st.markdown('<p class="main-header">💰 Vehicle Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Estimate the resale price of a used vehicle</p>', unsafe_allow_html=True)

    if not api_healthy:
        st.warning("⚠️ Backend API is not reachable.")
        st.stop()

    with st.form("price_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pp_make = st.selectbox("Make / Brand", COMMON_MAKES, key="pp_make")
            pp_model = st.text_input("Model", placeholder="e.g., Swift, Creta, City", key="pp_model")
            pp_year = st.number_input("Manufacturing Year", 1900, 2030, 2020, key="pp_year")
            pp_km = st.number_input("Kilometres Driven", 0, 2000000, 40000, key="pp_km")
        with col2:
            pp_fuel = st.selectbox("Fuel Type", FUEL_TYPES, key="pp_fuel")
            pp_trans = st.selectbox("Transmission", TRANSMISSION_TYPES, key="pp_trans")
            pp_cc = st.number_input("Engine Capacity (CC)", 50, 10000, 1200, key="pp_cc")
            pp_owners = st.number_input("Previous Owners", 0, 20, 1, key="pp_owners")

        predict_btn = st.form_submit_button("💰 Predict Price", use_container_width=True)

    if predict_btn:
        if not pp_model or not pp_model.strip():
            st.error("Please enter a vehicle model name.")
        else:
            payload = {
                "make": pp_make,
                "model": pp_model.strip(),
                "year": int(pp_year),
                "km_driven": int(pp_km),
                "fuel_type": pp_fuel,
                "transmission": pp_trans,
                "engine_cc": int(pp_cc),
                "previous_owners": int(pp_owners),
            }
            with st.spinner("🔮 Predicting price..."):
                resp = api_request("POST", "/predict/price", json_data=payload)

            if resp and resp.status_code == 200:
                result = resp.json()
                price = result.get("predicted_price", 0)

                st.markdown(f"""
                <div class="prediction-result">
                    <div style="font-size: 1rem; opacity: 0.9;">Estimated Resale Price</div>
                    <div class="prediction-price">{format_inr(price)}</div>
                    <div style="font-size: 0.85rem; opacity: 0.8;">({result.get('currency', 'INR')} {price:,.2f})</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**📋 Input Details:**")
                input_summary = result.get("input_summary", {})
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.markdown(f"- **Make:** {input_summary.get('make', 'N/A')}")
                    st.markdown(f"- **Model:** {input_summary.get('model', 'N/A')}")
                    st.markdown(f"- **Year:** {input_summary.get('year', 'N/A')}")
                    st.markdown(f"- **KM Driven:** {input_summary.get('km_driven', 'N/A'):,}")
                with summary_col2:
                    st.markdown(f"- **Fuel Type:** {input_summary.get('fuel_type', 'N/A')}")
                    st.markdown(f"- **Transmission:** {input_summary.get('transmission', 'N/A')}")
                    st.markdown(f"- **Engine CC:** {input_summary.get('engine_cc', 'N/A')}")
                    st.markdown(f"- **Previous Owners:** {input_summary.get('previous_owners', 'N/A')}")

                st.markdown(f"*Model: {result.get('model_name', 'N/A')}*")
                st.info(f"⚠️ {result.get('disclaimer', '')}")

            elif resp and resp.status_code == 503:
                st.error(f"🔧 {resp.json().get('detail', 'Model not available')}")
            elif resp:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                st.error(f"Prediction failed: {detail}")


# ══════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

elif page == "🏷️ Category Classification":
    st.markdown('<p class="main-header">🏷️ Vehicle Category Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classify a vehicle into Budget, Mid-range, or Premium</p>', unsafe_allow_html=True)

    if not api_healthy:
        st.warning("⚠️ Backend API is not reachable.")
        st.stop()

    st.markdown("""
    **Category Definitions (Price Bands):**
    - 🟢 **Budget**: Estimated resale price < ₹3,00,000
    - 🔵 **Mid-range**: ₹3,00,000 ≤ Estimated resale price < ₹8,00,000
    - 🟣 **Premium**: Estimated resale price ≥ ₹8,00,000

    *These categories represent price segments, not quality or condition assessments.*
    """)

    with st.form("category_form"):
        col1, col2 = st.columns(2)
        with col1:
            cc_make = st.selectbox("Make / Brand", COMMON_MAKES, key="cc_make")
            cc_model = st.text_input("Model", placeholder="e.g., Fortuner, Alto", key="cc_model")
            cc_year = st.number_input("Manufacturing Year", 1900, 2030, 2020, key="cc_year")
            cc_km = st.number_input("Kilometres Driven", 0, 2000000, 35000, key="cc_km")
        with col2:
            cc_fuel = st.selectbox("Fuel Type", FUEL_TYPES, key="cc_fuel")
            cc_trans = st.selectbox("Transmission", TRANSMISSION_TYPES, key="cc_trans")
            cc_cc = st.number_input("Engine Capacity (CC)", 50, 10000, 1200, key="cc_cc")
            cc_owners = st.number_input("Previous Owners", 0, 20, 1, key="cc_owners")

        classify_btn = st.form_submit_button("🏷️ Classify Vehicle", use_container_width=True)

    if classify_btn:
        if not cc_model or not cc_model.strip():
            st.error("Please enter a vehicle model name.")
        else:
            payload = {
                "make": cc_make,
                "model": cc_model.strip(),
                "year": int(cc_year),
                "km_driven": int(cc_km),
                "fuel_type": cc_fuel,
                "transmission": cc_trans,
                "engine_cc": int(cc_cc),
                "previous_owners": int(cc_owners),
            }
            with st.spinner("🔮 Classifying..."):
                resp = api_request("POST", "/predict/category", json_data=payload)

            if resp and resp.status_code == 200:
                result = resp.json()
                category = result.get("predicted_category", "Unknown")

                category_colors = {
                    "Budget": ("#48BB78", "budget"),
                    "Mid-range": ("#4299E1", "midrange"),
                    "Premium": ("#9F7AEA", "premium"),
                }
                color, css_class = category_colors.get(category, ("#718096", "default"))

                category_emojis = {"Budget": "🟢", "Mid-range": "🔵", "Premium": "🟣"}
                emoji = category_emojis.get(category, "⚪")

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}90, {color}); border-radius: 12px; padding: 2rem; text-align: center; color: white; margin: 1rem 0;">
                    <div style="font-size: 1rem; opacity: 0.9;">Predicted Category</div>
                    <div style="font-size: 2.5rem; font-weight: 700;">{emoji} {category}</div>
                    <div style="font-size: 0.95rem; opacity: 0.9; margin-top: 0.5rem;">{result.get('category_description', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"*Model: {result.get('model_name', 'N/A')}*")
                st.info(f"⚠️ {result.get('disclaimer', '')}")

            elif resp and resp.status_code == 503:
                st.error(f"🔧 {resp.json().get('detail', 'Model not available')}")
            elif resp:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                st.error(f"Classification failed: {detail}")


# ══════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Vehicle Management and Price Prediction System

    ### 👤 Project Owner
    - **Name:** Sahishnakumaar S S
    - **Roll Number:** 24ADR134

    ---

    ### 🎯 Objectives
    1. Manage vehicle records with full CRUD operations
    2. Predict used vehicle resale prices using machine learning
    3. Classify vehicles into price-band categories
    4. Provide an interactive web interface
    5. Expose a RESTful API with automated documentation

    ---

    ### 🛠️ Technologies Used

    | Component | Technology |
    |---|---|
    | Backend API | FastAPI + Uvicorn |
    | Frontend | Streamlit |
    | Data Validation | Pydantic |
    | ML Models | Scikit-learn (RandomForest) |
    | Data Processing | Pandas, NumPy |
    | Model Serialization | Joblib |
    | HTTP Client | Requests |
    | Testing | Pytest |
    | Containerization | Docker, Docker Compose |
    | Storage | JSON (local file) |

    ---

    ### 🏗️ Architecture

    ```
    User → Streamlit Frontend → FastAPI Backend → Storage / ML Models → Response
    ```

    - The **Streamlit frontend** communicates with the backend through HTTP requests
    - The **FastAPI backend** handles all business logic, data storage, and ML inference
    - **Vehicle records** are stored in a local JSON file with thread-safe operations
    - **ML models** are trained offline and loaded at backend startup

    ---

    ### ⚠️ Important Limitations

    1. **Synthetic Data**: The ML models are trained on synthetic data that approximates Indian used-vehicle market patterns. Predictions do **not** reflect real-world market accuracy.

    2. **Price Estimates**: Predicted prices are estimates only. They are **not** guaranteed sale prices or certified valuations.

    3. **Category Classification**: Vehicle categories (Budget/Mid-range/Premium) are based on price bands, **not** quality or condition assessments.

    4. **JSON Storage**: The vehicle record storage uses local JSON files. This is suitable for small-scale use but does not provide multi-process transactional guarantees.

    5. **No Authentication**: The current system does not implement user authentication or authorization.
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; padding: 1rem;'>"
        "Vehicle Management and Price Prediction System © 2024 — Sahishnakumaar S S (24ADR134)"
        "</div>",
        unsafe_allow_html=True
    )
