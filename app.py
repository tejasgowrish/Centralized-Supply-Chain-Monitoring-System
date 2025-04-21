import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from datetime import datetime, timedelta


#########################################################################################################################################

# Page Configuration
st.set_page_config(page_title="📦 Supply Chain Dashboard", layout="wide")


#########################################################################################################################################

# Sample data generation
def create_sample_data_if_not_exists():
    # Inventory data
    if not os.path.exists("inventory.csv"):
        inventory_data = {
            "item_id": range(1, 11),
            "item_name": ["Aluminum Sheets", "Circuit Boards", "Steel Bolts", "Hydraulic Pumps", 
              "Corrugated Boxes", "Industrial Lubricant", "Resistors - 10kΩ", "Plastic Casings", 
              "Gear Assemblies", "Smartphone Model X"],
            "stock_level": [120, 85, 50, 30, 200, 15, 65, 25, 40, 75],
            "reorder_threshold": [50, 40, 30, 20, 100, 30, 30, 20, 25, 40],
            "supplier": ["TMT Pvt Ltd", "Techtronix Components", "Bharat Fasteners Co.", "TMT Pvt Ltd",
                "HydroMech Systems", "Techtronix Components", "Bharat Fasteners Co.", "Ace Packaging India", 
                "Techtronix Components", "In-House Production"],
            "lead_time_days": [14, 7, 10, 14, 5, 7, 10, 21, 14, 3],
            "last_updated": [datetime.now() - pd.Timedelta(days=x) for x in range(10)]
        }
        pd.DataFrame(inventory_data).to_csv("inventory.csv", index=False)
    
    # Orders data
    if not os.path.exists("orders.csv"):
        orders_data = {
            "order_id": [f"ORD-{i:04d}" for i in range(1, 31)],
            "customer": [f"Customer {chr(65 + i % 8)}" for i in range(30)],
            "order_date": [(datetime.now() - pd.Timedelta(days=i)) for i in range(30)],
            "requested_delivery": [(datetime.now() + pd.Timedelta(days=i % 10 + 5)) for i in range(30)],
            "status": np.random.choice(["New", "Processing", "Shipped", "Delivered", "Cancelled"], 30, 
                                     p=[0.1, 0.3, 0.3, 0.2, 0.1]),
            "total_value": np.random.uniform(500, 10000, 30).round(2)
        }
        pd.DataFrame(orders_data).to_csv("orders.csv", index=False)
    
    # Shipments data
    if not os.path.exists("shipments.csv"):
        shipments_data = {
            "shipment_id": [f"SHP-{i:04d}" for i in range(1, 26)],
            "order_id": [f"ORD-{i:04d}" for i in range(1, 26)],
            "ship_date": [(datetime.now() - pd.Timedelta(days=i % 15)) for i in range(1, 26)],
            "carrier": np.random.choice(["FedEx", "UPS", "DHL", "USPS"], 25),
            "status": np.random.choice(["In Transit", "Delivered", "Delayed", "Scheduled"], 25, 
                                     p=[0.4, 0.3, 0.2, 0.1]),
            "tracking_number": [f"TRK{i:06d}" for i in range(100000, 100025)],
            "estimated_arrival": [(datetime.now() + pd.Timedelta(days=i % 7)) for i in range(1, 26)]
        }
        pd.DataFrame(shipments_data).to_csv("shipments.csv", index=False)
    
    # Costs data
    if not os.path.exists("costs.csv"):
        categories = ["Raw Materials", "Manufacturing", "Warehousing", "Transportation", 
                    "Inventory Holding", "Order Processing", "Returns", "Administrative"]
        costs_data = {
            "category": categories,
            "amount": [125000, 87500, 45000, 67500, 32000, 18500, 12000, 36000],
            "budget": [130000, 90000, 50000, 70000, 35000, 20000, 15000, 40000],
            "period": ["Q1 2025"] * 8
        }
        pd.DataFrame(costs_data).to_csv("costs.csv", index=False)
    
    # Supplier data - new dataset
    if not os.path.exists("suppliers.csv"):
        supplier_data = {
            "supplier_name": ["TMT Pvt Ltd", "Techtronix Components", "Bharat Fasteners Co.", "HydroMech Systems", "Ace Packaging India"],
            "reliability_score": [4.2, 3.8, 4.7, 3.5, 4.0],
            "avg_lead_time": [12, 8, 11, 6, 18],
            "on_time_delivery": [0.92, 0.85, 0.95, 0.78, 0.88],
            "quality_score": [4.5, 3.9, 4.8, 3.7, 4.1],
            "location": ["Mumbai, Maharashtra", "Bangalore, Karnataka", "Ludhiana, Punjab", "Hyderabad, Telangana", "Ahmedabad, Gujarat"]
        }
        pd.DataFrame(supplier_data).to_csv("suppliers.csv", index=False)



#########################################################################################################################################


# ML Component for Demand Forecasting
def forecast_demand():
    """Simple time series forecasting for inventory demand"""
    if "orders.csv" not in os.listdir():
        return None
    
    orders = pd.read_csv("orders.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    
    # Group by date and count orders
    daily_orders = orders.groupby(orders["order_date"].dt.date).size().reset_index()
    daily_orders.columns = ["date", "order_count"]
    daily_orders["date"] = pd.to_datetime(daily_orders["date"])
    
    # Simple moving average forecast
    if len(daily_orders) >= 7:
        # Create features (last 7 days)
        daily_orders["forecast"] = daily_orders["order_count"].rolling(window=7).mean().shift(1)
        daily_orders = daily_orders.dropna()
        
        # Forecast next 7 days
        last_date = daily_orders["date"].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        last_avg = daily_orders["order_count"].tail(7).mean()
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": [last_avg] * 7
        })
        
        return daily_orders, forecast_df
    return None





#########################################################################################################################################


# ✅ Data Loading with Cache
@st.cache_data(ttl=300)
def load_data():
    """Load all necessary data files with error handling"""
    data_files = {
        "inventory": "inventory.csv",
        "orders": "orders.csv",
        "shipments": "shipments.csv",
        "costs": "costs.csv",
        "suppliers": "suppliers.csv"
    }
    
    data = {}
    for key, filename in data_files.items():
        try:
            df = pd.read_csv(filename)
            # Convert date columns if they exist
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            data[key] = df
        except FileNotFoundError:
            st.warning(f"⚠️ Data file {filename} not found! Some features may be limited.")
            data[key] = pd.DataFrame()
    
    return data


#########################################################################################################################################


# Dashboard Pages
def render_dashboard_overview(data):
    inventory = data["inventory"]
    orders = data["orders"]
    shipments = data["shipments"]
    costs = data["costs"]
    
    # Main KPI Section
    st.title("📊 Centralized Supply Chain Monitoring Dashboard")
    
    # Date Filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From Date", datetime.now() - pd.Timedelta(days=30))
    with col2:
        end_date = st.date_input("To Date", datetime.now())
    
    # KPI Summary Cards with deltas (showing change from previous period)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    low_stock_count = len(inventory[inventory["stock_level"] < inventory["reorder_threshold"]])
    kpi1.metric("🔻 Low Stock Items", low_stock_count, delta="-2 from last week")
    
    if not orders.empty:
        total_orders = len(orders)
        new_orders = len(orders[orders["status"] == "New"])
        kpi2.metric("🛒 Total Orders", total_orders, delta=f"+{new_orders} new")
    
    if not shipments.empty:
        on_time = len(shipments[shipments["status"] != "Delayed"])
        on_time_pct = round((on_time / len(shipments)) * 100)
        kpi3.metric("📦 On-Time Delivery", f"{on_time_pct}%", delta=f"{on_time_pct - 95}% from target")
    
    if not costs.empty:
        total_cost = costs["amount"].sum()
        total_budget = costs["budget"].sum()
        cost_variance = round(((total_budget - total_cost) / total_budget) * 100, 1)
        kpi4.metric("💰 Budget Variance", f"${total_cost:,.2f}", delta=f"{cost_variance}% under budget")
    
    # Main overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Status Distribution")
        if not orders.empty:
            order_status = orders["status"].value_counts().reset_index()
            order_status.columns = ["Status", "Count"]
            fig = px.pie(order_status, values="Count", names="Status", hole=0.4,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Supply Chain Cost Breakdown")
        if not costs.empty:
            fig = px.bar(costs, x="category", y="amount", text_auto='.2s',
                       color="amount", labels={"amount": "Cost ($)"})
            fig.update_layout(xaxis_title="Category", yaxis_title="Amount ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Inventory Status
    st.subheader("Inventory Health")
    if not inventory.empty:
        # Calculate health metrics
        inventory["status"] = pd.cut(
            inventory["stock_level"] / inventory["reorder_threshold"],
            bins=[0, 0.8, 1.5, float('inf')],
            labels=["Critical", "Warning", "Healthy"]
        )
        
        status_counts = inventory["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        
        fig = px.bar(status_counts, x="Status", y="Count", color="Status",
                   color_discrete_map={"Critical": "red", "Warning": "orange", "Healthy": "green"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show critical items
        critical_items = inventory[inventory["status"] == "Critical"]
        if not critical_items.empty:
            st.error("🚨 Critical Stock Levels - Immediate Action Required")
            st.dataframe(critical_items[["item_name", "stock_level", "reorder_threshold"]])

def render_inventory_management(data):
    inventory = data["inventory"]
    
    st.title("📦 Inventory Management")
    
    # Search Bar
    st.subheader("🔍 Search Inventory")
    search_item = st.text_input("Search by item name:")

    if search_item:
        filtered_inventory = inventory[inventory["item_name"].str.contains(search_item, case=False)]
        st.dataframe(filtered_inventory)

        csv = filtered_inventory.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Filtered Inventory", data=csv, file_name="filtered_inventory.csv")
    
    # Inventory Overview
    with st.expander("📦 Inventory Overview", expanded=True):
        st.subheader("Current Inventory")
        st.dataframe(inventory)

        if not inventory.empty:
            fig_inventory = px.bar(inventory, x="item_name", y="stock_level", color="stock_level",
                                title="Stock Levels", labels={"stock_level": "Stock Level"})
            st.plotly_chart(fig_inventory)

            low_stock = inventory[inventory["stock_level"] < inventory["reorder_threshold"]]
            if not low_stock.empty:
                st.warning("⚠️ Low stock items detected:")
                st.dataframe(low_stock)
    
    # Manual Inventory Update
    with st.expander("✍️ Update Inventory"):
        with st.form(key="manual_inventory_update"):
            item_name = st.text_input("Item Name").strip().lower()
            stock_level = st.number_input("Stock Level", min_value=0)
            reorder_threshold = st.number_input("Reorder Threshold", min_value=0)
            submitted = st.form_submit_button("Update Inventory")

            if submitted:
                existing = inventory[inventory["item_name"].str.lower() == item_name].index
                if not existing.empty:
                    inventory.loc[existing, "stock_level"] = stock_level
                    inventory.loc[existing, "reorder_threshold"] = reorder_threshold
                    st.success(f"✅ '{item_name}' updated!")
                else:
                    new_row = pd.DataFrame([{
                        "item_name": item_name,
                        "stock_level": stock_level,
                        "reorder_threshold": reorder_threshold,
                        "supplier": "New Supplier",
                        "lead_time_days": 7,
                        "last_updated": datetime.now()
                    }])
                    inventory = pd.concat([inventory, new_row], ignore_index=True)
                    st.success(f"✅ '{item_name}' added!")

                inventory.to_csv("inventory.csv", index=False)
                st.rerun()
    
    # Reduce Stock Level
    with st.expander("➖ Reduce Inventory Stock"):
        if not inventory.empty:
            with st.form("reduce_stock_form"):
                item_to_reduce = st.selectbox("Select Item", options=inventory["item_name"].unique())
                max_qty = int(inventory[inventory["item_name"] == item_to_reduce]["stock_level"].values[0])
                reduce_qty = st.number_input("Quantity to reduce", min_value=1, max_value=max_qty if max_qty > 0 else 1, step=1)
                reduce_submitted = st.form_submit_button("Reduce")

                if reduce_submitted:
                    inventory.loc[inventory["item_name"] == item_to_reduce, "stock_level"] -= reduce_qty
                    inventory["stock_level"] = inventory["stock_level"].clip(lower=0)
                    inventory.to_csv("inventory.csv", index=False)
                    st.success(f"✅ Reduced {reduce_qty} units from '{item_to_reduce}'")
                    st.rerun()
        else:
            st.info("Inventory is empty.")
    
    # Inventory Metrics
    st.subheader("Inventory Metrics")
    
    if not inventory.empty:
        total_inventory = inventory["stock_level"].sum()
        avg_lead_time = inventory["lead_time_days"].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Items in Stock", total_inventory)
        col2.metric("Average Lead Time", f"{avg_lead_time:.1f} days")
        col3.metric("Stock Health", f"{len(inventory[inventory['stock_level'] > inventory['reorder_threshold']]) / len(inventory) * 100:.1f}%")
        
        # Inventory by Supplier
        st.subheader("Inventory by Supplier")
        supplier_inventory = inventory.groupby("supplier")["stock_level"].sum().reset_index()
        fig = px.pie(supplier_inventory, values="stock_level", names="supplier", 
                   title="Stock Distribution by Supplier")
        st.plotly_chart(fig)
        
        # Inventory Value Estimation
        st.subheader("Estimated Inventory Value")
        # Add dummy unit costs for demonstration
        inventory_value = inventory.copy()
        inventory_value["unit_cost"] = np.random.uniform(10, 100, len(inventory)).round(2)
        inventory_value["total_value"] = inventory_value["stock_level"] * inventory_value["unit_cost"]
        
        fig = px.bar(inventory_value.sort_values("total_value", ascending=False), 
                   x="item_name", y="total_value", 
                   color="supplier",
                   title="Estimated Value by Item")
        st.plotly_chart(fig)
        
        total_value = inventory_value["total_value"].sum()
        st.info(f"Total estimated inventory value: ${total_value:,.2f}")



#########################################################################################################################################



def render_order_shipment_tracking(data):
    orders = data["orders"]
    shipments = data["shipments"]
    
    st.title("🚚 Order & Shipment Tracking")
    
    tab1, tab2 = st.tabs(["📝 Orders", "🚚 Shipments"])
    
    with tab1:
        st.subheader("Order Management")
        
        # Order filters
        st.write("Filter Orders")
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect("Order Status", 
                                         options=orders["status"].unique() if not orders.empty else [],
                                         default=orders["status"].unique() if not orders.empty else [])
        with col2:
            date_range = st.date_input(
                "Order Date Range",
                value=(
                    orders["order_date"].min().date() if not orders.empty and "order_date" in orders.columns else datetime.now() - timedelta(days=30),
                    orders["order_date"].max().date() if not orders.empty and "order_date" in orders.columns else datetime.now()
                )
            )
        
        # Filter orders
        filtered_orders = orders
        if not orders.empty:
            if status_filter:
                filtered_orders = filtered_orders[filtered_orders["status"].isin(status_filter)]
                
            filtered_orders = filtered_orders[
                (filtered_orders["order_date"].dt.date >= date_range[0]) & 
                (filtered_orders["order_date"].dt.date <= date_range[1])
            ]
        
        # Display orders
        st.dataframe(filtered_orders)
        
        # Order analytics
        if not orders.empty:
            st.subheader("Order Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Orders by status
                status_counts = orders["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig = px.pie(status_counts, values="Count", names="Status", 
                           title="Orders by Status")
                st.plotly_chart(fig)
            
            with col2:
                # Order timeline
                orders_by_date = orders.groupby(orders["order_date"].dt.date).size().reset_index()
                orders_by_date.columns = ["Date", "Count"]
                fig = px.line(orders_by_date, x="Date", y="Count", 
                            title="Orders over Time",
                            labels={"Count": "# of Orders", "Date": "Date"})
                st.plotly_chart(fig)
            
            # Top customers
            st.subheader("Top Customers")
            customer_orders = orders.groupby("customer").agg(
                order_count=("order_id", "count"),
                total_value=("total_value", "sum")
            ).reset_index().sort_values("total_value", ascending=False).head(5)
            
            fig = px.bar(customer_orders, x="customer", y="total_value", 
                       text_auto='.2s',
                       color="order_count",
                       labels={"total_value": "Total Value ($)", "customer": "Customer", "order_count": "# Orders"},
                       title="Top 5 Customers by Order Value")
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Shipment Tracking")
        
        # Shipment filters
        st.write("Filter Shipments")
        col1, col2 = st.columns(2)
        with col1:
            shipment_status = st.multiselect("Shipment Status", 
                                         options=shipments["status"].unique() if not shipments.empty else [],
                                         default=shipments["status"].unique() if not shipments.empty else [])
        with col2:
            carrier_filter = st.multiselect("Carrier", 
                                         options=shipments["carrier"].unique() if not shipments.empty else [],
                                         default=shipments["carrier"].unique() if not shipments.empty else [])
        
        # Filter shipments
        filtered_shipments = shipments
        if not shipments.empty:
            if shipment_status:
                filtered_shipments = filtered_shipments[filtered_shipments["status"].isin(shipment_status)]
            if carrier_filter:
                filtered_shipments = filtered_shipments[filtered_shipments["carrier"].isin(carrier_filter)]
        
        # Display shipments
        st.dataframe(filtered_shipments)
        
        # Shipment analytics
        if not shipments.empty:
            st.subheader("Shipment Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Shipments by status
                status_counts = shipments["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig = px.pie(status_counts, values="Count", names="Status", 
                           title="Shipments by Status")
                st.plotly_chart(fig)
            
            with col2:
                # Carrier performance
                carrier_perf = shipments.groupby("carrier").agg(
                    total_shipments=("shipment_id", "count"),
                    delayed=("status", lambda x: sum(x == "Delayed")),
                ).reset_index()
                
                carrier_perf["on_time_pct"] = (1 - carrier_perf["delayed"] / carrier_perf["total_shipments"]) * 100
                
                fig = px.bar(carrier_perf, x="carrier", y="on_time_pct",
                           labels={"carrier": "Carrier", "on_time_pct": "On-Time %"},
                           color="on_time_pct",
                           color_continuous_scale=px.colors.sequential.Viridis,
                           title="Carrier On-Time Performance")
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig)
            
            # Delivery timeline
            st.subheader("Estimated Delivery Timeline")
            
            # Create dummy timeline data for visualization
            timeline_data = []
            for _, shipment in filtered_shipments.iterrows():
                if pd.notna(shipment["ship_date"]) and pd.notna(shipment["estimated_arrival"]):
                    timeline_data.append({
                        "Task": f"SHP-{shipment['shipment_id'][-4:]}",
                        "Start": shipment["ship_date"],
                        "Finish": shipment["estimated_arrival"],
                        "Status": shipment["status"]
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                fig = px.timeline(timeline_df, x_start="Start", x_end="Finish", y="Task", color="Status")
                fig.update_layout(title="Shipment Timeline")
                st.plotly_chart(fig)



#########################################################################################################################################


def render_cost_analysis(data):
    costs = data["costs"]
    
    st.title("💰 Supply Chain Cost Analysis")
    
    if not costs.empty:
        # Main cost metrics
        total_cost = costs["amount"].sum()
        total_budget = costs["budget"].sum()
        variance = total_budget - total_cost
        variance_pct = (variance / total_budget) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"${total_cost:,.2f}")
        col2.metric("Total Budget", f"${total_budget:,.2f}")
        col3.metric("Variance", f"${variance:,.2f}", delta=f"{variance_pct:.1f}%")
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        
        fig = px.bar(costs, x="category", y=["amount", "budget"],
                   barmode="group",
                   labels={"value": "Amount ($)", "variable": "Type"},
                   title="Cost vs Budget by Category")
        st.plotly_chart(fig)
        
        # Add variance calculation
        costs_analysis = costs.copy()
        costs_analysis["variance"] = costs_analysis["budget"] - costs_analysis["amount"]
        costs_analysis["variance_pct"] = (costs_analysis["variance"] / costs_analysis["budget"]) * 100
        
        # Variance analysis
        st.subheader("Variance Analysis")
        
        fig = px.bar(costs_analysis.sort_values("variance"), x="category", y="variance",
                   labels={"variance": "Budget Variance ($)", "category": "Category"},
                   color="variance",
                   color_continuous_scale=px.colors.diverging.RdYlGn,
                   title="Budget Variance by Category")
        st.plotly_chart(fig)
        
        # Cost table with variance
        st.subheader("Detailed Cost Analysis")
        display_cols = ["category", "amount", "budget", "variance", "variance_pct"]
        st.dataframe(costs_analysis[display_cols].style.format({
            "amount": "${:,.2f}",
            "budget": "${:,.2f}",
            "variance": "${:,.2f}",
            "variance_pct": "{:.1f}%"
        }))
        
        # Cost optimization opportunities
        st.subheader("Cost Optimization Opportunities")
        
        # Identify categories with significant overspending
        overspend = costs_analysis[costs_analysis["variance"] < 0].sort_values("variance")
        
        if not overspend.empty:
            st.warning("Categories exceeding budget:")
            for _, row in overspend.iterrows():
                st.write(f"⚠️ **{row['category']}**: ${-row['variance']:,.2f} over budget ({-row['variance_pct']:.1f}% variance)")
        else:
            st.success("All categories are within budget!")
        
        # Cost-saving recommendations
        st.subheader("Cost Optimization Recommendations")
        
        # Sample recommendations (in real application, these would be ML-driven)
        recommendations = [
            "Consider consolidating shipments to reduce transportation costs",
            "Evaluate supplier contracts for raw materials to identify potential savings",
            "Optimize inventory levels to reduce holding costs",
            "Review administrative expenses for potential streamlining",
            "Analyze manufacturing processes for efficiency improvements"
        ]
        
        for rec in recommendations:
            st.write(f"💡 {rec}")
    else:
        st.info("No cost data available. Please upload or generate sample cost data.")



#########################################################################################################################################


def render_supplier_performance(data):
    suppliers = data["suppliers"]
    inventory = data["inventory"]
    
    st.title("📊 Supplier Performance Analysis")
    
    if not suppliers.empty:
        # Supplier selection
        selected_supplier = st.selectbox("Select Supplier for Detailed Analysis", 
                                        options=["All Suppliers"] + suppliers["supplier_name"].tolist())
        
        if selected_supplier == "All Suppliers":
            # Radar chart for all suppliers
            st.subheader("Comparative Performance Analysis")
            
            fig = go.Figure()
            
            categories = ["Reliability", "Lead Time", "On-Time Delivery", "Quality", "Cost"]
            
            for _, supplier in suppliers.iterrows():
                # Normalize metrics for radar chart (higher is better)
                lead_time_score = 5 - (supplier["avg_lead_time"] / 20 * 5)  # Invert so lower is better
                
                fig.add_trace(go.Scatterpolar(
                    r=[supplier["reliability_score"], lead_time_score, 
                       supplier["on_time_delivery"] * 5, supplier["quality_score"], 4],
                    theta=categories,
                    fill='toself',
                    name=supplier["supplier_name"]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show supplier table
            st.subheader("Supplier Performance Metrics")
            suppliers_display = suppliers.copy()
            suppliers_display["on_time_delivery"] = suppliers_display["on_time_delivery"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(suppliers_display)
            
            # Supplier ranking
            st.subheader("Supplier Ranking")
            
            # Calculate overall score (weighted average)
            supplier_rank = suppliers.copy()
            supplier_rank["lead_time_score"] = 5 - (supplier_rank["avg_lead_time"] / 20 * 5)
            supplier_rank["overall_score"] = (
                supplier_rank["reliability_score"] * 0.25 +
                supplier_rank["lead_time_score"] * 0.25 +
                supplier_rank["on_time_delivery"] * 5 * 0.25 +
                supplier_rank["quality_score"] * 0.25
            )
            
            supplier_rank = supplier_rank.sort_values("overall_score", ascending=False)
            
            fig = px.bar(supplier_rank, x="supplier_name", y="overall_score",
                       labels={"supplier_name": "Supplier", "overall_score": "Overall Score"},
                       color="overall_score",
                       title="Supplier Performance Ranking")
            st.plotly_chart(fig)
            
        else:
            # Single supplier detailed analysis
            supplier_data = suppliers[suppliers["supplier_name"] == selected_supplier].iloc[0]
            
            st.subheader(f"Detailed Analysis: {selected_supplier}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Reliability Score", f"{supplier_data['reliability_score']:.1f}/5")
            col2.metric("Lead Time", f"{supplier_data['avg_lead_time']} days")
            col3.metric("On-Time Delivery", f"{supplier_data['on_time_delivery']*100:.1f}%")
            col4.metric("Quality Score", f"{supplier_data['quality_score']:.1f}/5")
            
            # Get items supplied by this supplier
            if not inventory.empty:
                supplier_items = inventory[inventory["supplier"] == selected_supplier]
                
                if not supplier_items.empty:
                    st.subheader(f"Items Supplied by {selected_supplier}")
                    st.dataframe(supplier_items[["item_name", "stock_level", "lead_time_days"]])
                    
                    # Stock levels of items from this supplier
                    fig = px.bar(supplier_items, x="item_name", y="stock_level", 
                               color="stock_level",
                               labels={"item_name": "Item", "stock_level": "Stock Level"},
                               title=f"Current Stock Levels - {selected_supplier} Items")
                    st.plotly_chart(fig)
                    
                    # Risk assessment
                    critical_items = supplier_items[supplier_items["stock_level"] < supplier_items["reorder_threshold"]]
                    if not critical_items.empty:
                        st.warning(f"⚠️ Critical stock levels for {len(critical_items)} items supplied by {selected_supplier}")
                        st.dataframe(critical_items[["item_name", "stock_level", "reorder_threshold"]])
                else:
                    st.info(f"No items currently sourced from {selected_supplier}")
            
            # Performance history (simulated)
            st.subheader("Performance History")
            
            # Generate dummy historical data
            dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
            history = pd.DataFrame({
                "date": dates,
                "on_time_delivery": np.clip(
                    np.random.normal(supplier_data["on_time_delivery"], 0.05, size=12),
                    0, 1
                ),
                "quality_score": np.clip(
                    np.random.normal(supplier_data["quality_score"], 0.2, size=12),
                    1, 5
                ),
                "lead_time": np.clip(
                    np.random.normal(supplier_data["avg_lead_time"], 2, size=12),
                    1, 30
                ).astype(int)
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history["date"], 
                y=history["on_time_delivery"],
                mode='lines+markers',
                name='On-Time Delivery %'
            ))
            fig.add_trace(go.Scatter(
                x=history["date"], 
                y=history["quality_score"] / 5,  # Normalize to 0-1 scale
                mode='lines+markers',
                name='Quality Score (normalized)'
            ))
            fig.update_layout(title="Historical Performance Metrics")
            st.plotly_chart(fig)
            
            # Notes and action items
            st.subheader("Notes & Action Items")
            
            notes = st.text_area("Supplier Notes", placeholder="Enter notes about this supplier...")
            
            col1, col2 = st.columns(2)
            with col1:
                action_required = st.selectbox("Action Required", 
                                             ["None", "Review Contract", "Request Quality Improvement", 
                                              "Negotiate Lead Times", "Find Alternative Supplier"])
            with col2:
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            
            if st.button("Save Notes & Actions"):
                st.success("✅ Supplier information updated successfully!")
    else:
        st.info("No supplier data available.")


#########################################################################################################################################


def render_demand_forecasting():
    st.title("📈 Demand Forecasting")
    
    # Get forecast data
    forecast_result = forecast_demand()
    
    if forecast_result:
        historical_data, forecast_df = forecast_result
        
        st.subheader("Order Volume Forecast")
        
        # Combine historical and forecast data for visualization
        historical_data["type"] = "Historical"
        forecast_df["type"] = "Forecast"
        forecast_df["order_count"] = forecast_df["forecast"]
        
        combined_data = pd.concat([historical_data, forecast_df])
        
        fig = px.line(combined_data, x="date", y="order_count", color="type",
                    title="Order Volume - Historical & Forecast",
                    labels={"order_count": "Number of Orders", "date": "Date"})
        st.plotly_chart(fig)
        
        # Forecast details
        st.subheader("Forecast Details")
        st.dataframe(forecast_df[["date", "forecast"]].rename(columns={"forecast": "Forecasted Orders"}))
        
        # Forecast accuracy (dummy metrics for now)
        st.subheader("Forecast Accuracy Metrics")
        col1, col2, col3 = st.columns(3)
        # col1.metric("MAPE", "15.3%")
        # col2.metric("MAE", "2.4 orders")
        # col3.metric("Forecast Bias", "+0.8")

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(historical_data["order_count"], historical_data["forecast"])
        mape = (np.abs((historical_data["order_count"] - historical_data["forecast"]) / historical_data["order_count"])).mean() * 100
        bias = (historical_data["forecast"] - historical_data["order_count"]).mean()

        col1.metric("MAPE", mape)
        col2.metric("MAE", mae)
        col3.metric("Forecast Bias", bias)

        # Inventory recommendations based on forecast
        st.subheader("Inventory Recommendations")
        
        st.info("Based on the forecast, consider the following actions:")
        recommendations = [
            "Increase stock levels for Raw Material A by 20% to meet projected demand",
            "Expedite orders for Component B to avoid stockout in the next 7 days",
            "Consider reducing order frequency and increasing order quantity for Part C to optimize costs",
            "Schedule production increase for Finished Product J starting next week"
        ]
        
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
    else:
        st.info("Insufficient order data for forecasting. Please add more order data.")
        
        # Show sample forecast chart anyway
        st.subheader("Sample Forecast Visualization")
        dates = pd.date_range(start=datetime.now() - pd.Timedelta(days=30), periods=45)
        sample_data = pd.DataFrame({
            "date": dates,
            "order_count": np.random.normal(loc=10, scale=2, size=45).round()
        })
        sample_data["type"] = "Historical"
        sample_data.loc[sample_data["date"] > datetime.now(), "type"] = "Forecast"
        
        fig = px.line(sample_data, x="date", y="order_count", color="type",
                    title="Sample Order Forecast",
                    labels={"order_count": "Number of Orders", "date": "Date"})
        fig.add_vline(x=datetime.now(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig)
        
        st.write("Upload order data to generate actual forecasts.")



#########################################################################################################################################


def render_alerts_notifications():
    st.title("🚨 Alerts & Notifications")
    
    # Generate sample alerts
    alerts = [
        {"severity": "Critical", "message": "Item 'Raw Material A' has reached critical stock level (5 units)", "time": "2 hours ago"},
        {"severity": "High", "message": "Shipment SHP-0015 is delayed by carrier", "time": "5 hours ago"},
        {"severity": "Medium", "message": "New order received from Customer B worth $8,500", "time": "Yesterday"},
        {"severity": "Low", "message": "Supplier D performance dropped below threshold", "time": "2 days ago"},
        {"severity": "Critical", "message": "Component G stock level is zero - production impact likely", "time": "2 days ago"}
    ]
    
    # Alert filters
    st.subheader("Alert Filters")
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect("Severity", 
                                       options=["Critical", "High", "Medium", "Low"],
                                       default=["Critical", "High"])
    with col2:
        date_filter = st.selectbox("Time Range", 
                                 options=["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
                                 index=0)
    
    # Filter alerts (in a real app, this would filter from database)
    filtered_alerts = [alert for alert in alerts if alert["severity"] in severity_filter]
    
    # Display alerts
    st.subheader("Current Alerts")
    
    if not filtered_alerts:
        st.success("No alerts matching the selected criteria.")
    else:
        for alert in filtered_alerts:
            if alert["severity"] == "Critical":
                st.error(f"**{alert['severity']}**: {alert['message']} - {alert['time']}")
            elif alert["severity"] == "High":
                st.warning(f"**{alert['severity']}**: {alert['message']} - {alert['time']}")
            elif alert["severity"] == "Medium":
                st.info(f"**{alert['severity']}**: {alert['message']} - {alert['time']}")
            else:
                st.success(f"**{alert['severity']}**: {alert['message']} - {alert['time']}")
    
    # Alert settings
    with st.expander("Alert Settings"):
        st.subheader("Notification Preferences")
        
        st.checkbox("Email notifications", value=True)
        st.checkbox("SMS notifications", value=False)
        st.checkbox("In-app notifications", value=True)
        
        st.subheader("Alert Thresholds")
        st.slider("Low stock threshold (%)", 0, 100, 20, 
                help="Alert when stock falls below this percentage of reorder level")
        st.slider("Delivery delay threshold (days)", 0, 10, 1, 
                help="Alert when delivery is delayed by this many days")
        
        if st.button("Save Settings"):
            st.success("✅ Alert settings saved successfully!")



#######################################################################################################################################


# Main application function
def main():
    # Create sample data if not exists
    create_sample_data_if_not_exists()
    
    # Sidebar navigation
    st.sidebar.title("📦 Supply Chain Dashboard")
    
    navigation = st.sidebar.radio("Navigation", [
        "Dashboard Overview", 
        "Inventory Management", 
        "Order & Shipment Tracking",
        "Cost Analysis",
        "Supplier Performance",
        "Demand Forecasting",
        "Alerts & Notifications",
    ])
    
    # Load data
    data = load_data()
    
    # Render selected page
    if navigation == "Dashboard Overview":
        render_dashboard_overview(data)
    elif navigation == "Inventory Management":
        render_inventory_management(data)
    elif navigation == "Order & Shipment Tracking":
        render_order_shipment_tracking(data)
    elif navigation == "Cost Analysis":
        render_cost_analysis(data)
    elif navigation == "Supplier Performance":
        render_supplier_performance(data)
    elif navigation == "Demand Forecasting":
        render_demand_forecasting()
    elif navigation == "Alerts & Notifications":
        render_alerts_notifications()
    
    # Footer
    st.sidebar.markdown("---")
    # st.sidebar.info("Supply Chain Dashboard v1.0.0")
    
    # Show loading state for data refresh
    with st.sidebar:
        if st.button("Refresh Data"):
            with st.spinner("Refreshing data..."):
                time.sleep(1)  # Simulate loading
                st.success("✅ Data refreshed!")
                st.rerun()


#########################################################################################################################################

if __name__ == "__main__":
    main()