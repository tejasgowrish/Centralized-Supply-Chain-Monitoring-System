# Centralized-Supply-Chain-Monitoring-System
A web application that provides real-time monitoring and analytics of key supply chain operations.

## In this repo
```
supply_chain_dashboard/
├── main.py                # Main Streamlit app entry point
├── config/
│   └── settings.py        # Configuration settings
├── data/
│   ├── __init__.py
│   ├── data_generator.py  # Sample data generation
│   └── data_loader.py     # Data loading functions
├── models/
│   ├── __init__.py
│   └── forecasting.py     # ML models and forecasting
├── pages/
│   ├── __init__.py
│   ├── dashboard.py       # Dashboard overview
│   ├── inventory.py       # Inventory management
│   ├── orders.py          # Order & shipment tracking
│   ├── costs.py           # Cost analysis
│   ├── suppliers.py       # Supplier performance
│   ├── forecasting.py     # Demand forecasting
│   └── alerts.py          # Alerts & notifications
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
└── requirements.txt       # Dependencies
```

## Key features
1. Key supply chain operations -
   - Supply chain overview - Provides a summary of the most critical components of the supply chain between a period of time of our choosing
   - Inventory management -  Lists of current inventory items, search options, manual update options and visualizations for various inventory-related metrics
   - Order and shipment tracking - Tracking the movement of goods and supplies through the various phases of the supply chain
   - Cost analysis - Analysis of all cost and finance related data in the supply chain
   - Supplier performance tracking - Analysis of the best performing suppliers, ranking of suppliers etc.
   - Demand forecasting - Simple Moving Average (SMA) over the last 7 days to predict the demand for the next week
   - Alerts and notifications - Stockouts, shipment delays etc.
2. Individual pages for the above-mentioned SCM operations.
3. Detailed visualizations -
   - Bar graphs
   - Pie charts
   - Line graphs
   - Web charts
4. Demand forecasting - Simple, straight-forward forecasting using Simple Moving Average (SMA)
