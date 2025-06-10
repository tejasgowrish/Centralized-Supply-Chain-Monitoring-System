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


## Key supply chain operations -
1. Supply chain overview - Provides a summary of the most critical components of the supply chain between a period of time of our choosing
  ![image](https://github.com/user-attachments/assets/7b03c501-a6e9-48ba-9c3f-12b8a9254cb8)

2. Inventory management -  Lists of current inventory items, search options, manual update options and visualizations for various inventory-related metrics
  ![image](https://github.com/user-attachments/assets/71ca03d1-9311-488c-a2eb-aa577984b51b)

3. Order and shipment tracking - Tracking the movement of goods and supplies through the various phases of the supply chain
  ![image](https://github.com/user-attachments/assets/131bb0a2-484f-4161-bf97-87c5c0ece00a)

4. Cost analysis - Analysis of all cost and finance related data in the supply chain
  ![image](https://github.com/user-attachments/assets/01d1be3e-96dc-4de6-b900-e8069cf03806)

5. Supplier performance tracking - Analysis of the best performing suppliers, ranking of suppliers etc.
  ![image](https://github.com/user-attachments/assets/f5a80e44-70eb-4c5e-8eca-f28b3dd65fdd)

6. Demand forecasting - Simple Moving Average (SMA) over the last 7 days to predict the demand for the next week
   ![image](https://github.com/user-attachments/assets/7b3cbbd5-dc26-43d0-8c68-217a292a033c)

7. Alerts and notifications - Stockouts, shipment delays etc.
   ![image](https://github.com/user-attachments/assets/6a03a668-9917-4368-adaf-dce46c88f767)


## Key features
1. Individual pages for the above-mentioned SCM operations.
2. Detailed visualizations -
   - Bar graphs
   - Pie charts
   - Line graphs
   - Web charts
3. Demand forecasting - Simple, straight-forward forecasting using Simple Moving Average (SMA)

