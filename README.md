# Personal Budget Tracker

A simple and intuitive budgeting application built with Python and Gradio that helps you track your expenses and visualize your spending by category.

## Features

- Add transactions with amount, category, description, and date
- View current balance
- Track spending by category with visual charts
- View recent transactions
- Data persistence (transactions are saved to a CSV file)

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python budget_app.py
```

The application will start a local web server and open in your default web browser. If it doesn't open automatically, you can access it at `http://localhost:7860`.

## How to Use

1. **Add a Transaction**:
   - Enter the amount spent
   - Select a category from the dropdown
   - Add a brief description
   - The date will be automatically set to today, but you can change it if needed
   - Click "Add Transaction"

2. **View Your Balance**:
   - Your current balance is displayed at the bottom of the form

3. **View Spending by Category**:
   - A bar chart shows your spending by category

4. **Recent Transactions**:
   - The right panel shows your most recent transactions
   - Click "Refresh Transactions" to update the list

## Data Storage

All transactions are saved in a file called `transactions.csv` in the same directory as the application. This file will be created automatically when you add your first transaction.

## Requirements

- Python 3.8+
- Gradio
- Pandas

## License

MIT
