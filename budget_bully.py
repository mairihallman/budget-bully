import matplotlib
matplotlib.use("Agg")

import gradio as gr
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict, Tuple
import ollama
import asyncio
from functools import partial
import matplotlib.pyplot as plt

class BudgetTracker:
    def __init__(self):
        self.transactions = []
        self.income_categories = ["Salary/Pay Day", "Government", "E-transfers", "Other"]
        self.expense_categories = [
            # Essentials
            "Groceries", "Rent", "Utilities", "Internet", 
            "Loans", "Transportation", "Savings", "Hygiene",
            # Non-essentials
            "Dining Out", "Subscriptions", "Entertainment", 
            "Intoxicants", "Personal Care", "Miscellaneous"
        ]
        # Initialize goal amounts
        self.loans_goal = 5000.0  # Default loan repayment goal
        self.savings_goal = 10000.0  # Default savings goal
        self.load_data()
    
    def add_transaction(self, amount: float, category: str, description: str, date: str, transaction_type: str = 'expense') -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if transaction_type not in ['income', 'expense']:
            raise ValueError("Transaction type must be 'income' or 'expense'")
        signed_amount = amount if transaction_type == 'income' else -amount
        self.transactions.append({
            'date': date,
            'amount': signed_amount,
            'category': category,
            'description': description,
            'type': transaction_type
        })
        self.save_data()
    
    def get_balance(self) -> float:
        return sum(t['amount'] for t in self.transactions)
    
    def get_income_total(self) -> float:
        return sum(t['amount'] for t in self.transactions if t['amount'] > 0)
    
    def get_expense_total(self) -> float:
        return abs(sum(t['amount'] for t in self.transactions if t['amount'] < 0))
    
    def get_balance_summary(self) -> str:
        """Get a formatted summary of the current balance."""
        balance = self.get_balance()
        income = self.get_income_total()
        expenses = self.get_expense_total()
        return f"💰 Balance: ${balance:,.2f}\n💸 Income: ${income:,.2f}\n💳 Expenses: ${expenses:,.2f}"
    
    def get_transactions_by_type(self, transaction_type: str) -> Dict[str, float]:
        transactions = [t for t in self.transactions if t['type'] == transaction_type]
        categories = {}
        for t in transactions:
            categories[t['category']] = categories.get(t['category'], 0) + abs(t['amount'])
        return categories
    
    def get_recent_transactions(self, n: int = 5) -> List[Dict]:
        return sorted(self.transactions, key=lambda x: x['date'], reverse=True)[:n]
    
    def save_data(self) -> None:
        df = pd.DataFrame(self.transactions)
        if not os.path.exists('data'):
            os.makedirs('data')
        df.to_csv('data/transactions.csv', index=False)
    
    def load_data(self) -> None:
        try:
            if os.path.exists('data/transactions.csv'):
                df = pd.read_csv('data/transactions.csv')
                # Patch old transactions that might be missing the 'type' field
                if 'type' not in df.columns:
                    df['type'] = df['amount'].apply(lambda x: 'income' if x > 0 else 'expense')
                self.transactions = df.to_dict('records')
        except Exception as e:
            print(f"Error loading data: {e}")
            self.transactions = []
    
    def get_loans_total(self) -> float:
        """Get total amount paid towards loans."""
        return sum(abs(t['amount']) for t in self.transactions 
                  if t['category'] == 'Loans' and t['type'] == 'expense')
    
    def get_savings_total(self) -> float:
        """Get total amount saved."""
        return sum(abs(t['amount']) for t in self.transactions 
                  if t['category'] == 'Savings' and t['type'] == 'expense')
    
    def get_loans_progress_data(self) -> pd.DataFrame:
        """Get data for loans progress donut chart."""
        paid = self.get_loans_total()
        remaining = max(0, self.loans_goal - paid)
        return pd.DataFrame([
            {"Category": "Paid", "Amount": paid, "Color": "#FF69B4"},
            {"Category": "Remaining", "Amount": remaining, "Color": "#FFE4E1"}
        ])
    
    def get_savings_progress_data(self) -> pd.DataFrame:
        """Get data for savings progress donut chart."""
        saved = self.get_savings_total()
        remaining = max(0, self.savings_goal - saved)
        return pd.DataFrame([
            {"Category": "Saved", "Amount": saved, "Color": "#C71585"},
            {"Category": "Remaining", "Amount": remaining, "Color": "#FFB6C1"}
        ])

    def get_daily_outcome_by_category(self, days: int = 30) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: day, category, amount for the last N days (default 30),
        only for outcome (expense) transactions. Fills missing days/categories with zero.
        """
        # Helper to generate an empty dataframe with all days/categories
        def get_empty_df():
            today = pd.Timestamp.now().normalize()
            days_range = pd.date_range(today - pd.Timedelta(days=days-1), today)
            categories = self.expense_categories
            data = [
                {'day': d.strftime('%b %-d'), 'category': cat, 'amount': 0}
                for d in days_range for cat in categories
            ]
            return pd.DataFrame(data)

        df = pd.DataFrame(self.transactions)
        if df.empty:
            return get_empty_df()

        df = df[df['type'] == 'expense']
        if df.empty:
            return get_empty_df()

        df['date'] = pd.to_datetime(df['date'])
        today = pd.Timestamp.now().normalize()
        days_range = pd.date_range(today - pd.Timedelta(days=days-1), today)
        
        # Filter by date range
        df = df[df['date'].dt.normalize().isin(days_range)]
        if df.empty:
            return get_empty_df()

        df['day'] = df['date'].dt.strftime('%b %-d')
        
        # Pivot to fill missing days/categories with 0
        pivot = df.pivot_table(index='day', columns='category', values='amount', aggfunc='sum', fill_value=0)
        
        # Ensure all days from the range are in the index
        all_days_in_range = [d.strftime('%b %-d') for d in days_range]
        pivot = pivot.reindex(all_days_in_range, fill_value=0)
        
        # Ensure all expense categories are columns
        all_categories = self.expense_categories
        pivot = pivot.reindex(columns=all_categories, fill_value=0)
        
        pivot = pivot.sort_index()
        
        # Unpivot for plotting
        grouped = pivot.reset_index().melt(id_vars='day', var_name='category', value_name='amount')
        grouped['amount'] = grouped['amount'].abs()
        return grouped

# Initialize the budget tracker instance
budget_tracker = BudgetTracker()

def create_donut_chart(data: pd.DataFrame, title: str, center_text: str) -> plt.Figure:
    """Create a donut chart with center text."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Extract data
    values = data['Amount'].values
    labels = data['Category'].values
    # Use consistent Y2K colors for donut charts
    default_colors = ['#FF1493', '#FFB6C1', '#C71585', '#FF69B4', '#E75480', '#F4BBFF']
    colors = data['Color'].values if 'Color' in data.columns else default_colors[:len(data)]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.5),
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Add center text
    ax.text(0, 0, center_text, ha='center', va='center', fontsize=16, weight='bold')
    
    # Set title
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Remove background
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    return fig

def create_loans_donut() -> plt.Figure:
    """Create loans progress donut chart."""
    data = budget_tracker.get_loans_progress_data()
    paid = budget_tracker.get_loans_total()
    goal = budget_tracker.loans_goal
    percentage = (paid / goal * 100) if goal > 0 else 0
    center_text = f"${paid:,.0f}\nof\n${goal:,.0f}\n({percentage:.0f}%)"
    return create_donut_chart(data, "Loans Repayment Progress", center_text)

def create_savings_donut() -> plt.Figure:
    """Create savings progress donut chart."""
    data = budget_tracker.get_savings_progress_data()
    saved = budget_tracker.get_savings_total()
    goal = budget_tracker.savings_goal
    percentage = (saved / goal * 100) if goal > 0 else 0
    center_text = f"${saved:,.0f}\nof\n${goal:,.0f}\n({percentage:.0f}%)"
    return create_donut_chart(data, "Savings Progress", center_text)

def create_stacked_bar_chart(days=30):
    """
    Create a responsive stacked bar chart of daily spending by category for the last N days.
    - X-axis: Days (MMM D)
    - Y-axis: Amount spent ($)
    - Stacks: Categories
    - Rounded corners, Y2K colors, legend with circles, rotated x labels
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import seaborn as sns
    from matplotlib import cm
    
    df = budget_tracker.get_daily_outcome_by_category(days)
    
    # Check if there's any spending data at all
    if df.empty or 'amount' not in df.columns or df['amount'].sum() == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No spending data for this range ✨", ha="center", va="center", fontsize=16, color="#FF1493", weight='bold')
        ax.axis('off')
        return fig

    # Filter out categories that have a total sum of zero for the period
    df = df[df.groupby('category')['amount'].transform('sum') > 0]
    
    # Pivot for stacked bar
    pivot = df.pivot(index='day', columns='category', values='amount').fillna(0)
    
    categories = pivot.columns.tolist()
    days_sorted = pivot.index.tolist()
    
    # Enhanced Y2K color palette with better contrast
    y2k_palette = [
        "#FF1493",  # Deep Pink
        "#FF69B4",  # Hot Pink  
        "#C71585",  # Medium Violet Red
        "#FFB6C1",  # Light Pink
        "#E75480",  # Pale Violet Red
        "#F88379",  # Salmon Pink
        "#F4BBFF",  # Light Lavender
        "#FFB3DE",  # Pink Lavender
        "#DDA0DD",  # Plum
        "#DA70D6",  # Orchid
        "#EE82EE",  # Violet
        "#FFC0CB"   # Pink
    ]
    colors = y2k_palette[:len(categories)] if len(categories) <= len(y2k_palette) else sns.color_palette('pastel', len(categories))
        
    fig, ax = plt.subplots(figsize=(max(8, len(days_sorted)*0.5), 5))
    bottom = [0]*len(pivot)
    
    for idx, cat in enumerate(categories):
        bars = ax.bar(
            pivot.index,
            pivot[cat],
            label=cat,
            bottom=bottom,
            color=colors[idx],
            edgecolor='white',
            linewidth=1.5,
            width=0.7,
            zorder=3
        )
        # Rounded tops
        for bar in bars:
            bar.set_linewidth(0)
            bar.set_linestyle('-')
            bar.set_capstyle('round')
            bar.set_joinstyle('round')
            bar.set_path_effects([])
            bar.set_alpha(0.95)
        bottom = [b + v for b, v in zip(bottom, pivot[cat])]
        
    # Y2K grid & background
    ax.set_facecolor('#fff0fa')
    fig.patch.set_facecolor('#fff0fa')
    ax.grid(axis='y', color='#F4BBFF', linestyle='--', linewidth=1, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#FF69B4')
    ax.spines['bottom'].set_color('#FF69B4')
    
    # X labels
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=11, color='#FF1493', weight='bold')
    ax.set_xlabel('Date', fontsize=13, color='#FF1493', weight='bold')
    ax.set_ylabel('Amount ($)', fontsize=13, color='#FF1493', weight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='y', labelsize=11, colors='#FF1493', labelcolor='#FF1493')
    
    # Legend with circular markers
    legend_handles = [mpatches.Patch(color=colors[i], label=cat, linewidth=0, alpha=0.95)
                      for i, cat in enumerate(categories)]
    leg = ax.legend(handles=legend_handles, title="Category", loc='upper left', bbox_to_anchor=(0, -0.18), ncol=3, frameon=False, fontsize=11, title_fontsize=12, handleheight=2.2, handlelength=2.2)
    for lh in leg.legend_handles:
        lh.set_height(10)
        lh.set_width(10)
        lh.set_linewidth(0)
        
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


async def generate_ai_feedback(transaction_type: str, amount: float, category: str, description: str) -> str:
    """Generate AI feedback for the transaction using Ollama's dolphin-mistral model."""
    try:
        prompt = f"""
        You are an unhinged financial advisor with a dark, shocking, offensive sense of humour who believes that 
        the best way to get people to make better financial decisions is by bullying them. 
        Roast the user's financial decision as brutally as possible, DO NOT HOLD BACK. Nothing is off-limits.
        
        Transaction details:
        - Type: {transaction_type}
        - Amount: ${amount:.2f}
        - Category: {category}
        - Description: {description}
        
        Keep the response to 1-4 sentences maximum. Make your client cry and everyone else laugh hysterically at their expense.
        
        Example responses:
        - "Your wallet doesn't need McDonald's, and neither does your waistline."
        - "No wonder your mom thought you'd never amount to anything, look at how you spend money."
        - "At this rate, you'll be living in a van down by the river. And not even a nice van."
        - "$200 for a gym membership? Looking at your snack budget, I don't think that's going to do much..."
        - "Makeup will just take you from ugly to slightly less ugly. Consider saving up for plastic surgery instead."
        """
        
        response = await asyncio.to_thread(
            ollama.chat,
            model='dolphin-mistral',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options={
                'temperature': 0.7,
                'max_tokens': 100
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error generating AI feedback: {e}")
        return "The AI is too shocked by your financial decisions to respond."

def get_recent_transactions_ui() -> str:
    """Format recent transactions for display."""
    transactions = budget_tracker.get_recent_transactions(5)
    if not transactions:
        return "No transactions yet. Add one to get started!"
    
    formatted = []
    for t in transactions:
        amount = f"${abs(t['amount']):.2f}"
        trans_type = "💵 Income" if t['type'] == 'income' else "💸 Expense"
        formatted.append(f"{t['date']} - {trans_type}: {t['category']} - {t['description']} - {amount}")
    
    return "\n\n".join(formatted)

def get_all_transactions_ui() -> str:
    """Format all transactions for display."""
    transactions = budget_tracker.get_recent_transactions(len(budget_tracker.transactions))
    if not transactions:
        return "No transactions yet. Add one to get started!"
    
    formatted = []
    for t in transactions:
        amount = f"${abs(t['amount']):.2f}"
        trans_type = "💵 Income" if t['type'] == 'income' else "💸 Expense"
        formatted.append(f"{t['date']} - {trans_type}: {t['category']} - {t['description']} - {amount}")
    
    return "\n\n".join(formatted)

def load_css_from_file(filename="y2k_styles.css"):
    """Load CSS from external file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: CSS file '{filename}' not found. Using default styles.")
        return ""
    except Exception as e:
        print(f"Error loading CSS file: {e}")
        return ""

def create_gradio_interface():
    """Create and return the Gradio interface with customizable theme."""
    # Load CSS from external file
    custom_css = load_css_from_file()
    # Force-apply title style to override caching issues
    title_style = """
    @import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');
    
    .y2k-title h1 {
        text-align: center !important;
        font-family: 'Bungee', 'Rubik Glitch', 'DynaPuff', cursive !important;
        font-size: 5em !important;
        letter-spacing: 2px !important;
        color: #FF1493 !important;
        margin: 0 0 0.1em 0 !important;
        text-shadow: 4px 4px 0px #8B0047, 8px 8px 0px rgba(255, 20, 147, 0.2) !important;
        line-height: 1 !important;
        text-transform: uppercase !important;
        -webkit-text-stroke: 1px #fff !important;
        text-stroke: 1px #fff !important;
    }
    """
    
    with gr.Blocks(title="budget bully", css=custom_css + title_style) as app:
        gr.HTML("""<link href='https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rubik+Glitch&display=swap' rel='stylesheet'>""")
        # Force label styling for input fields

        gr.Markdown("# budget bully", elem_classes=["y2k-title"])
        
        with gr.Tabs():
            # Tab 1: Add Transaction
            with gr.TabItem("💸 Add Transaction"):
                # Quick Balance Summary at the top
                gr.Markdown("### 💰 Quick Balance Check")
                with gr.Row():
                    output_balance = gr.Number(
                        label="💰 Current Balance",
                        value=budget_tracker.get_balance(),
                        precision=2,
                        elem_classes=["gradio-input"]
                    )
                    income_total = gr.Number(
                        label="💵 Total Income",
                        value=budget_tracker.get_income_total(),
                        precision=2,
                        elem_classes=["gradio-input"]
                    )
                    expense_total = gr.Number(
                        label="💸 Total Expenses",
                        value=budget_tracker.get_expense_total(),
                        precision=2,
                        elem_classes=["gradio-input"]
                    )
                
                # Main content area with transaction form and roast
                with gr.Row():
                    # Left column: Transaction form
                    with gr.Column(scale=1):
                        gr.Markdown("### ✨ Add New Transaction ✨")
                        
                        # Transaction type
                        transaction_type = gr.Radio(
                            choices=["expense", "income"],
                            label="💸 Transaction Type",
                            value="expense",
                            interactive=True,
                            elem_classes=["gradio-radio"]
                        )
                        
                        # Amount
                        amount = gr.Number(
                            label="💵 Amount ($)", 
                            precision=2, 
                            step=0.01,
                            elem_classes=["gradio-input"]
                        )
                        
                        # Category
                        category = gr.Dropdown(
                            label="📁 Category",
                            choices=budget_tracker.expense_categories,
                            value=budget_tracker.expense_categories[0],
                            elem_classes=["gradio-input"]
                        )
                        
                        # Description
                        description = gr.Textbox(
                            label="📝 Description",
                            elem_classes=["gradio-input"]
                        )
                        
                        # Date
                        date = gr.Textbox(
                            label="📅 Date (YYYY-MM-DD)",
                            value=datetime.now().strftime("%Y-%m-%d"),
                            elem_classes=["gradio-input"]
                        )
                        
                        # Add button
                        add_btn = gr.Button("Add Transaction", variant="primary", elem_classes=["gradio-button"])
                        
                        # Status message
                        status = gr.Textbox(
                            value="Ready to track your finances!",
                            label="Status",
                            interactive=False,
                            show_label=False,
                            elem_classes=["gradio-input"]
                        )
                        
                        # Quick Balance Summary
                        gr.Markdown("### 💰 Quick Balance")
                        balance_summary = gr.Textbox(
                            value=budget_tracker.get_balance_summary(),
                            label="",
                            interactive=False,
                            show_label=False,
                            lines=3,
                            elem_classes=["gradio-input"]
                        )
                    
                    # Right column: AI Roast and Recent Transactions
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Feedback")
                        
                        # AI Feedback - styled like a terminal
                        ai_feedback = gr.Textbox(
                            label="",
                            interactive=False,
                            visible=True,
                            lines=8, # Increased size
                            max_lines=10, # Increased size
                            show_label=False,
                            placeholder="Your financial decisions will be brutally roasted here... 🔥",
                            elem_classes=["gradio-input", "ai-feedback-terminal"], # Added terminal class
                            show_copy_button=True
                        )
                        
                        # Recent transactions directly under the roast
                        gr.Markdown("### 📜 Recent Transactions")
                        transactions_output_tab1 = gr.TextArea(
                            value=get_recent_transactions_ui(),
                            lines=10,
                            max_lines=12,
                            interactive=False,
                            show_copy_button=True,
                            label="",
                            show_label=False,
                            elem_classes=["gradio-input"]
                        )
                
                refresh_btn_tab1 = gr.Button("🔄 Refresh Recent Transactions", elem_classes=["gradio-button"])
                refresh_btn_tab1.click(
                    fn=lambda: get_recent_transactions_ui(),
                    outputs=transactions_output_tab1
                )
            
            # Tab 2: Analytics & Full History
            with gr.TabItem("📊 Analytics"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Summary Section
                        gr.Markdown("### 📊 Financial Dashboard")
                        with gr.Row():
                            output_balance_tab2 = gr.Number(
                                label="💰 Current Balance",
                                value=budget_tracker.get_balance(),
                                precision=2,
                                elem_classes=["gradio-input"]
                            )
                            income_total_tab2 = gr.Number(
                                label="💵 Total Income",
                                value=budget_tracker.get_income_total(),
                                precision=2,
                                elem_classes=["gradio-input"]
                            )
                            expense_total_tab2 = gr.Number(
                                label="💸 Total Expenses",
                                value=budget_tracker.get_expense_total(),
                                precision=2,
                                elem_classes=["gradio-input"]
                            )
                        
                        # Financial Goals Section
                        gr.Markdown("### 🎯 Financial Goals")
                        with gr.Row():
                            # Loans Goal Chart
                            with gr.Column():
                                gr.Markdown("#### 💳 Loans Repayment Goal")
                                with gr.Row():
                                    loans_goal_input = gr.Number(
                                        label="Target Amount",
                                        value=budget_tracker.loans_goal,
                                        precision=2,
                                        elem_classes=["gradio-input"]
                                    )
                                    loans_current = gr.Number(
                                        label="Amount Paid",
                                        value=budget_tracker.get_loans_total(),
                                        precision=2,
                                        interactive=False,
                                        elem_classes=["gradio-input"]
                                    )
                                loans_chart = gr.Plot(label="Loans Progress", elem_classes=["donut-chart"])
                            # Savings Goal Chart
                            with gr.Column():
                                gr.Markdown("#### 💰 Savings Goal")
                                with gr.Row():
                                    savings_goal_input = gr.Number(
                                        label="Target Amount",
                                        value=budget_tracker.savings_goal,
                                        precision=2,
                                        elem_classes=["gradio-input"]
                                    )
                                    savings_current = gr.Number(
                                        label="Amount Saved",
                                        value=budget_tracker.get_savings_total(),
                                        precision=2,
                                        interactive=False,
                                        elem_classes=["gradio-input"]
                                    )
                                savings_chart = gr.Plot(label="Savings Progress", elem_classes=["donut-chart"])
                        
                        # --- Time Range Filter (controls stacked bar chart) ---
                        with gr.Column(elem_classes=["chart-card"]):
                            gr.Markdown("### 📅 Select Time Range", elem_classes=["section-title"])
                            day_filter = gr.Radio(
                                choices=[("7 days", 7), ("14 days", 14), ("30 days", 30), ("90 days", 90), ("All Time", 3650)],
                                value=30,
                                label=None,
                                interactive=True,
                                elem_classes=["day-filter-group"]
                            )

                        # --- Stacked Bar Chart Section ---
                        with gr.Column(elem_classes=["chart-card"]):
                            gr.Markdown("### 📊 Daily Spending by Category", elem_classes=["section-title"])
                            stacked_bar_chart = gr.Plot(label=None, elem_classes=["gradio-plot"])

                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 All Transactions")
                        transactions_output_tab2 = gr.TextArea(
                            value=get_all_transactions_ui(),
                            lines=20,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True,
                            label="All Transactions",
                            elem_classes=["gradio-input"]
                        )
                        
                        refresh_btn_tab2 = gr.Button("🔄 Refresh All", elem_classes=["gradio-button"])
                        refresh_btn_tab2.click(fn=lambda: get_all_transactions_ui(), outputs=transactions_output_tab2)

        # --- Event Handlers ---
        def update_category_dropdown(trans_type):
            categories = (budget_tracker.income_categories if trans_type == 'income' else budget_tracker.expense_categories)
            return gr.Dropdown(choices=categories, value=categories[0] if categories else "", elem_classes=["gradio-input"])

        transaction_type.change(update_category_dropdown, inputs=transaction_type, outputs=category)

        def update_stacked_bar_chart(days):
            return create_stacked_bar_chart(days)

        day_filter.change(fn=update_stacked_bar_chart, inputs=day_filter, outputs=stacked_bar_chart)

        async def update_ui(transaction_type, amount, category, description, date, days):
            try:
                if amount:
                    budget_tracker.add_transaction(float(amount), category, description, date, transaction_type)
                    feedback_task = asyncio.create_task(generate_ai_feedback(transaction_type, float(amount), category, description))
                    ai_feedback_text = await feedback_task
                    status_msg = "Transaction added! That's hot!"
                else:
                    ai_feedback_text = "No transaction to add, but I'm watching..."
                    status_msg = "No amount entered."

                return [
                    budget_tracker.get_balance(), budget_tracker.get_income_total(), budget_tracker.get_expense_total(),
                    status_msg, budget_tracker.get_balance_summary(), ai_feedback_text, get_recent_transactions_ui(),
                    budget_tracker.get_balance(), budget_tracker.get_income_total(), budget_tracker.get_expense_total(),
                    get_all_transactions_ui(), budget_tracker.get_loans_total(), create_loans_donut(),
                    budget_tracker.get_savings_total(), create_savings_donut(), create_stacked_bar_chart(days)
                ]
            except Exception as e:
                return [
                    budget_tracker.get_balance(), budget_tracker.get_income_total(), budget_tracker.get_expense_total(),
                    f"OMG, error! {e}", budget_tracker.get_balance_summary(), "The AI is judging your error.",
                    get_recent_transactions_ui(), budget_tracker.get_balance(), budget_tracker.get_income_total(),
                    budget_tracker.get_expense_total(), get_all_transactions_ui(), budget_tracker.get_loans_total(),
                    create_loans_donut(), budget_tracker.get_savings_total(), create_savings_donut(), create_stacked_bar_chart(days)
                ]

        def wrap_update_ui(*args):
            return asyncio.run(update_ui(*args))

        add_btn.click(
            fn=wrap_update_ui,
            inputs=[transaction_type, amount, category, description, date, day_filter],
            outputs=[
                output_balance, income_total, expense_total, status, balance_summary, ai_feedback, transactions_output_tab1,
                output_balance_tab2, income_total_tab2, expense_total_tab2, transactions_output_tab2, 
                loans_current, loans_chart, savings_current, savings_chart, stacked_bar_chart
            ]
        )

        def update_loans_goal(new_goal):
            budget_tracker.loans_goal = new_goal
            return create_loans_donut()

        def update_savings_goal(new_goal):
            budget_tracker.savings_goal = new_goal
            return create_savings_donut()
        
        loans_goal_input.change(fn=update_loans_goal, inputs=loans_goal_input, outputs=loans_chart)
        savings_goal_input.change(fn=update_savings_goal, inputs=savings_goal_input, outputs=savings_chart)

        def update_ui_on_load():
            return [
                budget_tracker.get_balance(), budget_tracker.get_income_total(), budget_tracker.get_expense_total(),
                "*Sigh* Let's see the damage...", budget_tracker.get_balance_summary(), "Oh great, what did you do this time?",
                get_recent_transactions_ui(), budget_tracker.get_balance(), budget_tracker.get_income_total(),
                budget_tracker.get_expense_total(), get_all_transactions_ui(), budget_tracker.get_loans_total(),
                create_loans_donut(), budget_tracker.get_savings_total(), create_savings_donut(),
                create_stacked_bar_chart(30)
            ]
        
        app.load(
            fn=update_ui_on_load,
            inputs=None,
            outputs=[
                output_balance, income_total, expense_total, status, balance_summary, ai_feedback, transactions_output_tab1,
                output_balance_tab2, income_total_tab2, expense_total_tab2, transactions_output_tab2, 
                loans_current, loans_chart, savings_current, savings_chart, stacked_bar_chart
            ]
        )
        
        # Minimal footer
        gr.HTML("""
<div class='y2k-footer'>
  Fiscal irresponsibility is not hot.
</div>
""")

        return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
