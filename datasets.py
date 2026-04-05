"""
Hardcoded dirty CSV datasets for each task level.
Each dataset() function returns a pandas DataFrame with intentional issues.
"""
import pandas as pd
import numpy as np


def easy_dataset() -> pd.DataFrame:
    """
    Employee dataset.
    Issues: exact duplicate rows, missing salary values.
    """
    data = {
        "id":         [1, 2, 3, 4, 2, 5, 6, 3, 7, 8],
        "name":       ["Alice", "Bob", "Carol", "Dave", "Bob", "Eve", "Frank", "Carol", "Grace", "Heidi"],
        "department": ["Eng", "HR", "Eng", "Finance", "HR", "Eng", "Finance", "Eng", "HR", "Finance"],
        "salary":     [90000, None, 85000, 72000, None, 95000, None, 85000, 68000, 71000],
        "age":        [30, 25, 35, 40, 25, 28, 45, 35, 29, 38],
    }
    return pd.DataFrame(data)


def easy_clean_dataset() -> pd.DataFrame:
    """Expected clean output for grading."""
    data = {
        "id":         [1, 2, 3, 4, 5, 6, 7, 8],
        "name":       ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"],
        "department": ["Eng", "HR", "Eng", "Finance", "Eng", "Finance", "HR", "Finance"],
        "salary":     [90000, 80200.0, 85000, 72000, 95000, 80200.0, 68000, 71000],  # mean filled
        "age":        [30, 25, 35, 40, 28, 45, 29, 38],
    }
    return pd.DataFrame(data)


def medium_dataset() -> pd.DataFrame:
    """
    Customer dataset.
    Issues: inconsistent date formats, mixed phone formats,
            inconsistent gender categories.
    """
    data = {
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "name":        ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"],
        "dob":         ["01/15/1990", "1985-06-20", "March 3, 1992", "07/22/1988",
                        "1995-11-30", "12/01/1980", "1993-04-17", "05/09/1991"],
        "phone":       ["(555)123-4567", "555.234.5678", "5553456789", "+1-555-456-7890",
                        "555 567 8901", "(555)678-9012", "555-789-0123", "5558901234"],
        "gender":      ["Female", "male", "F", "MALE", "female", "Male", "f", "Female"],
        "city":        ["New York", "Los Angeles", "New York", "Chicago",
                        "Los Angeles", "Chicago", "New York", "Chicago"],
    }
    return pd.DataFrame(data)


def medium_expected_phone_format(phone: str) -> str:
    """Normalize phone to (XXX)XXX-XXXX."""
    import re
    digits = re.sub(r'\D', '', phone)[-10:]
    return f"({digits[:3]}){digits[3:6]}-{digits[6:]}"


def hard_dataset() -> pd.DataFrame:
    """
    Sales dataset.
    Issues: duplicates, missing values, outliers in revenue,
            inconsistent date formats, mixed product name casing.
    """
    data = {
        "order_id":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10],
        "date":        ["2024-01-05", "Jan 6 2024", "2024-01-07", "01/08/2024",
                        "2024-01-09", "2024-01-10", "Jan 11 2024", "2024-01-12",
                        "2024-01-13", "Jan 6 2024", "2024-01-14"],
        "product":     ["Widget A", "widget a", "Widget B", "WIDGET C",
                        "Widget A", "widget b", "Widget C", "Widget A",
                        "Widget B", None, "Widget C"],
        "quantity":    [5, 3, None, 8, 2, 4, 6, None, 3, 3, 7],
        "revenue":     [500, 300, 450, 800, 200, 400, 600, 350, 300, 300, 999999],
        "region":      ["North", "South", "East", "West", "North",
                        None, "East", "West", "South", "South", "North"],
        "salesperson": ["Alice", "Bob", "Carol", "Dave", "Alice",
                        "Bob", "Carol", None, "Dave", "Bob", "Alice"],
    }
    return pd.DataFrame(data)
