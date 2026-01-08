import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import os

def get_tip_momentum_score():
    """
    Calculates the TIP momentum score.
    """
    try:
        tip = yf.Ticker("TIP")
        hist = tip.history(period="1y")
        
        # Calculate momentum scores
        r1 = (hist['Close'][-1] - hist['Close'][-22]) / hist['Close'][-22] if len(hist) >= 22 else 0
        r3 = (hist['Close'][-1] - hist['Close'][-64]) / hist['Close'][-64] if len(hist) >= 64 else 0
        r6 = (hist['Close'][-1] - hist['Close'][-127]) / hist['Close'][-127] if len(hist) >= 127 else 0
        
        momentum_score = (r1 + r3 + r6) / 3
        return momentum_score
    except Exception as e:
        print(f"Error calculating TIP momentum score: {e}")
        return None

def update_google_sheet(score):
    """
    Updates a Google Sheet with the TIP momentum score.
    """
    try:
        # Authenticate with Google Sheets
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Check for credentials in environment variables
        creds_json = os.getenv('GSPREAD_CREDENTIALS')
        if not creds_json:
            print("GSPREAD_CREDENTIALS environment variable not set.")
            return

        creds_dict = eval(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet
        sheet = client.open("Your Google Sheet Name").sheet1  # Replace with your sheet name

        # Find the cell to update (e.g., A1)
        # You might need to adjust the cell based on your sheet's structure
        sheet.update_acell('E13', score)
        print("Google Sheet updated successfully.")

    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

if __name__ == "__main__":
    tip_score = get_tip_momentum_score()
    if tip_score is not None:
        print(f"TIP Momentum Score: {tip_score}")
        # update_google_sheet(tip_score) # This line is commented out until you set up your Google Sheet and credentials
