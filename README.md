# 📂 Advisor Flask App - Deploy on Render.com

This project demonstrates how to **build and deploy a Python Flask web app on Render.com** with GitHub integration and security best practices.
## 🚀 Live App

👉 **Deployed on Render.com:**  
`https://your-app-name.onrender.com`  
*(Replace this with your actual Render app URL)*

## 📦 Project Structure

advisor_flask/
├── app.py
├── requirements.txt
├── Procfile
├── templates/
│   └── dashboard.html
├── static/        # Optional CSS/JS files
├── .gitignore
🛠️ Technologies Used
Flask (Python Web Framework)

Gunicorn (Production WSGI Server)

Render.com (Cloud Hosting)

GitHub (Version Control)

Gemini API, DeepSeek API, etc. (Optional)

⚙️ Flask App Setup
📄 app.py (Required Structure)
python
from flask import Flask, render_template
import os
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('dashboard.html')  # Or your target template

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
✔️ Always use os.environ.get('PORT') → Render needs this to bind the correct port.

📑 requirements.txt Example
text
Copy
Edit
Flask
gunicorn
setuptools
python-dotenv
yfinance
pandas
numpy==1.26.4
pandas-ta
google-generativeai
requests
✔️ You must pin numpy to version 1.26.4 to avoid compatibility issues with pandas-ta.

📂 Procfile Example
web: gunicorn app:app
✔️ This tells Render to run your Flask app using Gunicorn.

🚀 Step-by-Step: Deploy on Render.com
Create a free account at 👉 https://render.com

Click New > Web Service

Connect your GitHub repository → Select your Flask app repo

Configure:

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

(Optional) Add Environment Variables in the Render dashboard:

GEMINI_API_KEY = your_api_key_here
DEEPSEEK_API_KEY = your_api_key_here
Click Create Web Service → Render will auto-deploy your app having URL-

https://advisor-flask.onrender.com
✅ .gitignore Example
.env
__pycache__/
*.pyc
venv/
✔️ Always use .gitignore to prevent pushing secret files to GitHub.

🔒 Security Best Practices
✅ Never push .env files to GitHub.

✅ Store API keys in Render’s Environment Variables panel.

✅ If a key is accidentally pushed → revoke immediately.

✅ Use GitGuardian CLI to scan for secret leaks:

✅ Clean Git history with BFG Repo-Cleaner if needed.

✅ Credits
Deployed and tested by: [Harendra Yadav]
Special thanks to: OpenAI ChatGPT Assistant for full step-by-step guidance.

### ✅ Next Steps:
- Save this file as `README.md` in your GitHub project.
- Replace:
  - https://advisor-for-equity.onrender.com/ → your actual Render app URL.
  - `[Harendra Yadav]` 

