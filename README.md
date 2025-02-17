# Chat to your Oracle Database 

A sophisticated AI-powered chatbot that interacts with an Oracle database to process natural language queries, analyze data, and perform CRUD operations. This Flask-based backend integrates OpenAI's GPT, LangChain, and SQLAlchemy, offering seamless interaction with structured data.

## Features

- **Natural Language to SQL**: Converts user queries into SQL statements to retrieve, analyze, or modify database records.
- **AI-Powered Query Refinement**: Enhances user input for optimal database interaction.
- **Data Analysis & Visualization**: Uses machine learning for clustering and generates insights with interactive charts.
- **Session Management**: Utilizes Redis for storing user sessions securely.
- **Automated Reports**: Periodic email reports with data insights.
- **Fine-Tuning**: Custom AI model training on structured HR database data.
- **Secure Authentication**: User authentication with password hashing and session control.

## Technologies Used

- **Backend**: Flask, Flask-Session, SQLAlchemy, oracledb
- **Database**: Oracle XE 21c
- **AI & NLP**: OpenAI API, LangChain
- **Machine Learning**: Scikit-learn (TF-IDF, KMeans)
- **Scheduling & Automation**: APScheduler, Schedule

## Installation

### Prerequisites
- Python 3.11+
- Oracle Database XE 21c (Installed locally)
- Redis (for session management)

### Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/ahmedali703/AI-Agent.git
   cd your-repo-name
   ```

2. **Create Virtual Environment & Install Dependencies**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**
   Create a `.env` file and configure the following:
   ```sh
   SECRET_KEY=your_secret_key
   ORACLE_USER=HR
   ORACLE_PASSWORD=HR
   ORACLE_DSN=localhost/xepdb1
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Application**
   ```sh
   python app.py
   ```

5. **Access the API**
   - The backend runs on `http://127.0.0.1:5002`
   - Example endpoints:
     - `POST /register` - User registration
     - `POST /login` - User authentication
     - `POST /chat` - AI chatbot interaction
     - `POST /feedback` - Collect user feedback



## API Endpoints

| Endpoint        | Method | Description |
|----------------|--------|-------------|
| `/register`    | POST   | User registration |
| `/login`       | POST   | User login |
| `/logout`      | POST   | User logout |
| `/chat`        | POST   | Chat interaction |
| `/feedback`    | POST   | Collects user feedback |

## License

MIT License. See `LICENSE` for details.

---

This project is actively maintained and welcomes contributions! Feel free to fork, create issues, or submit pull requests.

