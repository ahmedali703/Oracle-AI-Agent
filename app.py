from flask import Flask, request, jsonify, render_template, session, Response
from flask_session import Session
import redis
from flask_cors import CORS
import oracledb
import os
import re
import uuid
import schedule
import time
from sqlalchemy import create_engine, MetaData, Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from openai import OpenAI  
from dotenv import load_dotenv
import sys
import io
import threading
import numpy as np
import json
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import subprocess
import secrets
from langdetect import detect, LangDetectException
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI as LangOpenAI
import time

THRESHOLD_TIME = 2.0

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)  


# إعداد مفتاح سري للجلسات
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))  # مفتاح آمن

# إعداد Flask-Session لاستخدام Redis
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis.from_url("redis://localhost:6379") 
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True

Session(app)

# ===== إعداد اتصال أوراكل =====
os.environ["NLS_LANG"] = "AMERICAN_AMERICA.AL32UTF8"
os.environ["TNS_ADMIN"] = r"C:\app\Mopa\product\21c\dbhomeXE\instantclient"

try:
    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER", "HR"),
        password=os.getenv("ORACLE_PASSWORD", "HR"),
        dsn=os.getenv("ORACLE_DSN", "localhost/xepdb1")  # تأكد من صحة الـ SID أو Service Name
    )
    print("Successfully connected to Oracle database.")
except Exception as e:
    print(f"Error connecting to Oracle database: {e}")
    connection = None

# SQLAlchemy
# استخدام الـ Service Name في سلسلة الاتصال
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI", "oracle+oracledb://HR:HR@localhost/?service_name=xepdb1"))
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData(naming_convention={"skip_sorted_tables": True})
Base = declarative_base()

# نموذج المستخدم
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

    # تعريف نموذج المحادثة
class Conversation(Base):
    __tablename__ = 'conversations'
    conversation_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    question = Column(Text, nullable=False)  # يمكنك استخدام Text أو CLOB للبيانات الكبيرة
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback = Column(Text)  # حقل نصي لتخزين التعليقات
    classification = Column(String(100))  # الحقل الجديد لتخزين التصنيف

    user = relationship("User", backref="conversations")

# تعريف نموذج التغذية الراجعة
class Feedback(Base):
    __tablename__ = 'feedback'
    feedback_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'), nullable=False)
    rating_type = Column(Integer)  # تقييم رقمي مثلًا من 1 إلى 5
    comments = Column(Text)  # حقل نصي للتعليقات

    conversation = relationship("Conversation", backref="feedbacks")


# إنشاء الجداول إذا لم تكن موجودة
Base.metadata.create_all(bind=engine)

# إعداد OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # تهيئة العميل الجديد
lang_client = LangOpenAI(api_key=os.getenv("OPENAI_API_KEY"))



smtplib.debuglevel = 1


# إعداد تسجيل الأحداث (Logging)
logging.basicConfig(level=logging.ERROR , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Agent Base Class
class AgentBase:
    def __init__(self, name):
        self.name = name

    def process(self, input_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

# InputAgent Class
class InputAgent(AgentBase):
    def __init__(self, name):
        super().__init__(name)
        self.llm = LangOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.prompt_template = PromptTemplate.from_template(
            "Based on the following database schema: {schema}\n"
            "Improve the following question for better understanding and context:\n"
            "Question: {question}\n"
            "Improved Question:"
        )

    def process(self, user_input):
        start_time = time.time()
        schema_info = self.get_database_schema()  # Get schema details
        logger.info(f"{self.name} received input: {user_input}")
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        improved_question = chain.run(schema=schema_info, question=user_input)
        elapsed_time = time.time() - start_time
        logger.info(f"{self.name} improved input: {improved_question} (Processed in {elapsed_time:.2f} seconds)")
        return improved_question.strip()

    def get_database_schema(self):
        return get_all_table_schemas()



# ====================================================================
# الدوال المساعدة
# ====================================================================
# إنشاء PromptTemplate أساسي لتحسين الاستعلام (يمكن استخدامه في سلاسل LangChain)
prompt_template = PromptTemplate.from_template(
    "قم بتحسين الاستعلام التالي بناءً على السياق:\n"
    "السياق: {context}\n"
    "الاستعلام: {query}\n"
    "رسالة الخطأ: {error_message}\n"
    "الاستعلام المحسن:"
)
# باستخدام عامل الـ pipe لتحويل القالب إلى Runnable باستخدام lang_client
llm_chain = prompt_template | lang_client

def self_reflect_and_replan(query: str, error_message: str, context: dict) -> str:
    """
    تستخدم هذه الدالة سلسلة LLM لإعادة صياغة الاستعلام بناءً على رسالة الخطأ والسياق.
    """
    # استخدام متغير manual_prompt بدلاً من متغير غير معرف
    escaped_context = str(context).replace("{", "{{").replace("}", "}}")
    manual_prompt = f"""
لقد تم تنفيذ الاستعلام التالي:
{query}

ولكن ظهرت المشكلة التالية:
{error_message}

مع العلم بالسياق: {escaped_context}

قم بتوليد استعلام معدل يأخذ بعين الاعتبار تحسين بناء الجملة وأي مقترحات لتحسين الأداء.
"""
    # إنشاء سلسلة جديدة باستخدام النص الكامل للـ prompt مع lang_client
    manual_chain = PromptTemplate.from_template(manual_prompt) | lang_client
    new_query = manual_chain.invoke({})
    return new_query.strip()

def is_response_satisfactory(response: str) -> bool:
    """
    تتحقق هذه الدالة مما إذا كانت الاستجابة مرضية (غير فارغة ولا تحتوي على كلمة "error").
    """
    return bool(response.strip()) and "error" not in response.lower()

def auto_recursive_reasoning(prompt: str, max_depth=3, current_depth=0) -> str:
    """
    تطبيق استدلال تكراري لتحسين الاستجابة؛ إذا كانت الاستجابة غير مرضية يتم إعادة صياغة الـ prompt
    وإعادة المحاولة حتى الوصول إلى استجابة جيدة أو انتهاء عدد التكرارات.
    يقوم الكود بهروب الأقواس في prompt لتجنب تفسيرها كمتغيرات.
    """
    escaped_prompt = prompt.replace("{", "{{").replace("}", "}}")
    chain = PromptTemplate.from_template(escaped_prompt) | lang_client
    response = chain.invoke({})
    if is_response_satisfactory(response) or current_depth >= max_depth:
        return response.strip()
    else:
        new_prompt = f"راجع الاستجابة التالية وحسنها:\n{response}\nمع الأخذ في الاعتبار: {prompt}"
        return auto_recursive_reasoning(new_prompt, max_depth, current_depth + 1)


def generate_empty_result_response(question: str) -> str:
    """
    توليد رد طبيعي عندما يُرجع الاستعلام DataFrame فارغ.
    يقوم هذا الدالة بإرسال prompt إلى ChatGPT لتحليل السؤال وتقديم إجابة عامة 
    تُشير إلى عدم وجود بيانات مطابقة، دون استخدام أسماء حقول ثابتة وبصيغة Markdown.
    """
    prompt = f"""
You are a data analyst assistant. The SQL query executed for the question:
"{question}"
returned no results.

Based on the analysis of the question and historical trends, please provide a clear, concise, 
and natural language response in markdown format explaining that there are no matching records.
Do not include any specific field names or fixed data; instead, give a general explanation and any 
possible insights that might help the user.
"""
    # نستخدم عميل OpenAI الأصلي كما هو (client) لاستدعاء chat completions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "developer", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()




def execute_sql_with_iterative_refinement(query, max_retries=5):
    """
    تنفيذ استعلام SQL مع تحسين تكراري (Iterative Query Refinement) باستخدام التفكير الذاتي.
    تُدمج الدوال:
      - self_reflect_and_replan: لإعادة صياغة الاستعلام عند وقوع خطأ.
      - is_response_satisfactory: لفحص ما إذا كانت الاستجابة المُحسّنة جيدة.
      - auto_recursive_reasoning: لمحاولة تحسين الاستجابة تلقائيًا.
    
    إذا فشلت كل المحاولات تُعاد DataFrame فارغ.
    """
    current_query = query
    # نفترض أن get_all_table_schemas() مُعرفة وتُعيد وصف الجداول
    context = {"schema": get_all_table_schemas()}
    attempt = 0
    error_message = ""
    
    while attempt < max_retries:
        try:
            df = execute_sql_query(current_query)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            error_message = str(e)
            logging.error(f"Attempt {attempt+1} failed with error: {error_message}")
        
        # إعادة صياغة الاستعلام باستخدام self_reflect_and_replan
        refined_query = self_reflect_and_replan(current_query, error_message, context)
        # إذا لم تكن الاستجابة المُحسنة مرضية، استخدام auto_recursive_reasoning لتحسينها
        if not is_response_satisfactory(refined_query):
            refined_query = auto_recursive_reasoning(refined_query, max_depth=3)
        
        current_query = refined_query
        logging.info(f"Retrying with refined query (attempt {attempt+1}/{max_retries}): {current_query}")
        attempt += 1
    
    # في حالة استنفاذ المحاولات، نجرب تنفيذ الاستعلام مرة أخيرة؛ وإن فشل نُعيد DataFrame فارغ
    try:
        df = execute_sql_query(current_query)
        return df if df is not None else pd.DataFrame()
    except Exception as final_e:
        logging.error(f"Final attempt failed: {final_e}")
        return pd.DataFrame()



import json

def save_conversation(user_id, question, answer, classification, feedback=None):
    try:
        # تجميع المحادثة في كائن JSON
        conversation_data = {
            "question": question,
            "answer": answer,
            "classification": classification
        }

        # تحويل الكائن إلى سلسلة JSON
        response_json = json.dumps(conversation_data, ensure_ascii=False)

        # إنشاء سجل جديد في قاعدة البيانات
        db_session = SessionLocal()
        new_conversation = Conversation(
            user_id=user_id,
            question=question,  # يمكن حذف هذا الحقل إذا كنت تريد حفظ كل شيء في `response`
            response=response_json,  # حفظ المحادثة كـ JSON
            feedback=feedback,
            classification=classification
        )
        db_session.add(new_conversation)
        db_session.commit()
        db_session.refresh(new_conversation)
        db_session.close()

        # إرجاع معرف المحادثة
        return new_conversation.conversation_id
    except Exception as e:
        print(f"Error saving conversation: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def fetch_all_data():
    db_session = SessionLocal()
    # استخراج المحادثات
    conversations = db_session.query(Conversation).all()
    # استخراج التغذية الراجعة
    feedbacks = db_session.query(Feedback).all()
    db_session.close()

    # تحويل المحادثات إلى قائمة قواميس
    conv_data = [{
        "conversation_id": conv.conversation_id,
        "user_id": conv.user_id,
        "question": conv.question,
        "response": conv.response,
        "timestamp": conv.timestamp,
        "feedback": conv.feedback
    } for conv in conversations]

    # تحويل التغذية الراجعة إلى قائمة قواميس
    feedback_data = [{
        "feedback_id": fb.feedback_id,
        "conversation_id": fb.conversation_id,
        "rating_type": fb.rating_type,
        "comments": fb.comments
    } for fb in feedbacks]

    return conv_data, feedback_data

def analyze_question_patterns():
    conv_data, _ = fetch_all_data()
    df_conversations = pd.DataFrame(conv_data)

    # تجهيز نص الأسئلة وتحويله إلى متجهات
    vectorizer = TfidfVectorizer(stop_words='arabic')
    X = vectorizer.fit_transform(df_conversations['question'].astype(str))

    # تطبيق KMeans للتجميع
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    df_conversations['cluster'] = kmeans.labels_
    pattern_counts = df_conversations['cluster'].value_counts()
    print("أنماط الأسئلة الأكثر شيوعاً:\n", pattern_counts)


# دالة إرسال البريد الإلكتروني بصيغة HTML
def send_email_html(to_addresses, subject, html_body, smtp_server="apexexperts.net", smtp_port=465, username="ai@apexexperts.net", password="Ahmed@_240615"):
    msg = MIMEText(html_body, "html", "utf-8")
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = username
    # إذا كانت القائمة، قم بتجميعها لرأس الرسالة
    if isinstance(to_addresses, list):
        msg["To"] = ", ".join(to_addresses)
    else:
        msg["To"] = to_addresses
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(username, password)
        server.sendmail(username, to_addresses if isinstance(to_addresses, list) else [to_addresses], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_weekly_report():
    subject = "التقرير الأسبوعي"
    body = "<h1>التقرير الأسبوعي</h1><p>هذا هو التقرير الأسبوعي.</p>"
    to_email = "ahmed-alsaied@msn.com"
    send_email_html(to_email, subject, body)

def send_weekly_report_with_chart():
    logger.info("Running send_weekly_report_with_chart...")
    subject = "التقرير الأسبوعي مع الرسم البياني"
    body = "<h1>التقرير الأسبوعي</h1><p>هذا هو التقرير الأسبوعي.</p>"
    to_email = "recipient@example.com"
    send_email_html(to_email, subject, body)

def dicts_to_html_table(data):
    if not data:
        return "<p>لا توجد بيانات.</p>"
    
    df = pd.DataFrame(data)
    html_table = df.to_html(classes="min-w-full bg-white border border-blue-200 rounded-lg text-center", index=False)
    
    # تحسين التنسيق باستخدام Tailwind CSS
    styled_table = f"""
    <div class="overflow-x-auto">
        {html_table}
    </div>
    """
    return styled_table


def get_all_table_schemas():
    if not connection:
        print("No database connection.")
        return ""
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM ALL_TAB_COLUMNS
                WHERE OWNER = 'HR'
                ORDER BY TABLE_NAME, COLUMN_ID
            """)
            rows = cursor.fetchall()
        table_schemas = {}
        for row in rows:
            table, column, data_type = row
            if table not in table_schemas:
                table_schemas[table] = []
            table_schemas[table].append(f"{column} ({data_type})")
        schema_summary = ""
        for table, columns in table_schemas.items():
            schema_summary += f"\nTable {table} has the following columns:\n"
            for col in columns:
                schema_summary += f"- {col}\n"
        return schema_summary
    except Exception as e:
        print(f"Error fetching table schemas: {e}")
        return ""
    
def get_ddl_for_table(table_name):
    """
    تستخرج DDL لجدول معين وتنسيقه ككود SQL.
    """
    sql_query = f"SELECT DBMS_METADATA.GET_DDL('TABLE', '{table_name.upper()}') AS DDL FROM DUAL"
    df = execute_sql_with_iterative_refinement(sql_query, max_retries=5)
    
    if df is not None and not df.empty:
        ddl_text = df['DDL'].iloc[0]
        # تنسيق النتيجة كمقطع كود SQL مع إزالة الأسطر الفارغة غير الضرورية
        cleaned_ddl = "\n".join([line for line in ddl_text.split("\n") if line.strip() != ""])
        return f"```sql\n{cleaned_ddl}\n```"
    else:
        return None


def classify_question(question):
    """
    يصنف السؤال إلى أحد الأقسام الأربعة:
    - 'db_sql'    : استعلام SQL عادي (SELECT)
    - 'db_analysis': تحليل متقدم/إحصائي
    - 'db_action' : أوامر تعديل (INSERT / UPDATE / DELETE)
    - 'general'   : سؤال عام
    """

    question_lower = question.lower().strip()

    # ===== مفاتيح إجراء التعديل (Action) =====
    action_keywords = [
        "insert", "update", "delete", "remove", "add", "create record", "create table",
        "send mail", "create restful services",
        "drop", "truncate", "انشاء", "إضافة", "حذف", "تحديث", 
        "تعديل", "add column", "rename"
    ]

    # ===== مفاتيح تحليلية (Analysis) =====
    analysis_keywords = [
        "تحليل", "احصاء", "إحصاء", "إحصائي","رسم بياني"
    ]

    # ===== مفاتيح SQL (Reading) =====
    db_sql_keywords = [
        "select", "show me", "fetch", "retrieve","pivot"
        "استعلام", "query", "بيانات الجدول", "قائمة الأعمدة",
        "sum", "مجموع", "اجمالي", "إجمالي", "متوسط",
        "maximum", "minimum", "order by", "limit",
        "عرض", "اظهار", "   "
    ]

    # ===== مفاتيح أسئلة عامة (General) =====
    general_keywords = [
        "what is", "explain", "why", "how to", "كيفية",
        "عام", "معلومة", "what are", "متى", "أين", "لماذا"
    ]

    # 1) لو رصدنا action_keywords
    for ak in action_keywords:
        if ak in question_lower:
            print("Local classification => db_action")
            return "db_action"

    # 2) لو رصدنا analysis
    for kw in analysis_keywords:
        if kw in question_lower:
            print("Local classification => db_analysis")
            return "db_analysis"

    # 3) لو رصدنا db_sql
    for kw in db_sql_keywords:
        if kw in question_lower:
            print("Local classification => db_sql")
            return "db_sql"

    # 4) لو رصدنا general_keywords
    for kw in general_keywords:
        if kw in question_lower:
            print("Local classification => general")
            return "general"

    # 5) إذا لم نعثر على تصنيف واضح محليًا => نلجأ لـGPT
    try:
        schema_summary = get_all_table_schemas()  # يفترض أنها دالة لديك تعيد وصف الجداول
        prompt = f"""
You are a classification assistant. You have the following database schema:

{schema_summary}

You must classify any user question into exactly one category:
1) db_sql: direct query about the database, e.g. SELECT, pivot table retrieving rows/columns directly, "Show me employees"
2) db_analysis: advanced analysis or statistics on the data, e.g. "Calculate average salary or distribution"
3) db_action: modifying the database, e.g. "INSERT, UPDATE, DELETE, DROP, CREATE TABLE"
4) general: if not related to the HR database or no direct data/analysis request.

Important rules:
- If user question includes "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", or similar, classify as db_action.
- If user question includes "mean, distribution, correlation, statistic", or advanced analysis terms, classify as db_analysis.
- If user question includes "select, show me, fetch, retrieve" from a table, or direct request of rows, classify as db_sql.
- If user question is unrelated to the HR schema, or is general knowledge, classify as general.
- Use EXACT category name: db_sql, db_analysis, db_action, or general. No extra words.

Examples:
1) "Add new column to EMPLOYEES" => db_action
2) "Compute average salary of employees" => db_analysis
3) "SELECT * from employees" => db_sql
4) "What is AI?" => general

Now, classify the user question below:

Question: "{question}"
Answer with exactly one of: db_sql, db_analysis, db_action, or general.
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        classification = response.choices[0].message.content.strip().lower()
        print(f"GPT classification => {classification}")

        if 'db_sql' in classification:
            return 'db_sql'
        elif 'db_analysis' in classification:
            return 'db_analysis'
        elif 'db_action' in classification:
            return 'db_action'
        else:
            return 'general'
    except Exception as e:
        print(f"Error classifying question with GPT: {e}")
        return 'general'

def translate_ar_to_en(text):
    try:
        print("Translating:", text)
        translated = GoogleTranslator(source='ar', target='en').translate(text)
        print("Translated:", translated)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def clean_sql_query(sql_query):
    sql_query = sql_query.strip()
    sql_query = re.sub(r';$', '', sql_query)
    sql_query = re.sub(r'[؛؛]', '', sql_query)
    sql_query = sql_query.replace('\u200f', '')
    return sql_query

def natural_language_to_sql(question, is_chart=False, improved_question=None):
    """
    يحول أسئلة تتعلق بعرض البيانات (SELECT) إلى جملة SQL.
    """
    try:
        question_to_use = improved_question if improved_question else translate_ar_to_en(question)
        schema_summary = get_all_table_schemas()

        if is_chart:
            prompt = f"""
Translate the following question into a SQL query that returns data suitable for a bar chart with two columns: 'label' and 'value'. Provide only the SQL code without any explanations or additional text. Do not include a semicolon at the end.

Here is the schema of the database:
{schema_summary}

Question: '{question_to_use}'
"""
        else:
            prompt = f"""
Translate the following question into a SQL query. Provide only the SQL code without any explanations or additional text. Do not include a semicolon at the end.

Here is the schema of the database:
{schema_summary}

Question: '{question_to_use}'
"""

        print(f"Sending prompt to OpenAI: {prompt}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant that translates natural language questions into SQL queries compatible with Oracle Database."},
                {"role": "user", "content": prompt}
            ]
        )
        sql_query = response.choices[0].message.content.strip()
        print(f"Generated SQL Query: {sql_query}")
        return clean_sql_query(sql_query)
    except Exception as e:
        print(f"Error generating SQL query: {e}")
        return None

def natural_language_to_dml_action(question):
    """
    يحوّل الأسئلة المتعلقة بالتعديل (INSERT/UPDATE/DELETE)
    إلى جملة SQL خاصة بالأكشن.
    """
    try:
        question_en = translate_ar_to_en(question)
        schema_summary = get_all_table_schemas()

        prompt = f"""
You are an Oracle database expert. Always generate valid Oracle syntax, with the ability to execute any complex queries, as well as add, modify, and delete data. You have complete knowledge of all tables and data in the database and can provide effective solutions to any database-related issues. Your tasks include:

Executing complex SQL queries:

Do not use multi-row insert with VALUES (...),(...) style.

Use either multiple INSERT statements or the Oracle INSERT ALL syntax. 

Writing queries to extract required data from tables.

Using JOIN, GROUP BY, HAVING, SUBQUERIES, and other advanced operations.

Optimizing queries to ensure optimal performance.

Managing data:

Adding new data to tables using INSERT.

Updating existing data using UPDATE.

Deleting data using DELETE.

Analyzing data:

Analyzing relationships between tables and understanding the database structure.

Identifying data issues and providing solutions to fix them.

Creating and modifying tables:

Creating new tables using CREATE TABLE.

Modifying table structures using ALTER TABLE.

Dropping tables using DROP TABLE.

Managing indexes:

Creating indexes to improve query performance.

Analyzing existing indexes and determining the need for modifications.

Providing reports and results:

Delivering clear and organized results for queries.

Generating reports summarizing the required data.

Troubleshooting:

Analyzing errors in queries and fixing them.

Providing solutions for any issues related to data integrity or performance.

You are familiar with all tables, columns, and relationships between them, and you can provide accurate and effective answers to any questions or requests related to Oracle databases.

Database schema:
{schema_summary}

User request: '{question_en}'
"""

        print(f"Sending 'action' prompt to OpenAI:\n{prompt}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "developer", "content": "You are an assistant that writes Oracle-compatible DML statements."},
                {"role": "user", "content": prompt}
            ]
        )
        action_sql = response.choices[0].message.content.strip()
        print(f"Generated DML SQL: {action_sql}")
        return clean_sql_query(action_sql)
    except Exception as e:
        print(f"Error generating DML action: {e}")
        return None

def execute_sql_query(sql_query):
    """
    ينفّذ استعلام SELECT على قاعدة البيانات ويعيد DataFrame.
    """
    if not connection:
        print("No database connection.")
        return None
    try:
        with connection.cursor() as cursor:
            print(f"Executing SQL Query: {sql_query}")
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # تحويل جميع الأعمدة من نوع CLOB إلى نص عادي
            processed_rows = []
            for row in rows:
                processed_row = []
                for col in row:
                    if isinstance(col, oracledb.LOB):
                        processed_row.append(str(col.read()))
                    else:
                        processed_row.append(str(col) if col is not None else "")
                processed_rows.append(processed_row)
            
            # الحصول على أسماء الأعمدة
            columns = [desc[0] for desc in cursor.description]
            
            # إنشاء DataFrame
            df = pd.DataFrame(processed_rows, columns=columns)
            df = df.fillna("")
            
            # تنظيف أسماء الأعمدة
            df.columns = [col.replace(" ", "_").upper() for col in df.columns]
            
            print(f"Query Results:\n{df.to_string(index=False)}")
            return df
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None

def execute_sql_action(sql_statement):
    """
    ينفّذ (INSERT / UPDATE / DELETE / PL/SQL) على قاعدة البيانات ويعمل commit.
    يعيد (success, message, rows_affected).
    """
    if not connection:
        print("No database connection.")
        return False, "No DB connection", 0
    
    try:
        with connection.cursor() as cursor:
            # تنظيف الاستعلام من الرموز غير الضرورية
            cleaned_sql = sql_statement.strip()
            
            # إزالة أي نص غير SQL (مثل الشروحات)
            if "```sql" in cleaned_sql:
                cleaned_sql = cleaned_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in cleaned_sql:
                cleaned_sql = cleaned_sql.split("```")[1].strip()
            
            # إزالة أي فواصل منقوطة (;) في النهاية
            cleaned_sql = cleaned_sql.rstrip(';')
            
            # إزالة أي مسافات زائدة
            cleaned_sql = ' '.join(cleaned_sql.split())
            
            print(f"Executing SQL Action: {cleaned_sql}")
            
            # تحديد ما إذا كان الاستعلام عبارة عن PL/SQL
            is_plsql = cleaned_sql.lower().startswith(("begin", "declare"))
            
            if is_plsql:
                # تنفيذ كتلة PL/SQL
                cursor.execute(cleaned_sql)
                rows_affected = 0  # PL/SQL لا يعيد عدد Affected rows
            else:
                # تنفيذ SQL عادي
                cursor.execute(cleaned_sql)
                rows_affected = cursor.rowcount  # عدد Affected rows
            
            connection.commit()
            return True, "تم تنفيذ العملية بنجاح", rows_affected
    except Exception as e:
        print(f"Error executing SQL action: {e}")
        return False, str(e), 0

def get_salary_summary():
    """
    مثال بسيط: استعلام يُعيد (MIN(SALARY), MAX(SALARY), AVG(SALARY), COUNT(*))
    يمكنك التعديل حسب ما تريد تلخيصه.
    """
    if not connection:
        return {}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(SALARY), MAX(SALARY), AVG(SALARY), COUNT(*) FROM EMPLOYEES")
            row = cursor.fetchone()
            if row:
                min_sal, max_sal, avg_sal, cnt = row
                return {
                    "min_salary": float(min_sal or 0),
                    "max_salary": float(max_sal or 0),
                    "avg_salary": float(avg_sal or 0),
                    "total_employees": int(cnt or 0)
                }
    except:
        pass
    return {}

def generate_chart(data, x_key, y_key):
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_key], data[y_key], color='skyblue')
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"Sum of {y_key} by {x_key}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = f"./static/charts/{uuid.uuid4().hex}.png"
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    return f"./static/charts/{os.path.basename(chart_path)}"

def remove_markdown_fences(code_text):
    code_text = re.sub(r"```python\s*", "", code_text)
    code_text = re.sub(r"```", "", code_text)
    return code_text

def exec_python_code(code, df):
    code = remove_markdown_fences(code)

    old_stdout = sys.stdout
    mystdout = io.StringIO()
    sys.stdout = mystdout

    local_env = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt
    }

    try:
        exec(code, local_env)
    except Exception as e:
        sys.stdout = old_stdout
        error_output = mystdout.getvalue()
        tb = traceback.format_exc()
        error_output += f"\n\nحدث استثناء:\n{str(e)}\nTraceback:\n{tb}"
        return f"خطأ أثناء تنفيذ الكود:\n{error_output}", ""

    output = mystdout.getvalue()
    sys.stdout = old_stdout
    return "", output


def generate_response_with_context(question):
    conversation_history = session.get('conversation_history', [])
    context = " ".join([msg["content"] for msg in conversation_history])
    prompt = f"مع مراعاة السياق التالي: {context}\nالسؤال: {question}\nالإجابة:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

import json
from langdetect import detect, LangDetectException
from openai import ChatCompletion

def generate_summary(output, question):
    """
    تحليل البيانات وإنشاء ملخص ديناميكي باستخدام الذكاء الاصطناعي.
    """
    # اكتشاف لغة السؤال
    try:
        language = detect(question)
    except LangDetectException:
        language = 'en'

    try:
        # تحويل المخرجات إلى JSON
        data = json.loads(output)
        labels = data.get("labels", [])
        values = data.get("values", [])

        # التحقق من تطابق البيانات
        if not labels or not values or len(labels) != len(values):
            return "Cannot generate summary due to data mismatch." if language != 'ar' else "لا يمكن توليد ملخص نظرًا لعدم تطابق البيانات."

        # إعداد نص لتحليل البيانات بواسطة الذكاء الاصطناعي
        prompt = f"""
        You are a data analyst assistant. Analyze the following data and generate a dynamic summary in the same language as the user's question:

        - Question: "{question}"
        - Data Labels: {labels}
        - Data Values: {values}

        Your summary should include:
        - Key insights based on the data (e.g., highest value, lowest value, total, averages, trends, etc.).
        - Provide actionable recommendations if applicable.
        - Write the summary in the same language as the user's question.

        Output the result in plain text.
        """

        # إرسال الطلب إلى OpenAI API
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "developer", "content": "You are an advanced data analyst capable of generating insights dynamically."},
            {"role": "user", "content": prompt}
        ]
    )
        # استخراج النص من استجابة الذكاء الاصطناعي
        summary = response.choices[0].message.content.strip()

                # إضافة اتجاه RTL إذا كانت اللغة عربية
        if language == 'ar':
            rtl_mark = "\u202B"  # علامة بدء النص من اليمين إلى اليسار
            rtl_reset = "\u202C"  # علامة إعادة النص إلى الوضع الطبيعي
            summary = f"{rtl_mark}{summary}{rtl_reset}"

        return summary

    except json.JSONDecodeError:
        return "Unable to interpret analysis output." if language != 'ar' else "تعذر تفسير مخرجات التحليل."
    except Exception as e:
        return f"Error in generating summary: {str(e)}"

    
# ====================================================================
#             المسارات
# ====================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "يرجى إدخال اسم المستخدم وكلمة المرور"}), 400

        db = SessionLocal()
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return jsonify({"error": "اسم المستخدم موجود بالفعل"}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        db.close()

        # تسجيل الدخول تلقائي بعد التسجيل
        session['username'] = username
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['memory'] = {}

        return jsonify({"message": f"تم تسجيل المستخدم {username} بنجاح!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "يرجى إدخال اسم المستخدم وكلمة المرور"}), 400

        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({"error": "اسم المستخدم أو كلمة المرور غير صحيحة"}), 401

        db.close()

        # تسجيل الدخول وتعيين الجلسة
        session['username'] = username
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['memory'] = {}

        return jsonify({"message": f"تم تسجيل الدخول بنجاح كـ {username}!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({"message": "تم تسجيل الخروج بنجاح. تم مسح الـ Cookies"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        username = data.get('username')
        question = data.get('question')

        if not username:
            return jsonify({'error': 'Username is required.'}), 400
        if 'username' not in session or session['username'] != username:
            return jsonify({'error': 'User not logged in.'}), 401
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
            
        user = SessionLocal().query(User).filter(User.username == username).first()
        if not user:
            return jsonify({'error': 'User not found.'}), 404

        input_agent = InputAgent("InputAgent")
        improved_question = input_agent.process(question)

        question_lower = improved_question.lower()

        # تعيين قيمة افتراضية لـ `answer`
        answer = "حدث خطأ أثناء معالجة السؤال."

        classification = classify_question(question)
        print(f"Question classification: {classification}")

        # معالجة خاصية إرسال البريد الإلكتروني
        if ("send mail" in question_lower or 
            "send an email" in question_lower or 
            "إرسال بريد" in question_lower or 
            "ارسل بريد" in question_lower):

            emails = []   

            # حالة: إرسال بريد لجميع الموظفين
            if "all employees" in question_lower or "كل الموظفين" in question_lower:
                query = "SELECT EMAIL FROM EMPLOYEES"
                df_emails = execute_sql_with_iterative_refinement(query, max_retries=5)
                if df_emails is not None and not df_emails.empty:
                    emails = df_emails['EMAIL'].dropna().tolist()

            # حالة: إرسال بريد لكل من يعمل بوظيفة معينة
            elif "job" in question_lower or "وظيفة" in question_lower:
                job_match = re.search(r'وظيفة\s+(\w+)', question_lower)
                if job_match:
                    job_title = job_match.group(1)
                    query = f"SELECT EMAIL FROM EMPLOYEES WHERE LOWER(JOB_ID) LIKE '%{job_title.lower()}%'"
                    df_emails = execute_sql_with_iterative_refinement(query, max_retries=5)
                    if df_emails is not None and not df_emails.empty:
                        emails = df_emails['EMAIL'].dropna().tolist()

            # حالة: إرسال بريد إلى موظف محدد بناءً على الاسم الأول أو الأخير
            else:
                name_match = re.search(r'الى\s+(\w+)', question_lower)
                recipient_name = name_match.group(1) if name_match else None

                if recipient_name:
                    query = f"""
                    SELECT EMAIL FROM EMPLOYEES 
                    WHERE LOWER(FIRST_NAME) = '{recipient_name.lower()}'
                    OR LOWER(LAST_NAME) = '{recipient_name.lower()}'
                    """
                    df_emails = execute_sql_with_iterative_refinement(query, max_retries=5)
                    if df_emails is not None and not df_emails.empty:
                        emails = df_emails['EMAIL'].dropna().tolist()

            # التحقق من العثور على عناوين بريد إلكتروني
            if not emails:
                return jsonify({"error": "لا يمكن تحديد عناوين البريد الإلكتروني للمستلمين."}), 400

            # تنظيف عناوين البريد الإلكتروني للتأكد من صحتها
            emails = [email.strip() for email in emails if email.strip()]
            recipients_str = ", ".join(emails)

            conversation_history = session.get('conversation_history', [])
            last_assistant_msg = "لا توجد نتائج سابقة."
            last_assistant_data = None  # سنحاول استخراج بيانات من الرسائل السابقة

            if conversation_history:
                for msg in reversed(conversation_history):
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        last_assistant_msg = msg['content']
                        break

            html_body = f"""
            <html>
            <body>
                <h2>مرسل من الوكيل الذكي 😊</h2>
                <p>مرحبًا،</p>
                <p>هذه هي النتيجة السابقة:</p>
                <pre>{last_assistant_msg}</pre>
            </body>
            </html>
            """
            subject = "نتيجة الاستعلام من الوكيل الذكي 😊"

            email_sent = send_email_html(emails, subject, html_body)
            if email_sent:
                # تحديث تاريخ المحادثة
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": f"تم إرسال البريد الإلكتروني بنجاح إلى {recipients_str} 😊."})
                session['conversation_history'] = conversation_history

                return jsonify({
                    "results": [{"answer": f"تم إرسال البريد الإلكتروني بنجاح إلى {recipients_str} 😊."}],
                    "classification": "email"
                }), 200
            else:
                return jsonify({"error": "فشل في إرسال البريد."}), 500

        # =============== GENERAL ===============
        if classification == 'general':
            conversation_history = session.get('conversation_history', [])
            system_content = "You are a helpful assistant."
            if 'assistant_name' in session.get('memory', {}):
                assistant_name = session['memory']['assistant_name']
                system_content = f"You are {assistant_name}, a helpful assistant."

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "developer", "content": system_content},
                        *conversation_history,
                        {"role": "user", "content": question}
                    ]
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error in GPT response: {e}")
                answer = "حدث خطأ أثناء توليد الإجابة. يرجى المحاولة مرة أخرى."

            # حفظ المحادثة في قاعدة البيانات
            user = SessionLocal().query(User).filter(User.username == username).first()
            conv_id = None
            if user:
                conv_id = save_conversation(user.id, question, answer, classification='general')

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            session['conversation_history'] = conversation_history

            return jsonify({
                            'results': [{'answer': answer}],
                            'assistant_answer': answer,  # <-- أضف هذا
                            'classification': 'general',
                            'conversation_id': conv_id
                        })



        # =============== DB_ANALYSIS ===============
        if classification == 'db_analysis':
            translated_en = translate_ar_to_en(question).lower()
            is_chart = any(word in translated_en for word in ["statistics", "chart", "graph", "plot"])
            base_sql = natural_language_to_sql(        question=question,
                                                        is_chart=is_chart,
                                                        improved_question=improved_question)
            if not base_sql:
                return jsonify({'error': 'فشل في توليد استعلام SQL لجلب البيانات.'}), 500
            
                    # تنفيذ الاستعلام مع خاصية إعادة المحاولة
            df = execute_sql_with_iterative_refinement(base_sql, max_retries=5)

            
            if df is None or df.empty:
                    empty_answer = generate_empty_result_response(question)
                    # يمكنك حفظ المحادثة إذا أردت:
                    conv_id = save_conversation(user.id, question, empty_answer, classification='empty')
                    return jsonify({
                        'assistant_answer': empty_answer,
                        'classification': 'empty',
                        'conversation_id': conv_id
                      })

            df = df.fillna(0)
            columns_list = list(df.columns)
            sample_records = df.head(20).fillna(0).to_dict(orient='records')

            analysis_prompt = f"""
أنت محلل بيانات في بايثون. لدينا DataFrame باسم df يحوي الأعمدة:
{columns_list}

عينة من البيانات:
{sample_records}

السؤال التحليلي: "{question}"

*** هام ***:
- لا تستخدم تنسيق سلسلة مثل % أو f-string على كائنات معقّدة.
- استخدم json.dumps() لإخراج النتائج في شكل JSON.
- **يجب** أن يكون الخرج النهائي في شكل:
  result = {{
    "labels": [...],   # أي عمود نصي أو اسمه department_id أو city أو ... 
    "values": [...]    # أي مجموع/أرقام أو اسمه total_salary أو salary ...
  }}
  - إذا كان السؤال يتطلب إنشاء مخطط مكدس (مثل إجمالي الرواتب وعدد الموظفين)، قم بإنشاء مخطط مكدس .
- استخدم print(json.dumps(result)) لعرض النتائج.
print(json.dumps(result))

(لا تستخدم مفاتيح أخرى مثل department_id أو total_salary.)

اكتب كود بايثون فقط (بدون علامات ```python) لإجراء المطلوب على df. استخدم print() لعرض النتائج.
"""
            try:
                python_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "developer", "content": "You are a Python data analyst with a DataFrame named df."},
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                python_code = python_response.choices[0].message.content.strip()
            except Exception as e:
                return jsonify({"error": f"خطأ في توليد كود التحليل: {str(e)}"}), 500

            print("GPT generated code:\n", python_code)
            err, output = exec_python_code(python_code, df)
            if err:
                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({
                    "role": "assistant",
                    "content": "هذا كود التحليل ولكن حدث خطأ:\n" + python_code
                })
                session['conversation_history'] = conversation_history
                return jsonify({"error": err, "analysis_code": python_code}), 500
            else:
                chart_data = None
                try:
                    chart_data = json.loads(output)
                except json.JSONDecodeError:
                    chart_data = None

                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({
                    "role": "assistant",
                    "content": "كود التحليل:\n" + python_code + "\n\nالمخرجات:\n" + output
                })
                session['conversation_history'] = conversation_history
                # حفظ المحادثة
                user = SessionLocal().query(User).filter(User.username == username).first()
                conv_id = None
                if user:
                    conv_id = save_conversation(user.id, question, output, classification='db_analysis')
                    conversation_history.append({"role": "assistant", "content": answer})
                    session['conversation_history'] = conversation_history
                   

                return jsonify({
                    "analysis_code": python_code,
                    "analysis_output": output,
                    "chart_data": chart_data,
                    "summary": generate_summary(output,question),
                    "classification": "db_analysis",
                    'conversation_id': conv_id
                })

        
                # =============== DB_ACTION (INSERT/UPDATE/DELETE/PLSQL) ===============
                
        elif classification == 'db_action':
            # 1) إنشاء الاستعلام من ChatGPT
            action_sql = natural_language_to_dml_action(question)
            if not action_sql:
                return jsonify({"error": "فشل في توليد SQL التعديل."}), 500

            try:
                # 2) تنظيف النص من الحروف غير الضرورية:
                cleaned_sql = action_sql.strip()

                # إزالة النص المحصور في ```sql ...``` لو وجد
                if "```sql" in cleaned_sql:
                    cleaned_sql = cleaned_sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in cleaned_sql:
                    cleaned_sql = cleaned_sql.split("```")[1].strip()

                cleaned_sql = cleaned_sql.rstrip(';')
                cleaned_sql = ' '.join(cleaned_sql.split())  # إزالة المسافات الزائدة

                print(f"User asked for DB action:\n{cleaned_sql}\n")

                # 3) نتحقق هل هو PL/SQL block؟ إذا احتوى على BEGIN أو DECLARE في بدايته
                import re
                is_plsql = False
                # مثلاً لو الاستعلام يبدأ بـ BEGIN أو DECLARE (تجاهل الحالة بالأحرف)
                # ملاحظة: قد تحتاج تغطية حالات أكثر مثل "CREATE OR REPLACE PROCEDURE ..." إلخ
                plsql_pattern = r"^\s*(BEGIN|DECLARE|CREATE\s+OR\s+REPLACE\s+PROCEDURE|CREATE\s+OR\s+REPLACE\s+FUNCTION)"
                if re.search(plsql_pattern, cleaned_sql, re.IGNORECASE):
                    is_plsql = True

                # 4) لو لم يكن PL/SQL، نجرب تجزئة الاستعلامات المنفصلة بفواصل منقوطة
                statements = []
                if not is_plsql:
                    # نفترض أنه قد يحتوي على جمل متعددة مفصولة بـ ;
                    # split على ; ثم تخلص من الفراغات
                    possible_stmts = cleaned_sql.split(';')
                    for stmt in possible_stmts:
                        stmt = stmt.strip()
                        if stmt:
                            statements.append(stmt)
                else:
                    # استعلام واحد (PL/SQL block)
                    statements = [cleaned_sql]

                # 5) ننفذ كل جملة على حدة
                total_rows_affected = 0
                success = True
                message = ""
                for stmt in statements:
                    print(f"Executing statement: {stmt}")
                    ok, msg, rows_aff = execute_sql_action(stmt)
                    if not ok:
                        # لو فشل أحد الاستعلامات
                        success = False
                        message = msg
                        logging.error(f"خطأ في Oracle أثناء تنفيذ الأمر: {stmt}\nرسالة الخطأ: {msg}")

                        # طلب من GPT إصلاح الاستعلام
                        help_prompt = f"""
                        The following SQL/PLSQL statement caused an Oracle error:
                        {stmt}

                        Error message:
                        {message}

                        Please provide a corrected Oracle-compatible SQL statement or explanation.
                        """
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "developer", "content": "أنت مساعد خبير في قاعدة بيانات أوراكل."},
                                {"role": "user", "content": help_prompt}
                            ]
                        )
                        error_explanation = response.choices[0].message.content.strip()

                        return jsonify({
                            "error": error_explanation,
                            "action_sql": stmt
                        }), 500
                    else:
                        total_rows_affected += rows_aff

                # 6) إذا نجحت كل الاستعلامات
                message = "The operation has been successfully completed."
                rows_affected = total_rows_affected

                # 7) تجهيز البيانات للملخص
                action_data = {
                    "labels": ["Executed Statements", "Total Rows Affected", "Message"],
                    "values": [f"{len(statements)} statement(s)", rows_affected, message]
                }
                json_output = json.dumps(action_data)

                # 8) توليد الملخص من الذكاء الاصطناعي
                final_text = generate_summary(json_output, question)

                summary_prompt = f"""
        قم بإنشاء تقرير عن تنفيذ الإجراء التالي:
        - السؤال: {question}
        - عدد الأوامر المنفذة: {len(statements)}
        - مجموع الصفوف المتأثرة: {rows_affected}
        - رسالة التنفيذ: {message}

        الملخص يجب أن:
        1. يشرح العملية بلغة انجليزية واضحة
        2. يذكر عدد الصفوف المتأثرة
        3. يوضح تأثير العملية على البيانات
        4. يحتوي على أي تحذيرات مهمة إن وجدت
        5. يستعرض البيانات التي تم تأثيرها
        """
                try:
                    summary_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "developer", "content": "أنت مساعد ذكي يولد تقارير باللغة الانجليزية عن عمليات قاعدة البيانات."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    summary = summary_response.choices[0].message.content.strip()
                    from langdetect import detect
                    if detect(question) == 'ar':
                        summary = f"{summary}"
                except Exception as e:
                    summary = f"ملاحظة: تعذر توليد الملخص التفصيلي. {str(e)}"

                # إزالة أي أسطر في الملخص تحتوي على "SQL Command Executed:" أو ما شابهه
                import re
                summary = re.sub(
                    r"(?i)^\s*SQL Command Executed:.*$",
                    "",
                    summary,
                    flags=re.MULTILINE
                )
                summary = re.sub(
                    r"(?i)^\s*(SELECT|UPDATE|INSERT|DELETE)[^\n]*",
                    "",
                    summary,
                    flags=re.MULTILINE
                )

                # 9) إضافة كتلة الكود لكل الاستعلام منفذ
                # إن أردت جمعها كلها في مكان واحد:
                executed_block = "\n".join([f"```sql\n{stmt}\n```" for stmt in statements])
                final_text = summary + f"\n\n**Executed SQL Statement(s)**:\n{executed_block}"

                # 10) حفظ المحادثة في السجل
                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": final_text})
                session['conversation_history'] = conversation_history

                # 11) حفظ المحادثة في قاعدة البيانات
                user = SessionLocal().query(User).filter(User.username == username).first()
                if user:
                    conv_id = save_conversation(user.id, question, final_text, classification='db_action')
                    response_data = {
                        "action_sql": cleaned_sql,
                        "rows_affected": rows_affected,
                        "message": message,
                        "final_text": final_text,
                        "classification": "db_action",
                        "conversation_id": conv_id
                    }
                    return jsonify(response_data), 200, {'Content-Type': 'application/json; charset=utf-8'}

            except Exception as e:
                logging.error(f"حدث خطأ غير متوقع: {str(e)}")
                return jsonify({"error": f"حدث خطأ غير متوقع: {str(e)}"}), 500





        # =============== DB_SQL (SELECT) ===============

        elif classification == 'db_sql':
            translated_question = translate_ar_to_en(question).lower()
            is_chart = any(word in translated_question for word in ["statistics", "chart", "graph", "plot"])

            # توليد استعلام SQL من السؤال المحسن
            sql_query = natural_language_to_sql(question=improved_question, is_chart=is_chart)
            if not sql_query:
                return jsonify({'error': 'Failed to generate SQL query'}), 500

            try:
                # محاولة تنفيذ الاستعلام مع خاصية إعادة المحاولة
                df = execute_sql_with_iterative_refinement(sql_query, max_retries=5)

                # التحقق من وجود بيانات في DataFrame
                if df is None or df.empty:
                    empty_answer = generate_empty_result_response(question)
                    # يمكنك حفظ المحادثة إذا أردت:
                    conv_id = save_conversation(user.id, question, empty_answer, classification='empty')
                    return jsonify({
                        'assistant_answer': empty_answer,
                        'classification': 'empty',
                        'conversation_id': conv_id
                    })

                # ملء القيم الفارغة في DataFrame
                df = df.fillna(0)
                df_records = df.to_dict(orient='records')

                if is_chart:
                    # إعادة تسمية الأعمدة للرسم البياني
                    df = df.rename(columns={
                        'JOB_TITLE': 'label',
                        'NUM_EMPLOYEES': 'value'
                    })

                    if 'label' in df.columns and 'value' in df.columns:
                        # إعداد البيانات للرسم البياني
                        df_subset = df.head(5).to_dict(orient='records')
                        chart_path = generate_chart(df, "label", "value")

                        analysis_prompt = f"""
        السؤال: "{question}"
        لقد استخرجنا {len(df)} سجلاً.
        هذه عينة من أول {len(df_subset)} سجل:
        {df_subset}

        نتائج الاستعلام (للرسم):
        {df_records}

        "Your task is to analyze the provided database results and provide comprehensive, clear, and accurate responses. If the user's question is in Arabic, respond in Arabic. If the question is in English, respond in English. Always aim to provide in-depth insights, explain your reasoning, and suggest actionable recommendations when applicable. Ensure your responses are context-aware and cater to both technical and non-technical audiences."
                        """
                        try:
                            analysis_response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "developer",
                                        "content": (
                                            "You are a highly intelligent and detail-oriented data analyst. "
                                            "Your task is to analyze the provided database results and provide comprehensive, "
                                            "clear, and accurate responses. If the user's question is in Arabic, respond in "
                                            "Arabic. If the question is in English, respond in English. Always aim to provide "
                                            "in-depth insights, explain your reasoning, and suggest actionable recommendations "
                                            "when applicable. Ensure your responses are context-aware and cater to both technical "
                                            "and non-technical audiences."
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": analysis_prompt,
                                    },
                                ]
                            )

                            assistant_answer = analysis_response.choices[0].message.content.strip()
                        except Exception as e:
                            assistant_answer = f"Error occurred while generating insights: {str(e)}"

                        conversation_history = session.get('conversation_history', [])
                        conversation_history.append({"role": "user", "content": question})
                        conversation_history.append({
                            "role": "assistant",
                            "content": assistant_answer
                        })
                        session['conversation_history'] = conversation_history

                        # حفظ المحادثة
                        user = SessionLocal().query(User).filter(User.username == username).first()
                        if user:
                            save_conversation(user.id, question, assistant_answer, classification='db_sql')

                        return jsonify({
                            'results': df_records,
                            'chart_path': chart_path,
                            'assistant_answer': assistant_answer,
                            'classification': 'db_sql'
                        })
                    else:
                        return jsonify({'error': 'Invalid data for chart'}), 500
                else:
                    analysis_prompt = f"""
        السؤال: "{question}"
        هذه نتائج الاستعلام:
        {df_records}

        "Your task is to analyze the provided database results and provide comprehensive, clear, and accurate responses. If the user's question is in Arabic, respond in Arabic. If the question is in English, respond in English. Always aim to provide in-depth insights, explain your reasoning, and suggest actionable recommendations when applicable. Ensure your responses are context-aware and cater to both technical and non-technical audiences."
                    """
                    try:
                        analysis_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "developer", "content": "You are a helpful data analyst that uses the provided database result."},
                                {"role": "user", "content": analysis_prompt}
                            ]
                        )
                        assistant_answer = analysis_response.choices[0].message.content.strip()
                    except Exception as e:
                        assistant_answer = f"Error occurred while generating insights: {str(e)}"

                    conversation_history = session.get('conversation_history', [])
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_answer
                    })
                    session['conversation_history'] = conversation_history

                    # حفظ المحادثة
                    user = SessionLocal().query(User).filter(User.username == username).first()
                    if user:
                        conv_id = save_conversation(user.id, question, assistant_answer, classification='db_sql')
                        conversation_history.append({"role": "user", "content": question})
                        conversation_history.append({"role": "assistant", "content": assistant_answer})
                        session['conversation_history'] = conversation_history

                    return jsonify({
                        'results': df_records,
                        'assistant_answer': assistant_answer,
                        'classification': 'db_sql',
                        'conversation_id': conv_id
                    })
            except Exception as e:
                return jsonify({'error': f'Error while processing SQL query: {str(e)}'}), 500


              # ===== معالجة أسئلة DDL =====
        if "ddl" in question_lower or "generate the ddl" in question_lower:
            # استخراج اسم الجدول من السؤال
            table_match = re.search(r"from\s+(\w+)", question_lower, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                ddl_code = get_ddl_for_table(table_name)
                if ddl_code:
                    answer = ddl_code  # إرسال الكود مباشرةً بدون نص إضافي
                else:
                    answer = "⚠️ فشل في استخراج DDL."
            else:
                answer = "❗ يرجى تحديد اسم الجدول (مثال: generate the ddl from employees)."    

            # حفظ المحادثة في قاعدة البيانات
            user = SessionLocal().query(User).filter(User.username == username).first()
            if user:
                conv_id = save_conversation(user.id, question, answer, classification='db_sql')
                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": answer})
                session['conversation_history'] = conversation_history

            return jsonify({
                'results': [{'answer': answer}],
                'classification': 'db_sql',
                'conversation_id': conv_id
            })
        
        # =============== DEFAULT ===============
        else:
            # سؤال عام
            conversation_history = session.get('conversation_history', [])
            if re.match(r"انت\s+اسمك\s+(.+)", question, re.IGNORECASE):
                match = re.match(r"انت\s+اسمك\s+(.+)", question, re.IGNORECASE)
                if match:
                    assistant_name = match.group(1).strip()
                    session['memory']['assistant_name'] = assistant_name
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({"role": "assistant", "content": f"تم تعيين اسمي إلى {assistant_name}."})
                    session['conversation_history'] = conversation_history
                    return jsonify({
                        'results': [{'answer': f"تم تعيين اسمي إلى {assistant_name}."}],
                        'classification': 'general'
                    })

            system_content = "You are a helpful assistant."
            if 'assistant_name' in session.get('memory', {}):
                assistant_name = session['memory']['assistant_name']
                system_content = f"You are {assistant_name}, a helpful assistant."

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "developer", "content": system_content},
                        *conversation_history,
                        {"role": "user", "content": question}
                    ]
                )
                answer = response.choices[0].message.content.strip()
            except:
                answer = "حدث خطأ أثناء الحصول على استجابة من ChatGPT."

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            session['conversation_history'] = conversation_history

            return jsonify({
                            'results': [{'answer': answer}],
                            'assistant_answer': answer,  # <-- أضف هذا
                            'classification': 'general',
                            'conversation_id': conv_id
                        })

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    try:
        data = request.get_json()
        username = data.get('username')
        conversation_id = data.get('conversation_id')
        rating_type = data.get('rating_type')
        comments = data.get('comments')

        if not username or not conversation_id or ((rating_type is None or rating_type == "") and not comments):

            return jsonify({"error": "يرجى تقديم اسم المستخدم ومعرف المحادثة والتقييم."}), 400

        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return jsonify({"error": "المستخدم غير موجود."}), 404

        # تحقق من أن المحادثة تنتمي للمستخدم
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user.id
        ).first()

        if not conversation:
            return jsonify({"error": "المحادثة غير موجودة أو لا تنتمي إلى المستخدم."}), 404

        # إنشاء سجل التغذية الراجعة
        new_feedback = Feedback(
            conversation_id=conversation_id,
            rating_type=rating_type,
            comments=comments
        )
        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)
        db.close()

        return jsonify({"message": "تم تسجيل التغذية الراجعة بنجاح."}), 201

    except Exception as e:
        print(f"Error in /feedback endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# ====================================================================
#             ETL وتجهيز بيانات
# ====================================================================

def etl_process():
    session_db = SessionLocal()
    metadata.reflect(bind=engine)
    all_data = []

    for table in metadata.sorted_tables:
        # استخراج البيانات لكل جدول
        data = session_db.query(table).all()
        transformed_data = [dict(row) for row in data]
        
        # إزالة المفتاح _sa_instance_state
        for record in transformed_data:
            record.pop('_sa_instance_state', None)

        # إضافة بيانات الجدول مع اسمه
        all_data.append({
            "table_name": table.name,
            "columns": [column.name for column in table.columns],
            "sample_data": transformed_data[:5]  # أخذ أول 5 صفوف فقط كعينة
        })

    session_db.close()
    train_llm(all_data)

def train_llm(data):
    for table_data in data:
        table_name = table_data["table_name"]
        columns = table_data["columns"]
        sample_data = table_data["sample_data"]

        # إعداد نص التدريب
        prompt = f"""
You are a database assistant. Here is a table schema and sample data for training:

Table Name: {table_name}
Columns: {', '.join(columns)}
Sample Data:
{sample_data}

Use this data to understand the structure of the table and answer user queries related to this table effectively.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "developer", "content": "You are a knowledgeable assistant trained on the database schema."},
                    {"role": "user", "content": prompt}
                ]
            )
            print(f"Training completed for table: {table_name}")
        except OpenAI.OpenAIError as e:
            print(f"Error training model for table {table_name}: {e}")

schedule.every().day.at("00:00").do(etl_process)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(60)



def update_fine_tune():
    # تشغيل سكريبت prepare_finetune.py لتحديث النموذج
    logger.info("Running update_fine_tune...")
    python_executable = sys.executable
    subprocess.run([python_executable, "prepare_finetune.py"])

scheduler = BackgroundScheduler()
scheduler.add_job(update_fine_tune, 'cron', hour=0, minute=0)  # جدولة التحديث كل دقيقه
scheduler.add_job(send_weekly_report_with_chart, 'cron', hour=6, minute=0)
try:
    scheduler.start()
except Exception as e:
    print(f"Error starting scheduler: {e}")

if __name__ == '__main__':
    schedule_thread = threading.Thread(target=run_schedule)
    schedule_thread.daemon = True
    schedule_thread.start()
    app.run(debug=False, host='0.0.0.0', port=5002)