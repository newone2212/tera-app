from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.engine import URL, create_engine
from langchain_huggingface import HuggingFaceEndpoint
import spacy

credentials = {
    'username': 'amits',
    'password': 'amit12345',
    'host': '172.16.12.65',
    'database': 'Timeverse'
}

# Adjust the ODBC driver name or version as needed
query_string = dict(driver='ODBC Driver 17 for SQL Server')

# Create the SQLAlchemy connection URL using SQL Server authentication
connect_url = URL.create(
    'mssql+pyodbc',
    username=credentials['username'],
    password=credentials['password'],
    host=credentials['host'], 
    database=credentials['database'],
    query=query_string
)

engine = create_engine(connect_url)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", max_length=25000, temperature=0.1
)

db = SQLDatabase.from_uri(connect_url, include_tables=['AllWorkItems'])

# Load Spacy model
nlp = spacy.load("en_core_web_md")

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    When constructing SQL queries, always use proper alias names to avoid ambiguity errors. Ensure that each column reference is prefixed with its respective table alias.

    Write only the SQL query and nothing else. 
    
    Do not wrap the SQL query in any other text, not even backticks.

    **Use 'TOP' keyword in SQL queries instead of 'LIMIT' in all the SQL queries you generate for user questions (queries). Use TOP clause in beginning of the SQL query. Do not consider TOP in all the SQL queries, only in the SQL queries wherever required.**

    In all the user question of 'List' and 'How' use 'DISTINCT' keyword in the SQL queries (before TOP clause if it is used) so that the outputs of the sql queries are unique and are not duplicated

    Use 'TaskStartedAt' column instead of 'CreatedDate' for generting the SQL queries related to date and time. 

    For example:

    Question: List any 10 projects.
    SQL Query: SELECT TOP 10 project_name FROM AllWorkItems;

    Question: List any 5 tasks assigned to Divy.
    SQL Query: SELECT TOP 5 Title FROM AllWorkItems WHERE UserEmail='divy@dhyey.com';

    Question: List of tasks assigned to divy@dhyey.com.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'divy@dhyey.com';

    Question: List of projects assigned to divy@dhyey.com.
    SQL Query: SELECT DISTINCT project_name FROM AllWorkItems WHERE UserEmail = 'divy@dhyey.com';

    Question: How many tasks are assigned to vraj patel?
    SQL Query: SELECT DISTINCT Title FROM AllWorkItems WHERE UserEmail = 'vraj.patel@dhyey.com'
    
    Question: How many tasks are having delayed status?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE RemainingWork > 0 AND TaskStopedAt IS NOT NULL;

    Question: How many tasks are having ontime status?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE RemainingWork = 0 AND TaskStopedAt IS NOT NULL;

    Question: How many projects are assigned to Riddhi Chhaiya?
    SQL Query: SELECT COUNT(DISTINCT project_name) FROM AllWorkItems WHERE UserEmail = 'riddhi.chhaiya@dhyey.com';

    Question: How many projects are assigned to Divy? List them.
    SQL Query: SELECT DISTINCT project_name FROM AllWorkItems WHERE UserEmail='divy@dhyey.com';

    Question: How many tasks assigned to Nishu Singh have high priority?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE UserEmail = 'nishu.singh@dhyey.com' AND TimeEntryType = 1;

    Question: How many tasks are having additional work hours and are in high priority?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE CompletedWork > OriginalEstimate AND TimeEntryType = 1;

    Question: What is the total completed work from the tasks table?
    SQL Query: SELECT SUM(CompletedWork) FROM AllWorkItems;

    Question: What is the total original estimate from the tasks table?
    SQL Query: SELECT SUM(OriginalEstimate) FROM AllWorkItems;

    Question: What is the total completed work and original estimates from the prediction table?
    SQL Query: SELECT SUM(CompletedWork) AS TotalCompletedWork, SUM(OriginalEstimate) AS TotalOriginalEstimate FROM AllWorkItems;

    Question: How many tasks are in completed state for ankita.dholakia@dhyey.com and list them?
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'ankita.dholakia@dhyey.com' AND RemainingWork = 0;

    Question: Show the project name and task title date for tasks created by nishu.singh@dhyey.com.
    SQL Query: SELECT project_name, Title, TaskStartedAt FROM AllWorkItems WHERE UserEmail = 'nishu.singh@dhyey.com';

    Question: Show the task title, project name, and user email for tasks assigned to nishu.singh@dhyey.com.
    SQL Query: SELECT Title, project_name, UserEmail FROM AllWorkItems WHERE UserEmail = 'nishu.singh@dhyey.com';

    Question: Show the project name, task ID, and reason for tasks that have a specific state.
    SQL Query: SELECT project_name, WorkItemID, Comments FROM AllWorkItems WHERE WorkItemType = 'SpecificState';

    Question: How many tasks did Divy create yesterday?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE UserEmail='divy@dhyey.com' AND CONVERT(DATE, TaskStartedAt) = CONVERT(DATE, GETDATE() - 1);

    Question: How many tasks did Divy create on 12th July 2024?
    SQL Query: SELECT COUNT(*) FROM AllWorkItems WHERE UserEmail='divy@dhyey.com' AND CONVERT(DATE, TaskStartedAt) = '2024-07-12';

    Question: List of tasks Divy created yesterday.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail='divy@dhyey.com' AND CONVERT(DATE, TaskStartedAt) = CONVERT(DATE, GETDATE() - 1);

    Question: List of tasks Divy created on 12th July 2024.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail='divy@dhyey.com' AND CONVERT(DATE, TaskStartedAt) = '2024-07-12';

    Question: How many projects are assigned to Divy? List them.
    SQL Query: SELECT DISTINCT project_name FROM AllWorkItems WHERE UserEmail='divy@dhyey.com';

    Question: How many tasks did nishu.singh@dhyey.com complete last week? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'nishu.singh@dhyey.com' AND RemainingWork = 0 AND TaskStopedAt BETWEEN DATEADD(WEEK, -1, GETDATE()) AND GETDATE();

    Question: How many tasks are in progress for ankita.dholakia@dhyey.com? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'ankita.dholakia@dhyey.com' AND RemainingWork > 0;

    Question: How many tasks have high priority for divy@dhyey.com? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'divy@dhyey.com' AND TimeEntryType = 1;

    Question: How many tasks did nishu.singh@dhyey.com create last month? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'nishu.singh@dhyey.com' AND TaskStartedAt BETWEEN DATEADD(MONTH, -1, GETDATE()) AND GETDATE();

    Question: How many tasks are in delayed status for ankita.dholakia@dhyey.com? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'ankita.dholakia@dhyey.com' AND RemainingWork > 0 AND TaskStopedAt IS NOT NULL;

    Question: How many tasks did vraj.patel@dhyey.com create in May 2024? List them.
    SQL Query: SELECT Title FROM AllWorkItems WHERE UserEmail = 'vraj.patel@dhyey.com' AND TaskStartedAt BETWEEN '2024-05-01' AND '2024-05-31';

    Question: List of users working on project MBSPL.
    SQL Query: SELECT DISTINCT UserEmail FROM AllWorkItems WHERE project_name = 'MBSPL';

    Question: How many users are working on project MBSPL? List them.
    SQL Query: SELECT DISTINCT UserEmail FROM AllWorkItems WHERE project_name = 'MBSPL';

    Question: How many users are assigned to project MBSPL?
    SQL Query: SELECT COUNT(DISTINCT UserEmail) FROM AllWorkItems WHERE project_name = 'MBSPL';

    Question: How many users are assigned on project 'MBSPL'? List them.
    SQL Query: SELECT DISTINCT UserEmail FROM AllWorkItems WHERE project_name = 'MBSPL';

    Question: How many users are working on project 'MBSPL'? List them.
    SQL Query: SELECT DISTINCT UserEmail FROM AllWorkItems WHERE project_name = 'MBSPL';

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Your turn:

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def convert_name_to_email(name):
    # Split the name by space
    parts = name.split()
    # Check if the name contains two parts
    if len(parts) == 2:
        first_name, last_name = parts
        # Construct the email address
        email = f"{first_name.lower()}.{last_name.lower()}@dhyey.com"
        return email
    return None

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    # Handle greetings separately
    greetings = ["hi", "hello", "hola", "good morning", "good afternoon", "good evening", "good night"]
    if user_query.lower() in greetings:
        return "Dhyey Bot: Hello! How can I assist you today?"
        
    # Handle conversation separately
    conversations = ["ok", "thank you", "see you", "nice", "great"]
    if user_query.lower() in conversations:
        return "Dhyey Bot: Can I help you with anything else?"
    
    # Handle goodbye separately  
    goodbyes = ["goodbye", "bye", "ok bye"]
    if user_query.lower() in goodbyes:
        return "Dhyey Bot: Good Bye!"
    
    # Process user query with Spacy to handle similar questions
    doc = nlp(user_query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Check for user names in the entities
    for text, label in entities:
        if label == "PERSON":
            email = convert_name_to_email(text)
            if email:
                user_query = user_query.replace(text, email)
    
    # Process the modified user query
    sql_chain = get_sql_chain(db)

    template = """
       You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, SQL query, and SQL response, write "Dhyey Bot:" while providing a natural language response in output.

        Use the SQL Response to give the AI response. Convert the SQL response into natural language before presenting it. Do not print the SQL response in the output; only provide the natural language response in the AI output.
        
        If the SQL Response is a count (e.g., COUNT(*)), the natural language output should clearly and accurately state the count. Do not give false count in the natural language output.
        
        If the SQL response contains a single value (e.g., COUNT(*)), extract and use this value directly in the natural language response.

        If there is no data available for any SQL query, then print "Data not found" in the AI response.

        Do not print outputs in paragraph format. Do not print Conversation history in output; only print the final output that is converted to natural language from the SQL response.

        Strictly do not print {chat_history}, {query}, and {response} in the Dhyey Bot response.

        Do not print extra information; only give the required information to the user.

        Provide all natural language outputs in numbered or bullet list format.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        SQL Response: {response}
        User question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a Timeverse Chatbot. Ask me anything about the database."),
    ]

load_dotenv()

st.set_page_config(page_title="Timeverse Chatbot", page_icon=":speech_balloon:")
st.title("Timeverse Chatbot")

db = SQLDatabase.from_uri(connect_url)
st.session_state.db = db

if "chat_history" in st.session_state:  # Added check here
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

user_query = st.chat_input("Type a message")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
