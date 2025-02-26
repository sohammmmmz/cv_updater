# Import required libraries
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq  # Updated import based on common usage
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Ensure the key name matches your .env file

# Initialize Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="mixtral-8x7b-32768")  # Specify a model if needed

# Define structured output models using Pydantic
class CVAnalysis(BaseModel):
    content: dict  # e.g., {'skills': ['Python', 'Java'], 'experience': ['Project A: Used Python...']}
    structure: str  # LaTeX template without content

class JobRequirements(BaseModel):
    skills: list[str]  # e.g., ['Python', 'Machine Learning']
    technologies: list[str]  # e.g., ['TensorFlow', 'Docker']

class FitAssessment(BaseModel):
    rating: int  # 1 to 10
    missing_keywords: list[str]  # e.g., ['Machine Learning', 'AWS']

# Define agents
cv_processor = Agent(
    role='CV Processor',
    goal='Analyze and modify the CV to align with job requirements',
    backstory='Expert in LaTeX parsing and resume optimization',
    llm=llm,
    verbose=True
)

job_analyzer = Agent(
    role='Job Analyzer',
    goal='Extract key requirements from job descriptions',
    backstory='Specialist in interpreting job postings and identifying critical skills',
    llm=llm,
    verbose=True
)

matcher = Agent(
    role='Matcher',
    goal='Evaluate CV fit for jobs and suggest enhancements',
    backstory='Experienced in recruitment and candidate-job matching',
    llm=llm,
    verbose=True
)

# Define tasks with descriptions and dependencies via context
task1 = Task(
    description=(
        'Parse the provided LaTeX CV to extract its content (e.g., skills, experience sections) '
        'and structure (the LaTeX template without content). Return a structured output.'
    ),
    agent=cv_processor,
    expected_output=CVAnalysis
)

task2 = Task(
    description=(
        'Analyze the job description to identify required skills, technologies, and other key '
        'requirements. Return a structured list of these elements.'
    ),
    agent=job_analyzer,
    expected_output=JobRequirements
)

task3 = Task(
    description=(
        'Compare the CV content with the job requirements. Rate the CV fit for the job on a scale '
        'of 1 to 10 based on the overlap of skills and technologies. Identify missing keywords '
        'from the job requirements that could enhance the CV, focusing on the same domain as the '
        'original CV.'
    ),
    agent=matcher,
    context=[task1, task2],  # Uses outputs from Task 1 and Task 2
    expected_output=FitAssessment
)

task4 = Task(
    description=(
        'Modify the CV by updating the skills section with missing keywords that align with the '
        "original CV's domain. Optionally, integrate relevant keywords into the experience section "
        'if they match existing projects. Preserve the original LaTeX structure and return the '
        'updated LaTeX code.'
    ),
    agent=cv_processor,
    context=[task1, task3],  # Uses CV analysis (Task 1) and missing keywords (Task 3)
    expected_output=str  # Modified LaTeX code
)

# Create the crew
crew = Crew(
    agents=[cv_processor, job_analyzer, matcher],
    tasks=[task1, task2, task3, task4],
    verbose=True  # Enable detailed logging for debugging
)

# Streamlit Frontend
st.title("CV Customizer")

st.write(
    "This tool helps you optimize your LaTeX resume for a specific job description by analyzing "
    "and suggesting improvements."
)

# Input Sections
st.header("Input Your CV")
cv_latex = st.text_area(
    "",
    height=300,
    placeholder="\\documentclass{article}\n\\begin{document}\n...\n\\end{document}",
    key="cv_input"
)

st.header("Input the Job Description")
job_desc = st.text_area(
    "",
    height=200,
    placeholder="We are looking for a software engineer with experience in...",
    key="job_input"
)

# Customize CV Button and Processing Logic
if st.button("Customize CV"):
    if cv_latex and job_desc:
        with st.spinner("Analyzing and customizing your CV..."):
            try:
                inputs = {'cv_latex': cv_latex, 'job_description': job_desc}
                crew.kickoff(inputs=inputs)
                rating = task3.output.rating
                modified_cv = task4.output
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        st.success("Customization complete!")

        # Display Results
        st.metric(label="Fit Rating", value=f"{rating}/10")
        st.code(modified_cv, language='latex')
        st.download_button(
            label="Download Modified CV",
            data=modified_cv,
            file_name="modified_cv.tex",
            mime="text/plain"
        )
    else:
        st.error("Please provide both CV and job description.")
