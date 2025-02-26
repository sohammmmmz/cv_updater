# Import required libraries
from crewai import Agent, Task, Crew
from langchain_groq import Groq
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API')

# Initialize Groq LLM
llm = Groq(api_key=GROQ_API_KEY)

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

# Example usage
if __name__ == "__main__":
    # Sample inputs (replace with actual LaTeX CV and job description)
    cv_latex = """
    \\documentclass{article}
    \\begin{document}
    \\section{Skills}
    \\begin{itemize}
        \\item Python
        \\item Java
    \\end{itemize}
    \\section{Experience}
    \\begin{itemize}
        \\item Project A: Developed a tool using Python.
    \\end{itemize}
    \\end{document}
    """
    job_desc = """
    We are looking for a software engineer proficient in Python, Machine Learning, and AWS.
    Experience with TensorFlow and cloud-based projects is a plus.
    """

    # Run the crew with inputs
    result = crew.kickoff(inputs={'cv_latex': cv_latex, 'job_description': job_desc})

    # Note: CrewAI's exact output structure may vary; adjust based on actual API behavior
    # Assuming result provides access to task outputs
    rating = result.tasks[2].output.rating  # Task 3 output (FitAssessment)
    modified_cv = result.tasks[3].output    # Task 4 output (str)

    # Display results
    print(f"Original CV Fit Rating: {rating}/10")
    print("\nModified CV:")
    print(modified_cv)