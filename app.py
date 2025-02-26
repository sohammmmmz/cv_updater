# Import required libraries
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key="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"
)
 
llm1 = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key="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"
)
 
llm2 = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key="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"
)

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
    llm=llm1,
    verbose=True
)

matcher = Agent(
    role='Matcher',
    goal='Evaluate CV fit for jobs and suggest enhancements',
    backstory='Experienced in recruitment and candidate-job matching',
    llm=llm2,
    verbose=True
)

# Define tasks with descriptions and dependencies via context
# Note: expected_output should be a string descriptor, not a class
task1 = Task(
    description=(
        'Parse the provided LaTeX CV to extract its content (e.g., skills, experience sections) '
        'and structure (the LaTeX template without content). Return a structured output with content as a dictionary '
        'and structure as a string.'
    ),
    agent=cv_processor,
    expected_output="CV analysis with content dictionary and LaTeX structure"
)

task2 = Task(
    description=(
        'Analyze the job description to identify required skills, technologies, and other key '
        'requirements. Return a structured list of these elements organized into skills and technologies lists.'
    ),
    agent=job_analyzer,
    expected_output="Job requirements with skills and technologies lists"
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
    expected_output="Fit assessment with rating (1-10) and missing keywords list"
)

task4 = Task(
    description=(
        'Update ONLY the Skills section in the LaTeX CV by incorporating missing keywords from the job description. '
        'Make minimal changes to existing skill categories without altering their order or formatting. '
        'Only modify Experience entries if new skills directly match mentioned projects/technologies. '
        'Preserve all LaTeX commands, spacing, comments, and document structure exactly as in the original template. '
        'Never alter headers, objective, education, or other non-skill sections.'
    ),
    agent=cv_processor,
    context=[task1, task3],
    expected_output="LaTeX CV with identical structure to input, only modifying Skills section content"
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
                result = crew.kickoff(inputs=inputs)
                
                # Access task outputs, but be aware they might not be in the expected format
                # You'll need to parse them based on what the agents actually return
                try:
                    # Extract the rating from task3 output
                    rating = "N/A"
                    if hasattr(task3, 'output') and task3.output:
                        # The TaskOutput object in CrewAI has both .raw and .value properties
                        if hasattr(task3.output, 'raw'):
                            raw_output = task3.output.raw
                            # Try to extract rating from raw output
                            if isinstance(raw_output, dict) and 'rating' in raw_output:
                                rating = raw_output['rating']
                            elif hasattr(raw_output, 'rating'):
                                rating = raw_output.rating
                            elif isinstance(raw_output, str):
                                # Try to find a rating pattern in the string
                                import re
                                rating_match = re.search(r'rating[:\s]+(\d+)', raw_output, re.IGNORECASE)
                                if rating_match:
                                    rating = rating_match.group(1)
                        
                    # Extract the modified CV from task4 output
                    modified_cv = "Error retrieving modified CV"
                    if hasattr(task4, 'output') and task4.output:
                        # Get the string representation of the TaskOutput
                        if hasattr(task4.output, 'raw'):
                            modified_cv = str(task4.output.raw)
                        elif hasattr(task4.output, 'value'):
                            modified_cv = str(task4.output.value)
                        else:
                            # Fallback to string representation
                            modified_cv = str(task4.output)
                            
                except Exception as e:
                    st.warning(f"Could not extract some results: {e}")
                    rating = "N/A"
                    modified_cv = "Error processing CV"
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        st.success("Customization complete!")

        # Display Results
        st.metric(label="Fit Rating", value=f"{rating}/10" if rating != "N/A" else rating)
        st.code(modified_cv, language='latex')
        
        # Ensure modified_cv is a string for the download button
        if not isinstance(modified_cv, str):
            modified_cv = str(modified_cv)
            
        st.download_button(
            label="Download Modified CV",
            data=modified_cv,
            file_name="modified_cv.tex",
            mime="text/plain"
        )
    else:
        st.error("Please provide both CV and job description.")