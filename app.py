# Import required libraries
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import time


def my_step_callback(step_info):
    time.sleep(15)
    return step_info



# Load environment variables from .env file
llm_configs = {
    "llm": LLM(
        model="gemini/gemini-1.5-pro-latest",
        api_key='AIzaSyC3p1eyiArp6dFYktEdqf8SEp93Tfv1438'
    ),
    "llm1": LLM(
        model="gemini/gemini-1.5-pro-latest",
        api_key='AIzaSyC-yx3xmedNXBPtyHOpqvjAVqW_Mh0SOMU'
    ),
    "llm2": LLM(
        model="gemini/gemini-1.5-pro-latest",
        api_key='AIzaSyC3p1eyiArp6dFYktEdqf8SEp93Tfv1438'
    ),
    "llm3": LLM(
        model="gemini/gemini-1.5-pro-latest",
        api_key='AIzaSyCy4KHOG5IHG1lFL1L5R9HpH8ZMDQ42TiQ'
    ),
    "llm4": LLM(
        model="ollama/deepscaler:latest",
        base_url="http://localhost:11434"
    ),
    # "llm5": LLM(
    #     model="groq/llama-3.3-70b-versatile",
    #     api_key=os.getenv("LLM_API_KEY_6")
    # ),
    # "llm6": LLM(
    #     model="groq/llama-3.3-70b-versatile",
    #     api_key=os.getenv("LLM_API_KEY_7")
    # ),
    # "llm7": LLM(
    #     model="gemini/gemini-1.5-pro-latest",
    #     api_key=os.getenv("LLM_API_KEY_8")
    # ),
    # "manager_llm": LLM(
    #     model="groq/llama-3.3-70b-versatile",
    #     api_key=os.getenv("MANAGER_LLM_API_KEY")
    # )
}

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
    llm=llm_configs["llm4"],
    verbose=True,
    memory=True,
    callback=my_step_callback
)

job_analyzer = Agent(
    role='Job Analyzer',
    goal='Extract key requirements from job descriptions',
    backstory='Specialist in interpreting job postings and identifying critical skills',
    llm=llm_configs["llm4"],
    verbose=True,
    memory=True,
    callback=my_step_callback
)

cv_analyzer = Agent(
    role='CV Analyzer',
    goal='To analyze the CV content and structure',
    backstory='You are a experienced latex code interpretter and your task is to understand and work wit a latex code of a CV ',
    llm=llm_configs["llm4"],
    verbose=True,
    memory=True,
    callback=my_step_callback
)

matcher = Agent(
    role='Matcher',
    goal='Evaluate CV fit for jobs and suggest enhancements',
    backstory='Experienced in recruitment and candidate-job matching',
    llm=llm_configs["llm4"],
    verbose=True,
    memory=True,
    callback=my_step_callback
)

# Define tasks with descriptions and dependencies via context
# Note: expected_output should be a string descriptor, not a class
task1 = Task(
    description=(
        'You given a latex code of the original CV and your task is to extract the content and structure of the CV. '
        'content of the structure should contain the following keys: skills, experience,experience_in_yrs, education, and any other relevant sections. '
        'The structure should be a the same latex code of the original cv  but without the skills section and the sections that requires changing for applying for a different job role.'
        'please make sure to never compromise on the structure of the cv it should be exactly the same as the original cv.No preamble'
        'Here is the latex code ```latex {cv_latex}```'
    ),
    agent=cv_analyzer,
    expected_output="CV analysis with content dictionary and LaTeX structure",
    memory=True,
    output_pydantic=CVAnalysis
)

task2 = Task(
    description=(
        'Analyze the job description to identify required skills, technologies, and other key requirements that a candidate should possess.'
        'requirements. Return a structured list of these elements organized into skills and technologies lists.'
        'Here is the job description ```{job_description}```'
    ),
    agent=job_analyzer,
    expected_output="Job requirements with skills and technologies lists",
    output_pydantic=JobRequirements
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
    expected_output="Fit assessment with rating (1-10) and missing keywords list",
    output_pydantic=FitAssessment
)

task4 = Task(
    description=(
        'Your task is to precisely replicate the original CV LaTeX code with targeted modifications to the skills section only. '
        'STRICT REQUIREMENTS:\n'
        '1. Copy the entire original LaTeX code exactly as-is\n'
        '2. Only modify content within \\begin skills  and \\end skills section\n and the content within \begin rSection Objective \small  and \\end rSection (Update the Objective as well) according to the job description'
        '3. Maintain identical document structure, formatting, and commands\n'
        '4. Preserve all personal information unchanged\n'
        '5. Keep all other sections (education, experience, etc.) exactly as they appear\n'
        '6. Any additions must be relevant to original experience and job requirements\n\n'
        'OUTPUT FORMAT:\n'
        '- Return complete LaTeX document from \\documentclass to \\end document \n'
        '- Highlight modified skills section with a comment % MODIFIED SKILLS SECTION\n'
        'Original CV: {cv_latex}\n'
        
    ),
    agent=cv_processor,
    context=[task1, task3],
    expected_output="Complete LaTeX CV with identical structure, only skills section modified"
)

# Create the crew
crew = Crew(
    agents=[cv_processor, job_analyzer, matcher],
    tasks=[task1, task2, task3, task4],
    verbose=True,  # Enable detailed logging for debugging
    share_crew=True
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