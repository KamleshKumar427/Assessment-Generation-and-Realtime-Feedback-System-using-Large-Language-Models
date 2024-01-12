import streamlit as st
import requests

@st.cache(suppress_st_warning=True)
def generate_feedback(answers):
    url = 'http://10.3.40.213:8000/feedback/'

    data = {"param1": answers}

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        api_response = response.json()
        feedback = api_response.get('result', 'No feedback available')
    else:
        # Handle errors (you can customize this part as per your need)
        feedback = 'Error: Unable to get feedback from the server.'

    return feedback

@st.cache(suppress_st_warning=True)
def generate_assessment(topic):
    # Dummy function to simulate assessment generation
    # Replace this with your actual function logic
    
    url = 'http://10.3.40.213:8000/interact_with_teacher/'

    data = {"param1": topic}

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        api_response = response.json()
        assessment_response = api_response.get('result', 'No feedback available')
    else:
        # Handle errors (you can customize this part as per your need)
        assessment_response = 'Error: Unable to get feedback from the server.'

    return assessment_response

@st.cache(suppress_st_warning=True)
def generate_Study_material(feedback_final):
    url = 'http://10.3.40.213:8000/StudyMaterial/'

    data = {"param1": feedback_final}

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        api_response = response.json()
        Study_Mater = api_response.get('result', 'No study material available')
    else:
        # Handle errors (you can customize this part as per your need)
        Study_Mater = 'Error: Unable to get study material from the server.'

    return Study_Mater


def compile_answers(answers):
    return "\n".join([f"Answer for question{i+1}: {answer}" for i, answer in enumerate(answers)])

def main():
    st.title("Assignment Generator")

    # Initialize session state variables if they don't exist
    if 'assignment' not in st.session_state:
        st.session_state['assignment'] = ''
    if 'answers' not in st.session_state:
        st.session_state['answers'] = ['' for _ in range(5)]

    # User input for the topic
    topic = st.text_input("Enter a Topic:")

    # Button to generate the assignment
    if st.button("Generate Assessment"):
        st.session_state['assignment'] = generate_assessment(topic)

    # Layout for assignment and answers
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Assignment")
        st.text(st.session_state['assignment'])

    with col2:
        st.subheader("Your Answers")
        for i in range(5):
            if i < 3:
                # Short answer box for MCQs
                st.session_state['answers'][i] = st.text_input(f"Answer {i+1}", key=f"answer_{i+1}", value=st.session_state['answers'][i])
            else:
                # Larger text area for other questions
                st.session_state['answers'][i] = st.text_area(f"Answer {i+1}", key=f"answer_{i+1}", value=st.session_state['answers'][i], height=100)

    # Submit button to compile answers
    if st.button("Submit Answers"):
        compiled_answers = "Assessment is: " + compile_answers(st.session_state['answers'])
        compiled_answers += "\nAnswers for assessment are: " + st.session_state['assignment']

        # Show a message while waiting for feedback
        with st.spinner('Generating feedback, please wait...'):
            feedback = generate_feedback(compiled_answers)
       
        st.subheader("Feedback")
        st.text(feedback)

        # Show a message while waiting for feedback
        with st.spinner('Generating Study Material, please wait...'):
            Study_Material = generate_Study_material(feedback)

        st.subheader("Suggested Study Material: ")
        st.text(Study_Material)

if __name__ == "__main__":
    main()
