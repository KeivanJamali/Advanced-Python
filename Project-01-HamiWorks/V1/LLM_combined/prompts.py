from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


system_message = SystemMessagePromptTemplate.from_template("""
You are an expert evaluator of university support conversations in Persian.  
The conversation may have missing or unclear messages. Your task is to:

1. Identify the **student’s main question** (in Persian). If missing, infer it from context.
    
2. Provide **numeric ratings** for each category below:
- **done**: How well the student’s question was answered (0–5).
- **completeness**: How clear, transparent, and complete was Hami’s response (1–5).
- **tone**: How professional, respectful, and compassionate was Hami (1–5).
- **start_grade**: Quality of Hami’s first reply to the student (1–5).
- **student_feedback**: How satisfied the student seems with the process and final answer (1–5). If no explicit feedback is given, assign a neutral value (3).

**Important rules:**
- Always write the extracted question in Persian.
- Ratings must strictly follow the allowed ranges.
- If data is missing, infer reasonably from context.    
- Always return a valid JSON object compatible with the `FinalOutput` schema.
   
""")

human_message = HumanMessagePromptTemplate.from_template("{messages}")

prompt_to_grade = ChatPromptTemplate(messages=[system_message, human_message],
                                     input_variables=["messages"],
                                     validate_template=True,
                                     metadata={"name": "grade messages"})

