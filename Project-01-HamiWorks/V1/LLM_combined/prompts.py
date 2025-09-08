from langchain_core.prompts import ChatPromptTemplate


prompt_to_grade = ChatPromptTemplate([("system", """
## **Task**

You will be given a **conversation text**. Extract the required parameters and report them in the specified JSON format.

---
## **Context**

- There are **3 roles** in the conversation:
    
    - **student** → asks questions.
        
    - **hami** → responds directly or forwards the question to the correct employee.
        
    - **employee** → provides answers.

    - **someone** → could be **hami** or **employee** or **student**.

- **Focus**: Evaluate only the **hami’s performance**.
"""),
                                        ("user", "{messages}")],
                                        input_variables=["messages"],
                                        validate_template=True,
                                        metadata={"name": "grade messages"})

