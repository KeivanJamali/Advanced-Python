from pydantic import BaseModel, Field

class FinalOutput(BaseModel):
    """Evaluation of Hami's handling of the student request conversation."""

    question: str = Field(
        description="The main question asked by the student, inferred if missing. Write in Persian."
    )
    done: int = Field(
        description="How well the student’s question was answered. \
0 = totally unclear or missing question, \
1 = barely answered, \
2 = partial answer, \
3 = moderate answer, \
4 = good answer, \
5 = excellent answer."
    )
    completeness: int = Field(
        description="How clear, transparent, and complete was Hami’s response? (1 = bad, 5 = excellent)"
    )
    tone: int = Field(
        description="How professional, respectful, and compassionate was Hami? (1 = bad, 5 = excellent)"
    )
    start_grade: int = Field(
        description="Quality of Hami’s first reply to the student. (1 = bad, 5 = excellent)"
    )
    student_feedback: int = Field(
        description="How satisfied the student seems with the process and final answer. \
(1 = very dissatisfied, 5 = very satisfied). If no feedback is given, rate as neutral (3)."
    )
