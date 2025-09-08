from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class FinalOutput(BaseModel):
    """Find the grading of the hamis."""
    question: str = Field(description="The main question asked by the student.")
    done: str = Field(description="Rate student questions from 1 to 5: 1 = not answered, 2 = barely addressed, 3 = partially answered, 4 = mostly answered, 5 = fully answered. If the question is unclear, respond 'not-clear-question'.")
    completeness: int = Field(description="How clear, transparent, and complete was hami’s response?(from 1(bad) to 5(Excellent))")
    tone: int = Field(description="How professional, respectful, and compassionate was hami?(from 1(bad) to 5(Excellent))")
    start_grade: int = Field(description="Quality of hami’s first reply to the student.(from 1(bad) to 5(Excellent))")
    student_feedback: int = Field(description="How well hami handled student’s feedback (if present).(from 1(bad) to 5(Excellent))")

