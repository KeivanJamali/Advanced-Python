from dotenv import load_dotenv
load_dotenv()

# from graph import graph
from LLM_combined.chain import chain

async def run_agent(query: str):
    # Agent 1
    print(f"[INFO] Agent 1 started...")
    res = await chain.ainvoke(query)
    done = res.done
    completeness = res.completeness
    tone = res.tone
    start_grade = res.start_grade
    student_feedback = res.student_feedback
    question = res.question
    print(f"[INFO] Agent 1 finished (Detected done: {done}, completeness: {completeness}, tone: {tone}, start_grade: {start_grade}, student_feedback: {student_feedback}).\n")
    return question, done, completeness, tone, start_grade, student_feedback

# query1 = """hami to employee : موضوع: 'درخواست صدور کارت دانشجویی'

# employee to hami : موضوع: 'پاسخ: درخواست صدور کارت دانشجویی'
# سلام کارت دانشجویی صادر گردید از روز دوشنبه 1404/03/19ب بعد در زمانهای اداری به ساختمان طاهری همکف اتاق 5 ( امور دانشجویی- آقای معلایی ) مراجعه نمایند. باتشکر 03531872405

# hami to student : سلام و عرض ادب
# با احترام ضمن تشکر از همکاران گرامی که پاسخگوی این ایمیل بودند، پاسخگوی شما در درخواست های بعدی خواهیم بود.
# با تشکر

# hami to student : سلام و عرض ادب
# با احترام ضمن تشکر از همکاران گرامی که پاسخگوی این ایمیل بودند، پاسخگوی شما در درخواست های بعدی خواهیم بود.
# با تشکر"""
# run_agent(query=query1)
