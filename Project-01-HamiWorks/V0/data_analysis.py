import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jdatetime
from datetime import datetime
import re

from pathlib import Path

CATEGORY_MAP = {
    "جزئیات درخواست": "request_details",
    "پیام اصلی": "main_message",
    "اطلاعات دانشجو": "student_info",
    "موضوع": "subject",
    "کد": "code"
}

class Pre_Analysis:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.index_data: pd.DataFrame = self._read_indexes()
        self.bad_format = []

    def fit_to_get_data(self):
        self._interate_through_each_hami()

    def fit_to_analysis(self):
        self._read_output_file()
        
    def _read_indexes(self):
        index_data = pd.read_csv(self.input_path/"index.csv")
        index_data.columns = ["id", "full_name", "reference_id"]
        return index_data
    
    def _interate_through_each_hami(self):
        for row, (id, name, reference_id) in self.index_data.iterrows():
            # if not reference_id in ["file_11"]:
                # continue
            print(f"[INFO] Start to work on {id}-{name} with reference id of {reference_id} in data.")
            reformatted_data = pd.DataFrame(columns=["file_name", "subject", "refrence_code", "student_name", "national_id", "student_id", "major", "message", "latency", "duration", "is_null"])
            i = 0
            while True:
                i += 1
                file_name = self.input_path/f"file_{reference_id[5:]}_{i}.txt"
                if not file_name.exists():
                    break
                # try:
                reformatted_data = self._extract_data_from(file_name, reformatted_data)
                # except:
                    # self.bad_format.append(file_name.name)
                    # continue
            reformatted_data.to_csv(self.output_path/f"hami_{id}_{name}_{reference_id}.csv", encoding="utf-8-sig")
            self.reformatted_data = reformatted_data

    def _extract_data_from(self, full_path: Path, to: pd.DataFrame):
        data = [full_path.name]
        print(full_path.name)
        with open(full_path, "r") as f:
            ##############################################################################
            text = f.readlines()
            subject, reference_code, text = self._extract_subject_reference(text)
            data.append(subject)
            data.append(reference_code)
            ##############################################################################
            conversation_blocks = self._extract_conversation_blocks(text)
            ##############################################################################
            data.extend(self._extract_student_info(conversation_blocks))
            ##############################################################################
            data.append(self._get_first_and_second_latency(conversation_blocks))
            data.append(self._get_total_conversation_duration(conversation_blocks))
            data.append(self._check_if_followup_messages_empty(conversation_blocks))
            ##############################################################################

        to.loc[len(to)] = data
        return to

    def _read_output_file(self):
        self.summary_data = pd.DataFrame(columns=["id", "name", "refrence_id", "total_messages", "null_mean", "delay_mean", "duration_mean"])
        for row, (id, name, reference_id) in self.index_data.iterrows():
            data = [id, name, reference_id]
            raw_data = pd.read_csv(self.output_path/f"hami_{id}_{name}_{reference_id}.csv", index_col=0)
            data.append(len(raw_data))
            raw_data["duration"] = pd.to_timedelta(raw_data["duration"])
            raw_data["latency"] = pd.to_timedelta(raw_data["latency"])
            data.append(raw_data["is_null"].mean())
            data.append(raw_data["latency"].mean())
            data.append(raw_data["duration"].mean())
            self.summary_data.loc[len(self.summary_data)] = data
        self.summary_data.to_csv(self.output_path/"summary_file.csv", encoding="utf-8-sig")

    def _extract_subject_reference(self, text: str):
        subject = text[0][8:-1]
        reference_code = text[1][6:-1]
        text = "\n".join(text[4:])
        return subject, reference_code, text

    def _extract_conversation_blocks(self, text: str) -> list:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Regex: capture lines like "---- category ----"
        pattern = r"(?m)^-+\s*([^-\n]+?)\s*-+$"
        matches = list(re.finditer(pattern, text))
        
        result = {}
        
        for i, match in enumerate(matches):
            category_fa = match.group(1).strip()
            if not category_fa:
                continue
            
            start_idx = match.end()
            end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
            block = text[start_idx:end_idx].strip()
            
            # Translate category
            category_en = CATEGORY_MAP.get(category_fa, category_fa)
            
            # Find datetime inside block (look for "ارسال شده:")
            timestamp_obj = None
            ts_match = re.search(r"ارسال شده:\s*'([^']+)'", block)
            if ts_match:
                timestamp_str = ts_match.group(1)
                try:
                    date_part = timestamp_str.split("،")[1].strip()
                    timestamp_obj = jdatetime.datetime.strptime(date_part, "%d %B %Y %H:%M:%S")
                except Exception as e:
                    print(f"⚠️ Failed to parse timestamp: {timestamp_str}, error: {e}")
            block = re.sub(r"ارسال شده:\s*'[^']+'\s*", "", block).strip()
            # Build key
            key = (category_en, timestamp_obj)
            result[key] = block
        
        return result
        
    def _extract_student_info(self, blocks_dict):
        name = national_id = student_id = major = message = None
        conversation = {}

        for (category, dt), content in blocks_dict.items():
            # normalize newlines
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            if category == "student_info":
                # field regexes
                m_name = re.search(r"نام و نام خانوادگی:\s*(.+)", content)
                m_nid  = re.search(r"کد ملی:\s*([0-9۰-۹]+)", content)
                m_sid  = re.search(r"شماره دانشجویی:\s*([0-9۰-۹]+)", content)
                m_major= re.search(r"رشته محل:\s*(.+)", content)

                if m_name and not name:
                    name = m_name.group(1).strip()
                if m_nid and not national_id:
                    national_id = int(m_nid.group(1).strip())
                if m_sid and not student_id:
                    try:
                        student_id = int(m_sid.group(1).strip())
                    except ValueError:
                        student_id = None
                if m_major and not major:
                    major = m_major.group(1).strip()

                # پیام lives at the *end* of student_info
                if not message:
                    msg = self._extract_message_from_student_info(content)
                    if msg:
                        message = msg
            else:
                # other categories: treat content directly as a message
                conversation[(category, dt)] = content.strip()

        # Sort conversation by time
        sorted_items = sorted(conversation.items(), key=lambda kv: kv[0][1])
        conversation = dict(sorted_items)
        # Handle student_info message concatenation
        if message:
            # student_msg is non-empty → must append to the earliest message
            if sorted_items:
                first_key, first_text = sorted_items[0]
                # if first_text != "":
                #     raise ValueError(
                #         f"Expected first message text to be empty, got: {first_text!r}"
                #     )
                conversation[first_key] = (first_text + " " + message).strip()
            else:
                # no other messages at all → message becomes the only entry
                conversation[("student_info", None)] = message
        to_nan = lambda x: x if (x is not None and str(x).strip() != "") else np.nan
        return [
            to_nan(name),
            to_nan(str(national_id)),
            to_nan(str(student_id)),
            to_nan(major),
            conversation,
        ]
        
    def _extract_message_from_student_info(self, content: str):
        # split by dashed lines (5+ dashes)
        parts = re.split(r"(?m)(?:^|\n)\s*-{5,}\s*(?:\n|$)", content)
        tail = parts[-1].strip()

        # remove any field-lines if separator was missing
        cleaned_lines = []
        for line in tail.splitlines():
            if re.match(r"\s*(نام و نام خانوادگی:|کد ملی:|شماره دانشجویی:|رشته محل:)", line):
                continue
            if re.match(r"\s*-{5,}\s*$", line):
                continue
            cleaned_lines.append(line)
        msg = "\n".join([l for l in cleaned_lines if l.strip()]).strip()
        return msg if msg else None
    
    def _get_first_and_second_latency(self, blocks_dict):
        times = sorted([dt for (_cat, dt) in blocks_dict.keys() if dt is not None])
        if len(times) < 2:
            return np.nan
        return times[1] - times[0]
    
    def _get_total_conversation_duration(self, blocks_dict):
        times = sorted([dt for (_cat, dt) in blocks_dict.keys() if dt is not None])
        if len(times) < 2:
            return np.nan
        return times[-1] - times[0]

    def _check_if_followup_messages_empty(self, blocks_dict):
        # sort by datetime
        items = sorted([(dt, content) for (_cat, dt), content in blocks_dict.items() if dt is not None],
                    key=lambda x: x[0])
        
        if len(items) <= 1:
            return True  # no follow-ups
        
        followups = [content.strip() for (_dt, content) in items[1:]]
        return all(f == "" for f in followups)

class Post_Analysis:
    def __init__(self, output_path):
        self.output_path = output_path
        self.index_data, self.hami_data, self.summary_data = self._read_data()

    def fit(self):
        self.split_students_dataframes()

    def _read_data(self):
        index_data = pd.read_csv(self.output_path/"index.csv")
        index_data.columns = ["id", "full_name", "reference_id"]
        hami_data = {}
        for _, row in index_data.iterrows():
            hami_data[row['id']] = pd.read_csv(self.output_path/f"hami_{row['id']}_{row['full_name']}_{row['reference_id']}.csv", index_col=0)
        summary_data = pd.read_csv(self.output_path/"summary_file.csv", index_col=0)

        return index_data, hami_data, summary_data

    def split_students_dataframes(self):
        # Combine all operator dataframes into one
        all_data = pd.concat(self.hami_data.values(), ignore_index=True)

        # Group by national_id
        for national_id, df_student in all_data.groupby("national_id"):
            # Save each student dataframe
            file_name = f"student_{str(int(national_id))}.csv"
            df_student.to_csv(self.output_path / file_name, index=False, encoding="utf-8-sig")
