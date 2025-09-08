from pathlib import Path
import jdatetime
import pandas as pd
import shutil
import os
import time

class Raw_Data:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_data(self, data_file_name: str):
        # Load data from the specified file
        if not (self.folder_path / data_file_name).exists():
            return []
        with open(self.folder_path / data_file_name, "r", encoding="utf-8") as f:
            return f.readlines()

class Loader:
    def __init__(self, raw_data: Raw_Data,):
        self.raw_data = raw_data

    def fit(self, file_name_number: str):
        self.file_num = file_name_number
        self.i, self.j = file_name_number.split("_")
        self.data_text: list[str] = self.raw_data.load_data("file_" + file_name_number + ".txt")
        self.data_flow: list[str] = self.raw_data.load_data("workflow_" + file_name_number + ".txt")
        if self.data_text == [] and self.data_flow == []:
            # print()
            # print(f"############# file_{file_name_number}.txt or workflow_{file_name_number}.txt does not exist. #############")
            return False
        else:
            return True

    def extract_subject(self) -> str:
        """
        Extracts the subject from a list of strings containing email data.
        Args:
            data (list[str]): A list of strings where each string represents a line from an email.
                              Expected to contain a line starting with "Subject : ".
        Returns:
            str: The extracted subject line text with leading/trailing whitespace removed.
                   Returns None if no subject line is found.
        Example:
            >>> data = ["From: sender@email.com", "Subject : Hello World", "Body: ..."]
            >>> extract_subject(data)
            'Hello World'
        """

        for line in self.data_text:
            if "Subject" in line:
                subject = line.split("Subject : ")[1].strip()
        return subject

    def extract_reference_code(self) -> str:
        """
        Extracts the reference code from a list of strings.
        Searches through each line in the input data for the text "Code" and extracts
        the reference code that follows after "Code: ".
        Args:
            data (list[str]): List of strings to search through for reference code.
        Returns:
            str: The extracted reference code value found after "Code: " in the input data.
        Raises:
            UnboundLocalError: If no line containing "Code: " is found in the input data.
            IndexError: If "Code: " is found but no value follows it.
        """

        for line in self.data_text:
            if "Code" in line:
                reference_code = line.split("Code: ")[1].strip()
        return reference_code

    def extract_major(self) -> str:
        """
        Extracts the major information from a list of strings.
        Args:
            data (list[str]): A list of strings containing student information.
        Returns:
            str: The extracted major value from the data.
        Example:
            >>> data = ["Name: John Doe", "Major: Computer Science", "Year: 2023"]
            >>> extract_major(data)
            'Computer Science'
        """
        
        for line in self.data_text:
            if "Major" in line:
                major = line.split("Major: ")[1].strip()
        return major

    def extract_student_info(self) -> dict:
        """
        Extracts student information from a list of strings containing student data.
        This function parses student information from a text document, looking for specific
        Persian language markers to identify and extract student details including name,
        national ID, student ID, and field of study.
        Args:
            data (list[str]): A list of strings containing the student information document,
                              where each string represents a line of text.
        Returns:
            dict: A dictionary containing the extracted student information with the following keys:
                - name (str): Student's full name
                - national_id (str): Student's national ID number
                - field (str): Student's field of study
                - student_id (str): Student's university ID number
                All values will be None if the corresponding information is not found in the input data.
        Example:
            data = ["اطلاعات دانشجو", "نام و نام خانوادگی: علی محمدی", "کد ملی: 1234567890"]
            student_info = extract_student_info(data)
            # Returns: {'name': 'علی محمدی', 'national_id': '1234567890', 'field': None, 'student_id': None}
        """

        student_section = False
        student_info = {"name": "<empty>", "national_id": "<empty>", "field": "<empty>", "student_id": "<empty>"}
        for line in self.data_text:
            if "اطلاعات دانشجو" in line:
                student_section = True
                continue
            if student_section:
                if "نام و نام خانوادگی:" in line:
                    student_info["name"] = line.split("نام و نام خانوادگی:")[1].strip()
                elif "کد ملی:" in line:
                    student_info["national_id"] = line.split("کد ملی:")[1].strip()
                elif "شماره دانشجویی:" in line:
                    student_info["student_id"] = line.split("شماره دانشجویی:")[1].strip()
                elif "رشته محل:" in line:
                    student_info["field"] = line.split("رشته محل:")[1].strip()
                    break
        return student_info
    
    def extract_messages(self) -> dict:
        """
        Extracts messages from raw data and organizes them by date.
        This function processes a list of strings containing message data and creates a dictionary
        where each key is a datetime object (in Jalali calendar) and the value is the corresponding message text.
        Args:
            data (list[str]): Raw data containing messages, where each element is a line of text.
                              Expected format includes lines with "ارسال شده:" for dates and
                              message content in subsequent lines until a separator line.
        Returns:
            dict: A dictionary where:
                - keys are jdatetime.datetime objects representing message timestamps
                - values are strings containing the full message text
        Example format of returned dictionary:
            {
                datetime(1402,1,1,12:30:00): "First message content...",
                datetime(1402,1,1,13:45:00): "Second message content...",
                ...
            }
        Note:
            - Date strings are expected in Persian format and converted to Jalali datetime
            - Messages are concatenated and stripped of extra whitespace
            - Each message block should be separated by a line containing 8 or more hyphens
        """
        messages = {}
        current_date = None
        current_message = []
        
        for line in self.data_text:
            if "ارسال شده:" in line:
                if current_date and current_message:
                    ms = "".join(current_message).strip()
                    messages[current_date] = ms if ms else "<empty>"
                date_str = line.split("ارسال شده: ")[1].strip().strip("'")
                date_part = date_str.split("،")[1].strip()
                current_date = jdatetime.datetime.strptime(date_part, "%d %B %Y %H:%M:%S")
                current_message = []
            elif current_date is not None and "-" * 8 in line:
                if current_date and current_message:
                    ms = "".join(current_message).strip()
                    messages[current_date] = ms if ms else "<empty>"
                current_date = None
                current_message = []
            elif current_date is not None:
                current_message.append(line)
                
        if current_date and current_message:
            ms = "".join(current_message).strip()
            messages[current_date] = ms if ms else "<empty>"

        return messages

    def extract_external_message(self) -> str:
        """
        Extracts any message that appears after the رشته محل: line, searching from the end.
        Args:
            data (list[str]): List of strings to search through.
        Returns:
            str: The extracted message if found, None otherwise.
        """
        message = []
        
        for line in reversed(self.data_text):
            if "رشته محل:" in line:
                break
            if line.strip():
                message.insert(0, line)
        msg = "".join(message)
        msg = msg.replace("-", "").strip()
        return msg if message else None
    
    def extract_workflow(self) -> dict:
        """
        Extracts workflow information from data and organizes it by date.
        For multiple entries on same date, adds incremental hours to differentiate.
        Returns:
            dict: Keys are jdatetime dates, values are dicts containing from/to/to_email
        """
        workflow = {}
        prev_name = "STUDENT"
        
        current_date = None
        current_name = None 
        current_email = None
        date_counters = {}
        
        for line in self.data_flow:
            if "date: " in line:
                date_str = line.split("date: ")[1].strip("'")
                date_part = date_str.split("،")[1].strip()
                base_date = jdatetime.datetime.strptime(date_part, "%d %B %Y %H:%M")
                
                # Add counter for repeated dates
                if base_date in date_counters:
                    date_counters[base_date] += 1
                else:
                    date_counters[base_date] = 0
                    
                # Add hours to differentiate same-day entries
                current_date = base_date.replace(second=date_counters[base_date])
                
            elif "name:" in line:
                current_name = line.split("name: ")[1].strip()
            elif "email:" in line:
                current_email = line.split("email: ")[1].strip()
            elif "parent_id:" in line:
                parent_id = line.split("parent_id: ")[1].strip()
                if parent_id == "None":
                    parent_id = "<empty>"
            elif "id:" in line:
                current_id = line.split("id: ")[1].strip()
            elif "-" * 8 in line and current_date:
                for date, entry in workflow.items():
                    if entry["id"] == parent_id:
                        prev_name = entry["to"]
                        break
                workflow[current_date] = {
                    "from": prev_name,
                    "to": current_name,
                    "to_email": current_email,
                    "id": current_id,
                    "parent_id": parent_id
                }
                current_date = None
                
        return workflow


class Express_Data:
    def __init__(self):
        pass

    def fit(self, loader: Loader):
        self.loader = loader
        self.subject = self.loader.extract_subject()
        self.reference_code = self.loader.extract_reference_code()
        self.major = self.loader.extract_major()
        self.student_info = self.loader.extract_student_info()
        self.messages = self.loader.extract_messages()
        self.external_message = self.loader.extract_external_message()
        self.workflow = self.loader.extract_workflow()
        self.compressed_data = self._combine_text_flow()

    def _combine_text_flow(self):
        combined_data = []
        
        for date_w in sorted(self.workflow.keys()):
            found = False
            for date_m in sorted(self.messages.keys()):
                if (date_m.year == date_w.year and 
                    date_m.month == date_w.month and 
                    date_m.day == date_w.day and
                    date_m.minute == date_w.minute):
                    if date_m.hour == date_w.hour or date_m.hour + 12 == date_w.hour or date_m.hour - 12 == date_w.hour:
                        row = {
                            'date': date_w,
                            'message': self.messages[date_m],
                            'from': self.workflow[date_w]['from'],
                            'to': self.workflow[date_w]['to'],
                            'to_email': self.workflow[date_w]['to_email'],
                            "from_id": self.workflow[date_w]['parent_id'],
                            "to_id": self.workflow[date_w]['id'],
                            "matched": True
                        }
                        combined_data.append(row)
                        found = True
            if not found:
                row = {
                    'date': date_w,
                    'message': "There is nothing about this message in the Emails.",
                    'from': self.workflow[date_w]['from'],
                    'to': self.workflow[date_w]['to'],
                    'to_email': self.workflow[date_w]['to_email'],
                    "from_id": self.workflow[date_w]['parent_id'],
                    "to_id": self.workflow[date_w]['id'],
                    "matched": False
                }
                combined_data.append(row)

        for date_m in sorted(self.messages.keys()):
            found = False
            for date_w in sorted(self.workflow.keys()):
                if (date_m.year == date_w.year and 
                    date_m.month == date_w.month and 
                    date_m.day == date_w.day and
                    date_m.minute == date_w.minute):
                    if date_m.hour == date_w.hour or date_m.hour + 12 == date_w.hour or date_m.hour - 12 == date_w.hour:
                        found = True
            if not found:
                row = {
                    'date': date_m,
                    'message': self.messages[date_m],
                    'from': "Not in workflow",
                    'to': "Not in workflow",
                    'to_email': "Not in workflow",
                    "from_id": "Not in workflow",
                    "to_id": "Not in workflow",
                    "matched": False
                }
                combined_data.append(row)

        if self.external_message:
            row = {
                'date': jdatetime.datetime(1500, 1, 1, 0, 0, 0),
                'message': self.external_message,
                'from': "Not in workflow",
                'to': "Not in workflow",
                'to_email': "Not in workflow",
                "from_id": "Not in workflow",
                "to_id": "Not in workflow",
                "matched": False
                }
        combined_data.append(row)

        return pd.DataFrame(combined_data).sort_values(by='date').reset_index(drop=True)

    def save_combined_data(self, file_name: Path):
        self.compressed_data.to_csv(file_name, index=False, encoding="utf-8-sig")

    def save_hami_data(self, file_name: Path):
        row = {
            'number': self.loader.j,
            'subject': self.subject,
            'reference_code': self.reference_code,
            'major': self.major,
            'name': self.student_info['name'],
            'national_id': str(self.student_info['national_id']),
            'student_id': str(self.student_info['student_id']),
            'field': self.student_info['field']
        }
        if file_name.exists():
            df = pd.read_csv(file_name)
            df.loc[len(df)] = row
        else:
            df = pd.DataFrame([row])
        df.to_csv(file_name, index=False, encoding="utf-8-sig")

############################################################################################################################

hami_data_path = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/hami_data")
hami_output_path = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/hami_output")
raw_data = Raw_Data(folder_path=hami_data_path)
loader = Loader(raw_data=raw_data)
express = Express_Data()

# Clear combined_output directory
combined_dir = hami_output_path / "combined_output"
if combined_dir.exists():
    for file in combined_dir.iterdir():
        if file.is_file():
            file.unlink()

# Clear hami_output directory
hami_dir = hami_output_path / "hami_output"
if hami_dir.exists():
    for file in hami_dir.iterdir():
        if file.is_file():
            file.unlink()

time.sleep(4)

i = 1
while i < 25:
    j = 1
    while j < 1000:
        # print(f"Processing file_{i}_{j}.txt and workflow_{i}_{j}.txt")
        exist = loader.fit(file_name_number=f"{i}_{j}")
        if not exist:
            j += 1
            break
        express.fit(loader=loader)
        express.save_combined_data(file_name=hami_output_path / "combined_output" / f"combined_{i}_{j}.csv")
        express.save_hami_data(file_name=hami_output_path / "hami_output" / f"hami_{i}.csv")
        # print(f"Finished processing file_{i}_{j}.txt and workflow_{i}_{j}.txt")
        j += 1
    i += 1