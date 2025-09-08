from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()


def save_docs(i, j):
    # Re-find the rows in the popup table
    popup_rows = driver.find_elements(By.XPATH, "//tr[@mat-row]")
    for row in popup_rows:
        j += 1
        cell = row.find_element(By.XPATH, ".//td[contains(@class, 'mat-column-sender')]")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", cell)
        driver.execute_script("arguments[0].click();", cell)
        time.sleep(1)
        
        # Input with placeholder (موضوع)
        input_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='موضوع']"))
        )
        input_value = input_element.get_attribute("value")  # what the user typed
        placeholder = input_element.get_attribute("placeholder")  # "موضوع"

        readonly_inputs = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[readonly='true']"))
        )
        readonly_value = readonly_inputs[1].get_attribute("value")  # take the first one


        file_path = os.path.join(OUTPUT_DIR, f"file_{i}_{j}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{placeholder} : {input_value}\ncode: {readonly_value}\n\n\n")

        # Get TinyMCE content
        iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        driver.switch_to.frame(iframe)
        tinymce_body = driver.find_element(By.ID, "tinymce")
        popup_text = tinymce_body.text
        
        # Save to file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(popup_text)
        time.sleep(sleep_time)

        driver.switch_to.default_content()

        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//app-font-icon[@name='close']/parent::button"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", button)
        driver.execute_script("arguments[0].click();", button)
        time.sleep(1)
    return j



# ---------- Setup ----------
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--incognito")  # optional

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 20)

# ---------- Configuration ----------
LOGIN_URL = "https://mail.iau.ac.ir"  # replace with your login URL
USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]
sleep_time = 2
OUTPUT_DIR = r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/hami_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Step 1: Login ----------
driver.get(LOGIN_URL)
time.sleep(6)

# Replace these selectors with the actual ones on your login page
wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys(USERNAME)
driver.find_element(By.ID, "password").send_keys(PASSWORD)
driver.find_element(By.NAME, "_eventId").click()
time.sleep(7)

# # ---------- Step 2: Go to Hami tab ----------
driver.find_element(By.ID, "consultantApplicationButton").click()
time.sleep(5)

# ---------- Step 3: Click the sidebar button ----------
sidebar_button_xpath = "/html/body/app-root/section/app-container/div/app-consultant/app-layout/div/mat-sidenav-container/mat-sidenav[1]/div/app-menu-sidebar/aside/div[2]/app-consultant-sidebar/ul/li[6]/a/div[2]/span"
wait.until(EC.element_to_be_clickable((By.XPATH, sidebar_button_xpath))).click()
time.sleep(5)

# ---------- Step 4: Select all Hami items ----------
# Find all rows containing the buttons
rows = driver.find_elements(By.XPATH, "//td[contains(@class, 'mat-column-action')]/ancestor::tr")

# Step 1: Get all rows (just identifiers, not WebElements)
rows_info = []
rows = driver.find_elements(By.XPATH, "//td[contains(@class, 'mat-column-action')]/ancestor::tr")
for idx, row in enumerate(rows):
    sender = row.find_element(By.XPATH, ".//td[contains(@class, 'mat-column-sender')]").text
    date = row.find_element(By.XPATH, ".//td[contains(@class, 'mat-column-date')]").text
    rows_info.append({"sender": sender, "date": date, "index": idx + 1})

# Step 2: Loop through rows by identifiers, re-finding elements each time
for i, row_info in enumerate(rows_info, start=1):
    if i in [20, 21, 22, 23]:
        # Re-find the row using unique info (or index)
        row_xpath = f"(//td[contains(@class, 'mat-column-action')]/ancestor::tr)[{row_info['index']}]"
        row = driver.find_element(By.XPATH, row_xpath)
        
        # Click the “move_to_inbox” button in this row
        button = row.find_element(By.XPATH, ".//app-font-icon[@name='move_to_inbox']")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", button)
        driver.execute_script("arguments[0].click();", button)
        time.sleep(5)
        
        # Click the “همه” toggle
        button_all = driver.find_element(
            By.XPATH, "//button[@class='mat-button-toggle-button mat-focus-indicator' and .//span[text()='همه']]"
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", button_all)
        driver.execute_script("arguments[0].click();", button_all)
        time.sleep(5)

        #######################################
        j = 0
        j = save_docs(i=i, j=j)
        #######################################
        
        while True:
            next_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.mat-mdc-paginator-navigation-next"))
            )
            is_disabled = next_button.get_attribute("aria-disabled") == "true"
            if not is_disabled:
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(9)
                j = save_docs(i=i, j=j)
            else:
                break

        sidebar_button_xpath = "/html/body/app-root/section/app-container/div/app-consultant/app-layout/div/mat-sidenav-container/mat-sidenav[1]/div/app-menu-sidebar/aside/div[2]/app-consultant-sidebar/ul/li[6]/a/div[2]/span"
        wait.until(EC.element_to_be_clickable((By.XPATH, sidebar_button_xpath))).click()
        time.sleep(sleep_time)
        

