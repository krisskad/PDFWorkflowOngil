import ast
import json
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import zipfile
import os
import shutil
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import difflib
import janitor
import random
import string
from typing import List
import Levenshtein
import glob
from seleniumwire import webdriver


def setup_chrome_driver_with_manager():
    custom_options = webdriver.ChromeOptions()
    custom_options.add_argument("start-maximized")
    custom_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    custom_options.add_experimental_option('useAutomationExtension', False)
    custom_options.binary_location = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
    service = Service(ChromeDriverManager().install())
    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=custom_options)
    return driver


def setup_chrome_driver_with_manager2():
    PROXY_HOST = "dc.oxylabs.io"
    PROXY_PORT = "8000"
    PROXY_USER = "user-srengara_d2Wd8-country-IN"
    PROXY_PASS = "Welcome2684+"
    chrome_options = {
        'proxy': {
            'http': f'http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}',
            'https': f'https://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}',
        }
    }
    custom_options = webdriver.ChromeOptions()
    custom_options.add_argument("start-maximized")
    custom_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    custom_options.add_experimental_option('useAutomationExtension', False)
    custom_options.binary_location = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
    service = Service(ChromeDriverManager().install())
    # Initialize the WebDriver
    driver1 = webdriver.Chrome(service=service, options=custom_options, seleniumwire_options=chrome_options)
    return driver1


## Casse URLS extraction form the search terms

def get_case_urls(company_list):
    df = pd.DataFrame()
    extracted_links = []
    extracted_companynames = []
    searched_string = []

    page_source = "https://www.casemine.com/"
    driver = setup_chrome_driver_with_manager()
    driver.maximize_window()
    driver.get(page_source)
    wait = WebDriverWait(driver, 30)
    try:
        ## Login credentials
        element = wait.until(EC.visibility_of_element_located(
            (By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "bt-top-pd", " " ))]')))
        element = wait.until(EC.element_to_be_clickable(
            (By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "bt-top-pd", " " ))]')))
        element.click()
        time.sleep(random.randint(1, 5))
        element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="j_username"]')))
        element.clear()
        time.sleep(random.randint(1, 3))
        element.send_keys("soumikc83@gmail.com")
        element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="j_password"]')))
        element.clear()
        element.send_keys("Stat036521#")
        time.sleep(random.randint(1, 6))
        element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="loginForm"]/div[5]/button')))
        element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="loginForm"]/div[5]/button')))
        element.click()
        time.sleep(random.randint(1, 6))
        ## Click the dropdown to select title
        # element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "select__selected", " " ))]')))
        # element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "select__selected", " " ))]')))
        # element.click()
        # time.sleep(random.randint(1, 3))

        ## Click the title from the dropdown to select for search
        # element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[(@id = "tit_Title")]')))
        # element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[(@id = "tit_Title")]')))
        # element.click()
        # time.sleep(random.randint(1, 3))
        for name in tqdm(company_list):
            ## Passing Construction name or Promoter name to the search box
            element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[(@id = "query")]')))
            element.clear()
            element.send_keys(name)

            ## Click the search box
            element = wait.until(EC.visibility_of_element_located((By.XPATH,
                                                                   '//*[contains(concat( " ", @class, " " ), concat( " ", "top-submit", " " ))] | //*[(@id = "query")] | //*[(@id = "tit_Title")]')))
            element = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                             '//*[contains(concat( " ", @class, " " ), concat( " ", "top-submit", " " ))] | //*[(@id = "query")] | //*[(@id = "tit_Title")]')))
            element.click()
            time.sleep(random.randint(2, 4))

            ## Get all the links and corresponding parites from the search results
            links = driver.find_elements(By.XPATH,
                                         '//*[contains(concat( " ", @class, " " ), concat( " ", "jdlink", " " ))]')
            extractedlinks = [link.get_attribute('href') for link in links]
            extractedcompanynames = [text.text for text in links]

            ## Appending the files
            extracted_links.append(extractedlinks)
            extracted_companynames.append(extractedcompanynames)
            searched_string.append(name)
            time.sleep(random.randint(1, 4))
    except Exception as e:
        print(f"An error occurred: {e}")

    driver.close()

    ## Making the final data frame of case urls and case parties
    df['company_promoter_name'] = searched_string
    df['party_names'] = extracted_companynames
    df['case_links'] = extracted_links
    df_exploded = df.explode(["party_names", "case_links"]).reset_index(drop=True)
    df_exploded['petitioner'] = df_exploded['party_names'].apply(
        lambda x: x.split(' v. ')[0].strip() if ' v. ' in str(x) else x)
    df_exploded['respondent'] = df_exploded['party_names'].apply(
        lambda x: x.split(' v. ')[1].strip() if ' v. ' in str(x) else '')
    return df_exploded


## Case details extracton for the extracted case urls

def case_details_extraction(filtered_df):
    case_numbers = []
    advocates = []
    judges = []
    acts = []
    case_summaries = []
    party_names = []
    company_promoter_name = []
    petitioners = []
    respondents = []
    case_date = []

    for party_name, case_link, company_promoter_names, petitioner_name, respondent_name in tqdm(
            zip(filtered_df['party_names'], filtered_df['case_links'], filtered_df['company_promoter_name'],
                filtered_df['petitioner'], filtered_df['respondent']), total=len(filtered_df)):
        driver1 = setup_chrome_driver_with_manager2()
        driver1.maximize_window()
        driver1.get(case_link)
        wait = WebDriverWait(driver1, 30)
        element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="myTab1"]/a')))
        element.click()
        time.sleep(random.randint(7, 10))
        try:
            case_no = driver1.find_element(By.XPATH,
                                           '//*[contains(concat( " ", @class, " " ), concat( " ", "judgement-header-case", " " ))]').text
        except:
            case_no = None
        try:
            date = driver1.find_element(By.XPATH,
                                        '//*[contains(concat( " ", @class, " " ), concat( " ", "jd-data-new", " " ))]').text
        except:
            date = None
        try:
            advocate = driver1.find_element(By.XPATH,
                                            '//*[contains(concat( " ", @class, " " ), concat( " ", "print_advocates", " " ))]').text
        except:
            advocate = None
        try:
            judge = driver1.find_element(By.XPATH,
                                         '//*[contains(concat( " ", @class, " " ), concat( " ", "print_judges", " " ))]').text
        except:
            judge = None
        try:
            act = driver1.find_element(By.XPATH,
                                       '//*[contains(concat( " ", @class, " " ), concat( " ", "acts_holder_inners", " " ))]').text
        except:
            act = None
        try:
            try:
                case_summary = driver1.find_element(By.XPATH, '//*[(@id = "show_judgment")]').text
            except:
                case_summary = driver1.find_element(By.XPATH,
                                                    '//*[contains(concat( " ", @class, " " ), concat( " ", "pdf-pages", " " ))]').text
        except:
            case_summary = None
        try:
            partyname = party_name
        except:
            partyname = None
        try:
            companyname = company_promoter_names
        except:
            companyname = None
        try:
            petitionername = petitioner_name
        except:
            petitionername = None
        try:
            respondentname = respondent_name
        except:
            respondentname = None

        case_numbers.append(case_no)
        case_date.append(date)
        advocates.append(advocate)
        judges.append(judge)
        acts.append(act)
        case_summaries.append(case_summary)
        party_names.append(partyname)
        company_promoter_name.append(companyname)
        petitioners.append(petitionername)
        respondents.append(respondentname)
        driver1.delete_all_cookies()
        driver1.execute_script("window.localStorage.clear();")
        driver1.execute_script("window.sessionStorage.clear();")
        time.sleep(1)
        driver1.close()

    case_details = pd.DataFrame()
    case_details['search_term'] = company_promoter_name
    case_details['party_names'] = party_names
    case_details['petitioner'] = petitioners
    case_details['respondent'] = respondents
    case_details['case_number'] = case_numbers
    case_details['dated'] = case_date
    case_details['advocates'] = advocates
    case_details['judges'] = judges
    case_details['acts'] = acts
    case_details['case_summary'] = case_summaries
    return case_details


def main():
    search_terms = ['NH-27', 'NH-8E']
    ## Extracting the case urls
    case_urls = get_case_urls(search_terms)
    case_urls_selected = pd.DataFrame()
    ## Selecting only the top 4 urls for each search terms
    for name, group in case_urls.groupby('company_promoter_name'):
        top_4 = group.head(4)
        case_urls_selected = pd.concat([case_urls_selected, top_4])
    # case_urls_selected.to_csv('case_urls1.csv', index=False)
    ## Extracting the case details
    case_details = case_details_extraction(case_urls_selected)

    case_details.to_csv('case_details_130325.csv', index=False)


if __name__ == "__main__":
    main()