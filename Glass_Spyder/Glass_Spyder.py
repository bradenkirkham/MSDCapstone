import random

import requests
# import selenium
import os
import time
import json

# from selenium import webdriver
# from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, Tag
import lxml

# I thought selenium would be needed to navigate webpages but found a way to do it without selenium.
# these are not important for glassdoor i\I think, but might be important when I come back around to doing indeed
cookie = """CTK=1g52e4uimh1iu800; indeed_rcc=CTK; _ga=GA1.2.179321185.1662065436; SURF=2z0RUSUNsHYg7WMQQJmdyHD3yiLBK54F; __ssid=249b1ecce6c98c81611c837cd891d64; RF="mzwn06pHXTmqzpCh2lfQELqhs8gY3yhJuRL7KXLOWRPB8DyeSNWouVc0n1mpGYOm5-s_W3xG0s1CA8DQSASqhA=="; SHOE="aELj6_l6BHSgsA_mDHkLaGwXx3mZtSVcOzghUOEmhjCbrSvBQSbZl_eLi2EsWFqBXeeriT7BJFRORmk7a0KocGatg6ZkmtzhfH2Rypkv6PEEMI5ixMgTdFlgfL6uKclz7qKr-q1_UnYKcYSiGpNkEk6V"; SOCK="y_hVVebQRYwLMgUIETrJArVQ5fo="; CSRF=24vV1CQ1Q6YPAnq2Xzn5tH4LlGE23Ikp; SHARED_INDEED_CSRF_TOKEN=nfsV8yKYmCxYi0AXY2FkssW8XreFSXVX; LC="co=US"; _gcl_au=1.1.647639205.1662066612; LOCALE=en; MICRO_CONTENT_CSRF_TOKEN=zIZiq5cyjMZ0MYE8o8bep8lydTg8sM2g; PPID=eyJraWQiOiIwMjhiOTNmMi1lYzAzLTRmZTAtYmM4NS04OWIxMGU3YWI2MjIiLCJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJzdWIiOiIzMjc2NjMzMTE0MTA0MTg0IiwiYXVkIjoiYzFhYjhmMDRmIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF1dGgiOiJnb29nbGUiLCJjcmVhdGVkIjoxNDczNzA0Mjc4MDAwLCJyZW1fbWUiOnRydWUsImlzcyI6Imh0dHBzOlwvXC9zZWN1cmUuaW5kZWVkLmNvbSIsImV4cCI6MTY2NDA0MTczOCwiaWF0IjoxNjY0MDM5OTM4LCJsb2dfdHMiOjE2NjIwNjY2MDA5NTgsImVtYWlsIjoiYnJhZGVuLmtpcmtoYW1AZ21haWwuY29tIn0.RVMxChq3ncC7Vyrz9Qn2zVQe-ZozmWG2Tdxf6DsPioNr8ub_52RKh84yVzdY-kIZnEgGFBNtFMwBpQxnFIFa8Q; __cf_bm=i91nJQFuAFzqgMoPNOxcFX5eCBV3Qa8qYiz22Mrclxs-1664039939-0-AfxOk+fH6DI7I1MCx++eHMfmbMJM4GhG+1pVgk+do7j8+Iczvt63STcwNAszjLrfI3Rp+kmgFW57YbOvaTA/9wY=; _cfuvid=SWsPfSltygIIVdWOA9R089CRNjybvxgKeClbFyWXlPE-1664039939975-0-604800000; _gid=GA1.2.2053266301.1664039941; _gat=1"""
headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
}

# This is my first attempt using selenium to try and navigate the webpage,
# after analyzing the URL i realized i didn't need to do it this way and could just use the requests' library.
# driver = webdriver.Safari()
#
# driver.get("https://www.glassdoor.com/Job/herriman-software-developer-jobs-SRCH_IL.0,8_IC1128279_KO9,28.htm?clickSource=searchBox")
#
#
# next_button = driver.find_element(By.CSS_SELECTOR, ".css-1mfvd1a .nextButton")
# next_button.click()
#
# time.sleep(2)
#
# close_button = driver.find_element(By.CSS_SELECTOR, ".css-adktyj .modal_closeIcon svg path")
# close_button.click()
# print(close_button)
#
# time.sleep(10)
# driver.close()

# def pagination(self):

# tracking which page in glass door i'm on
page_num = 1

# this is the full URL for the job search
URL = "https://www.glassdoor.com/Job/software-engineer-jobs-SRCH_KO0,17.htm?context=Jobs&clickSource=searchBox"

# Breaking the URL into two fragments and adding to it the bits needed to navigate pages of the search
base_url = URL.split('/Job')[0]
Frag_1 = URL.split('.htm')[0] + '_IP'
Frag_2 = '.htm' + URL.split('.htm')[1]

# the next 4 lines are to find how many pages the search has.
response = requests.get(URL, headers=headers)

soup = BeautifulSoup(response.text, features="lxml")

pageination_footer = soup.find('div', {"class": "paginationFooter"})

num_pages = int(pageination_footer.text[-2:])
# num_pages = 1

# list to record all of the links we find for the search.
job_links = []

# repeat the loop for each page
for i in range(num_pages):
    # create the URL for the current page
    URL = Frag_1 + str(page_num) + Frag_2

    # get the servers response to the HTTP request
    response = requests.get(URL, headers=headers)

    # increment page_num for next loop
    page_num = page_num + 1

    # get the Beutiful soup object of the response text
    soup = BeautifulSoup(response.text, features="lxml")

    # create a list of all the links
    job_list = soup.find_all('a', href=True)

    # for each link in the list if it says /partner, and isn't the index link add it to the job_links list
    # this gives us the unique list of relative links that we can use later with our base URL to create
    # a link that takes us to the page for a specific job posting.
    for a in job_list:
        href = a['href']
        if '/partner' in href:
            if 'index' not in href:
                if href not in job_links:
                    job_links.append(href)

    # sleep for a second
    time.sleep(random.randint(0, 3))

# print("got through crawler")

# declare an empty list to save dictionaries to
All_Jobs_list = []

# for every job in the list of links.
for link in job_links:
    # create the URL using the base URL and the relative link,
    # then use request and beutifulsoup to get the response text.
    # print("beginning")
    Job_URL = base_url + link
    response = requests.get(Job_URL, headers=headers)
    soup = BeautifulSoup(response.text, features="lxml")

    time.sleep(random.randint(0, 3))

    # how to extract company name negative indexing to reduce the star rating,
    # couldn't figure out how to just remove the spans...
    #                               css-16nw49e e11nt52q1
    company_name = soup.find('div', {'class': 'css-16nw49e e11nt52q1'})
    while not company_name:  # make sure html loads
        # print("company " + Job_URL)
        company_name = soup.find('div', {'class': 'css-16nw49e e11nt52q1'})
    company_name = company_name.text[:-4]
    # print(company_name)

    # else:
    #     print("error " + Job_URL)
    #     exit(1)

    # how to extract job title
    job_title = soup.find('div', {'class': 'css-17x2pwl e11nt52q6'})
    while not job_title:
        # print("title " + Job_URL)
        job_title = soup.find('div', {'class': 'css-17x2pwl e11nt52q6'})
    job_title = job_title.text

    # how to extract location
    location = soup.find('div', {'class': 'css-1v5elnn e11nt52q2'})
    while not location:
        # print("location " + Job_URL)
        location = soup.find('div', {'class': 'css-1v5elnn e11nt52q2'})
    location = location.text

    # how to extract salary estimate need to save whether it is a glassdoor or employer estimate.
    # empty string if there is no estimate
    salary_estimate = ''
    if soup.find('span', {'class': 'small css-10zcshf e1v3ed7e1'}):
        salary_estimate = soup.find('span', {'class': 'small css-10zcshf e1v3ed7e1'}).text

        if 'Glassdoor' in salary_estimate:
            salary_estimate = salary_estimate[:-17]

        if 'Employer' in salary_estimate:
            salary_estimate = salary_estimate[14:]

    # how to extract the job description as plain text (needs clean up?))
    job_description = soup.find("div", {'class': 'desc css-58vpdc ecgq1xb5'})
    while not job_description:
        # print("job_desc " + Job_URL)
        job_description = soup.find("div", {'class': 'desc css-58vpdc ecgq1xb5'})

    job_description = str(job_description)
    # print(job_description)

    # create the dictionary with all of the information of that particular job
    job_info = {'Company': company_name, 'Job Title': job_title, 'Location': location, 'Salary': salary_estimate,
                'Description': job_description}

    # add the dictionary of this job to the list of all job, may change it later to directly save to a json file here.
    All_Jobs_list.append(job_info)
    time.sleep(random.randint(0, 3))
    # print("end")

# job_data = json.dumps(All_Jobs_list, indent=2)
print(len(All_Jobs_list))

with open("/Users/bkirkham/FinalProject/jobs_data.json", "w", newline="\r\n") as out:
    json.dump(All_Jobs_list, out, indent=2, ensure_ascii=False)
    # out.write(job_data)


# js = soup.find_all('script')
# with open('js.txt','w') as f:
#     f.write(soup.prettify())

# href = soup.find_all('a', href=True)

# nextbtn = soup.find_all("button", {"class": "nextButton css-1hq9k8 e13qs2071"})


# for a in href:
#     print(a['href'])
