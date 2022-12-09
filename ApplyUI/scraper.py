import random

import requests
import time
from bs4 import BeautifulSoup


def glass_scraper(Job_URL):
    # use request and beutifulsoup to get the response text.
    print('actually made it to the scrapper')
    response = requests.get(Job_URL)

    soup = BeautifulSoup(response.text, features="lxml")

    # how to extract company name negative indexing to remove the star rating,
    # couldn't figure out how to just remove the spans...
    #                               css-16nw49e e11nt52q1
    company_name = soup.find('div', {'class': 'css-16nw49e e11nt52q1'})
    while not company_name:
        # print("here " + Job_URL)
        company_name = soup.find('div', {'class': 'css-16nw49e e11nt52q1'})
    company_name = company_name.text[:-4]
    # print(company_name)

    # else:
    #     print("error " + Job_URL)
    #     exit(1)

    # how to extract job title
    job_title = soup.find('div', {'class': 'css-17x2pwl e11nt52q6'})
    while not job_title:
        job_title = soup.find('div', {'class': 'css-17x2pwl e11nt52q6'})
    job_title = job_title.text

    # how to extract location
    location = soup.find('div', {'class': 'css-1v5elnn e11nt52q2'})
    while not location:
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
        job_description = soup.find("div", {'class': 'desc css-58vpdc ecgq1xb5'})
    job_description = str(job_description)

    # create the dictionary with all of the information of that particular job
    job_info = {'Job_URL': Job_URL, 'Company': company_name, 'Job Title': job_title, 'Location': location,
                'Salary': salary_estimate,
                'Description': job_description}

    # add the dictionary of this job to the list of all job, may change it later to directly save to a json file here.
    # time.sleep(random.randint(1,3))

    return job_info
