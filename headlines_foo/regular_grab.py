from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time
from selenium.common.exceptions import NoSuchElementException
import json

from apscheduler.schedulers.background import BlockingScheduler
from selenium.webdriver.firefox.options import Options

import datetime
import time

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


options = Options()
options.headless = True

sched = BlockingScheduler()



@sched.scheduled_job('interval', id='nato', minutes=30)
def taker():
    driver = webdriver.Firefox(options=options)

    run_time = time.time()


    print("\n\nStarting iteration at: {0}".format(datetime.datetime.now().isoformat()))
    driver.get('https://www.newsnow.co.uk/h/Sport/Football?type=ln')
    html = driver.page_source

    soup = BeautifulSoup(html)

    aa = soup.find_all('div', {'class': 'split_in'})[0]

    data = {}
    for tag in aa.find_all('a', {'class': 'hll'}):
        # print(tag.text)

        # driver.find_element(By.XPATH, '//button[text()="Some text"]')
        # driver.find_element_by_xpath('//a[@href="'+url+'"]')
        item = driver.find_element(By.XPATH, '//a[@href="' + tag['href'] + '"]')
        driver.execute_script("arguments[0].click();", item)
        # WebDriverWait(driver, 5)
        time.sleep(5)
        c = driver.window_handles[1]
        driver.switch_to.window(c)
        try:
            huge_text = driver.find_element(By.XPATH, "/html/body").text
            data[tag.text] = huge_text
        except NoSuchElementException as e:
            print(e)
        driver.close()

        c = driver.window_handles[0]
        driver.switch_to.window(c)

    with open('./data/data_{0}.p'.format(int(time.time())), 'ab') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    driver.quit()
    del driver
    run_time = time.time() - run_time

    print("Run time: {0} min {1} sec".format(run_time // 60, run_time % 60))

sched.start()

# taker()
