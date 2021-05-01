from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep

text_translated = []

# This script requires an up to date version of Chromedriver to work,
# or may be outdated and not work without being updated, as
# this solution interacts directly with the Google Translate webpage,
# which is subject to changes in structure at any time.

# temporary solution to using the google translate API
# make the the opened window be hidden
def run_translate(string_to_translate, source, target):
    global text_translated
    
    if (source == target):
        print("Source and target languages must be different.")
        exit(1)
    
    options = Options()
    options.headless = True

    print (string_to_translate)
    
    driver = webdriver.Chrome(r'C:\Users\accountname\Desktop\chromedriver_win32\chromedriver.exe')
    sleep(0.01)
    driver.get("http://translate.google.com/#"+source+"/"+target+"/")

    # Element names will need to be updated as Google
    # Translate site itself updates
    input_element  = "//textarea[@id='source']"
    output_element = "//span[@class='tlid-translation translation']"

    input_box = WebDriverWait(driver, 1).until(lambda driver: driver.find_element_by_xpath(input_element))
    input_box.send_keys(string_to_translate)

    sleep(0.01)

    result_element = WebDriverWait(driver, 1).until(lambda driver: driver.find_element_by_xpath(output_element))

    myString = result_element.text.encode('utf-8', 'ignore')
    myString = myString.decode("utf-8")

    driver.close()
    driver.quit()

    text_translated = myString
    return text_translated    

run_translate("This is my red car", "en", "fr")
print ('translated input text to "{0}"'.format(text_translated))
