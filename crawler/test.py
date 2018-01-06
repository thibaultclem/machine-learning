import platform
from bs4 import BeautifulSoup
from selenium import webdriver

PHANTOMJS_PATH = './phantomjs'
browser = webdriver.PhantomJS(PHANTOMJS_PATH)
#browser = webdriver.FireFox()
browser.get('https://www.betbrain.com/football/england/premier-league/#/matches/')

//*[@id="app"]/div/section/section/main/div[3]/div[2]/div[2]/div[1]/ul/li/West Ham United
0                Chelsea
0      Huddersfield Town
0        Manchester City
0    Newcastle United FC
0          Stoke City FC
0           Swansea City
0         Southampton FC
0                Everton
0      Tottenham Hotspur