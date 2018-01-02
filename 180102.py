# pip install selenium
# https://sites.google.com/a/chromium.org/chromedriver/downloads
# 다운 받아서 python과 같은 경로에 위치
# from selenium import webdriver

# browser = webdriver.Chrome()
# browser.get("http://python.org")

# browser = webdriver.Chrome()
# browser.get("http://python.org")

# # find_elements_by_id,name,class_name
# menus = browser.find_elements_by_css_selector('#top ul.menu li')
# pypi = None
# for m in menus:
# 	if m.text == "PyPI":
# 		pypi = m

# pypi.click()

# time.sleep(2)
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np

# # set graph image size
# fig = plt.figure(figsize=(8, 6), dpi=80)
# ax = fig.gca(projection='3d')

# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# pip install matplotlib
# y = x^2 y=x^3 그래프 그리기
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(8,6),dpi=80)
# xs = np.linspace(0.0,10.0,1000)
# ys = [x * x for x in xs]
# zs = [x * x * x for x in xs]

# plt.plot(xs,ys)
# plt.plot(xs,zs)
# plt.title("A simple graph")
# plt.legend(['y=x^2','y=x^3'], loc='upper left')
# plt.show()

# pip install pyscreenshot
# https://github.com/ponty/pyscreenshot

# import pyscreenshot as ImageGrab

# if __name__ == "__main__":
# 	# im = ImageGrab.grab()
# 	# im.save('screenshot.jpg')
# 	# im.show()

# 	im = ImageGrab.grab(bbox=(10,10,510,510))
# 	im.show()

# Random
# from random import *

# 난수 맞추기 게임
# 1, 100사이의 정수가 랜덤으로 생겨요
# WHILE 돌면서 맞추면 종료, 못맞추면 다시 맞추기
# + 낮으면 낮다고, 높으면 높다고 알려주는 가이드

# n = randint(1,100)
# while True:
# 	ans = input("맞춰봐!,종료 하려면 Q")

# 	if (ans == "Q"):
# 		break

# 	if int(ans) == n:
# 		print ("정답입니다.")
# 	elif int(ans) > n : 
# 		print ("더 낮습니다.")
# 	else:
# 		print ("더 높습니다.")

# i = random()
# print (i)

# i = randint(1,100)
# print (i)

# i = uniform(1.0,36.5)
# print (i)

# o = randrange(0,100,2)
# print (o)

# 이미지 처리 모듈, PIL, OpenCV
# python3 : pillow
# pip install Pillow
# https://pillow.readthedocs.io/en/4.3.x/
# from PIL import Image, ImageFilter, ImageDraw, ImageFont

# im = Image.open('a.jpg')
# print ("image info : " + str(im.info))
# print ("image size : " + str(im.size))
# print ("image format : " + str(im.format))
# # print (im.getpixel(10,10))

# lrim = im.transpose(Image.FLIP_LEFT_RIGHT)
# lrim.save('a_lr.jpg')
# tbim = im.transpose(Image.FLIP_TOP_BOTTOM)
# tbim.save('a_tb.jpg')

# size = (64,64)
# im.thumbnail(size)
# im.save('a_thum.jpg')

# im.rotate(90)
# im.save('a_90.jpg')

# size = (1000,1000)
# im.resize(size)
# im.save('a_1000.jpg')

# cropImg = im.crop((100,100,150,150))
# cropImg.save('a_crop.jpg')

# blurImg = im.filter(ImageFilter.SHARPEN)
# blurImg.save('a_blur.jpg')

#Text 이미지 만들기
# font_size=36
# width=500
# height=100
# back_ground_color=(255,255,255)

# font_color=(0,0,0)
# unicode_text = "python is good"

# im  =  Image.new ( "RGB", (width,height), back_ground_color )
# draw  =  ImageDraw.Draw ( im )
# draw.text ( (10,10), unicode_text, fill=font_color )

# im.save("text.jpg")

# JSON
# import json

# customer = {
# 	'id':1,
# 	'name':'홍길동',
# 	'history':[
# 		{'date':'2018-01-01','item':'Mac'},
# 		{'date':'2018-01-02','item':'iphone'}
# 	]
# }

# jsonString = json.dumps(customer, indent=5)

# print (jsonString)
# print (type(jsonString))

# dict = json.loads(jsonString)
# print (dict['name'])
# for i in dict['history']:
# 	print (i['date'],i['item'])