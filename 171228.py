# 엑셀 다루기 
# openpyxl
# pip install openpyxl
# https://openpyxl.readthedocs.io/
# import openpyxl

# wb = openpyxl.load_workbook("score.xlsx")
# ws = wb.active

# for r in ws.rows:
# 	row_index = r[0].row
# 	kor = r[1].value
# 	eng = r[2].value
# 	math = r[3].value
# 	sum = kor+eng+math

# 	ws.cell(row=row_index, column=5).value = sum

# wb.save("score2.xlsx")
# wb.close

# 알고리즘 연습 (http://www.codewars.com)

# webbrowser 포팅
# import webbrowser

# target = ['수지','아이유','양세형','정준하']
# url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query="
# wiki_url = "https://en.wikipedia.org/wiki/"

# for i in target:
# 	target_url = wiki_url + i
# 	webbrowser.open(target_url)

# forecastio + geocoder 연습

# import forecastio
# from geopy.geocoders import Nominatim

# tartget = input("어디가 궁금하세요?")
# geolocator = Nominatim()
# location = geolocator.geocode(tartget)

# api_key = "c3c46f2ceb471e5d5bf6cd29b5708bfe"
# lat = location.latitude
# lng = location.longitude

# forecast = forecastio.load_forecast(api_key,lat,lng)
# byHour = forecast.hourly()

# print (str(location)+"의 날씨는 "+byHour.summary)
# print (byHour.icon)

# for hourlyData in byHour.data:
# 	print (str(hourlyData.time)+"="+str(hourlyData.temperature))


# geocoding 포팅
# pip install geopy
# https://pypi.python.org/pypi/geopy
# from geopy.geocoders import Nominatim

# tartget = input("어디가 궁금하세요?")
# geolocator = Nominatim()
# location = geolocator.geocode(tartget)
# print ("주소 : " + str(location))
# print ("위도 : " + str(location.latitude))
# print ("경도 : " + str(location.longitude))

# [알고리즘 연습]
# 1. factorial algorithm
# def factorial(4) => 4*3*2*1 반환하는 함수
# def factorial(6) => 6*5*4*3*2*1 반환하는 함수

# 1 번 방법
# import math
# print (math.factorial(10))

# # 2 번 방법
# def factorial_yame(n):
# 	l = range(1,n+1)
# 	result = 1
# 	for i in l:
# 		result = result* i
# 	return result
# print (factorial_yame(10))

# # 3 번 방법 (재귀 이용)
# def factorial(n):
# 	if n == 0:
# 		return 1
# 	else:
# 		return n * factorial(n-1)
# print (factorial(10))

# [알고리즘 연습]
# 2. n개의 계단을 오른다. 한번에 1개,2개,3개씩 오를수 있다.
# 계단을 오르는 경우의 수가 몇가지 있는지 구하는 함수를 구하라
# ex. 계단의 갯수가 
# 입력: 3, 출력: 4 _ (1,1,1),(1,2),(2,1)(3)
# 입력: 4, 출력: 7 _ (1,1,1,1)(1,1,2)(1,2,1)(2,1,1)(2,2)(3,1)(1,3)
# 재귀 사용
# def countWay(n):
# 	if n<0:
# 		return 0
# 	elif n == 0:
# 		return 1
# 	else:
# 		return countWay(n-1) + countWay(n-2)+countWay(n-3)
# print (countWay(5))

# [알고리즘 연습]
# 3. 주어진 수 중에 소수가 몇개 인지 출력하는 프로그램
# 입력 : 1000 -> 1000이하의 수 중에 소수가 몇개인지 출력/168
# 키워드 : 에라토스테네스의 체
# def countPrime(n):
# 	if n < 2 : 
# 		return []
# 	s = [0,0] + [1] * (n-1) # 1000개의 요소를 갖는 배열 초기화

# 	for i in range(2,int(n**.5)+1): # 루트 n이하의 요소로 나누
# 		if s[i]:
# 			s[i*2::i] = [0] * ((n-i)//i) # i의 배수 지우기
# 	l = [i for i, v in enumerate(s) if v] # 리스트만들기
# 	return l, len(l)

# l,c = countPrime(1000)
# print (l)
# print (c)


# 날씨 가져오기 forecastio
# pip install python-forecastio
# https://github.com/ZeevG/python-forecast.io

# import forecastio

# api_key = "c3c46f2ceb471e5d5bf6cd29b5708bfe"
# lat = 37.501354
# lng = 127.039763

# forecast = forecastio.load_forecast(api_key,lat,lng)
# byHour = forecast.hourly()

# print (byHour.summary)
# print (byHour.icon)

# for hourlyData in byHour.data:
# 	print (str(hourlyData.time)+"="+str(hourlyData.temperature))



# 환율 변환
# pip install --user currencyconverter
# https://github.com/alexprengere/currencyconverter
# from currency_converter import CurrencyConverter
# from datetime import date

# c = CurrencyConverter()
# print (c.convert(100,'EUR', 'USD'))
# print (c.convert(100,'EUR', 'USD', date=date(2016,6,24)))

# print (c.convert(100,'EUR')) #기본값은 유로
# print (c.convert(100,'USD')) 

# 루비코드와 비교
# require 'eu_central_bank'

# def exchange(from, to)
# 	bank = EuCentralBank.new
# 	bank.update_rates
	
# 	bank.exchange(100, from, to)
# end

# puts "$1 => #{exchange 'USD', 'JPY'}엔"

# numpy 사용법
# import numpy as np

# list1 = [1,2,3,4]
# a = np.array(list1)
# print (a.shape)
# b = np.array([[1,2,3],[4,5,6]])
# print (b.shape)
# print (b[0][0])

# #numpy로 배열 만들기
# c = np.zeros((2,2))
# print (c)
# d = np.ones((3,4))
# print (d)
# e = np.full((2,6),8)
# print (e)
# f = np.eye(3)
# print (f)
# g = np.array(range(20)).reshape((4,5))
# print (g)

# #numpy indexing, 슬라이싱
# lst = [
# 	[1,2,3],
# 	[4,5,6],
# 	[7,8,9]
# ]
# lst_n = np.array(lst)
# a = lst_n[0:2,0:2]
# print (a)
# a = lst_n[1:,1:]
# print (a)

# lst = [
# 	[1,2,3],
# 	[4,5,6],
# 	[7,8,9]
# ]
# a = np.array(lst)
# # s = a[0,2]
# # s1 = a[1,3]
# # print (s1)

# # boolean indexing
# # 1
# boolean = np.array([
# 	[False,True,False],
# 	[True,False,True],
# 	[False,True,False]
# ])

# k = a[boolean]
# print (k)

# # 2
# lst = [
# 	[1,2,3],
# 	[4,5,6],
# 	[7,8,9]
# ]
# a = np.array(lst)
# boolean2 = (a % 2 == 0)
# print (boolean2)
# print (a[boolean2])

# numpy 연산
# a = np.array([1,2,3])
# b = np.array([4,5,6])

# c = a+b
# d = a-b
# e = np.multiply(a,b)
# f = np.divide(a,b)
# print (c)
# print (d)
# print (e)
# print (f)

# 행렬, 벡터 연산
# lst1 = [
# 	[1,2],
# 	[3,4]
# ]

# lst2 = [
# 	[5,6],
# 	[7,8]
# ]

# a = np.array(lst1)
# b = np.array(lst2)
# c = np.dot(a,b)
# print (c)

# a = np.array([
# 	[1,2],
# 	[3,4]
# ])

# print (np.sum(a))
# print (np.prod(a))

# print (np.sum(a, axis=0))
# print (np.sum(a, axis=1))




# Fibonacci Numbers
# def Fibonacci(n):
# 	result = [0,1]

# 	for i in range(2,n+1):
# 		result.append(result[i-1]+result[i-2])
# 	print (result)
# 	return result[-1]

# print (Fibonacci(10)) #55

# Join Operator
# food = ["123","apple","grape","boat"]
# print (''.join(food))
# print ('/'.join(food))
# print (','.join(food))
# print ('\n'.join(food))

# 두 문자열이 서로 anagram 인지 검사하는 알고리즘 작성
# def isAnagram(s1,s2):

# 	a = ''.join(sorted(s1.lower())).strip()
# 	b = ''.join(sorted(s2.lower())).strip()

# 	if a==b:
# 		return True
# 	else:
# 		return False

# print (isAnagram("Listen","slient")) # True
# print (isAnagram("listen","slienk")) # False
# print (isAnagram("apple","eppal")) # True
# print (isAnagram("apple","epzkz")) # False