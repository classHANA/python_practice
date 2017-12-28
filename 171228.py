# 날씨 가져오기 forecastio
# pip install python-forecastio
# https://github.com/ZeevG/python-forecast.io

import forecastio

api_key = "c3c46f2ceb471e5d5bf6cd29b5708bfe"
lat = 37.501354
lng = 127.039763

forecast = forecastio.load_forecast(api_key,lat,lng)
byHour = forecast.hourly()

print (byHour.summary)
print (byHour.icon)

for hourlyData in byHour.data:
	print (str(hourlyData.time)+"="+str(hourlyData.temperature))



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