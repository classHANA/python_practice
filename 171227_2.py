# fizzbuzz
for i in range(1,100):
	if i %3==0:
		if i%5==0:
			print("fizzbuzz")
		print("fizz")
	elif i % 5==0:
		print("buzz")
	else:
		print (i)

# Python Crawling 
#-*-coding:utf-8
# import urllib.request
# from bs4 import BeautifulSoup 

# req = urllib.request.Request("http://finance.naver.com/sise/")
# data = urllib.request.urlopen(req).read()
# bs = BeautifulSoup(data, 'html.parser')

# target = ['KOSPI_now','KOSDAQ_now','KPI200_now']
# for i in target:
#     print (i+" = "+bs.find(id=i).get_text())

# 1. myReverse 함수 만들기
# 'abcd'->'dcba' 출력
# reverse 함수 쓰지 않고.
# pop()
# def myReverse(str):
# 	stack = []
# 	for i in str:
# 		stack.append(i)

# 	result = ""
# 	while len(stack)>0:
# 		result = result + stack.pop()

# 	return result

# def myReverse_2(str):
# 	return str[::-1]

# print (myReverse("abcd"))
# print (myReverse_2("abcd"))

# # 2. 문자열에 포함된 문자가 유일한지 검사하는 프로그램
# # ABCD - pass / ABAD - fail
# def isUniq(str):
# 	if len(str)>256:
# 		return False

# 	hash=[False]*256
# 	for i in str:
# 		if hash[ord(i)] is True:
# 			return False
# 		else:
# 			hash[ord(i)] = True
# 	return True

# print (isUniq("ABAD"))
# print (isUniq("ABCD"))

# # 3. 공백을 '%20'으로 바꿔주는 프로그램을 작성하시오.
# # 입력 : Mr John Smith
# # 출력 : Mr%20John%20Smith
# def change(str):
# 	result = str.strip().replace(" ","%20")
# 	return result
# print (change("Mr John Smith"))

# # 4. 같은 문자 반복횟수를 이용하여 문자열을 압축
# # 입력 : AABBDDD
# # 출력 : A2B2D3

# def compressWord(input):
# 	buffer = None
# 	output = ""
# 	cnt = 1

# 	for ch in input:
# 		if buffer is None:
# 			output = output + ch
# 			buffer = ch
# 		else:
# 			if buffer == ch:
# 				cnt = cnt+1
# 			else:
# 				output = output + str(cnt)
# 				cnt = 1
# 				output = output + ch
# 				buffer = ch
# 	output = output + str(cnt)
# 	if len(output)>len(input): return input
# 	else: return output

# print (compressWord("AABBDDDAAA"))
# print (compressWord("ABCDEF"))

















# # 1. 'apple' 문자열을 'aple'로 바꾸는 코드를 작성하시오.
# a = 'apple'
# print(a[0:2]+a[3:])

# # 2. 'Python is fun'에서 'fun'이라는 문자열을 추출 하시오.
# a = 'Python is fun'
# print (a[10:])
# # 3. [1,2,3,[4,[5,6,7]],8,9]에서 6을 추출 하시오.
# a = [1,2,3,[4,[5,6,7]],8,9]
# print (a[3][1][1])
# # 4. 티셔츠가 10,000원, 스웨터가 30,000원이다
# # 10만원 이하 구매시 5% 할인, 10만원 초과 구매시 15% 할인
# # 구매 수량 입력 받고, 총액 출력하시오.
# t = 10000
# s = 30000
# t_n = int(input("t 몇개?"))
# s_n = int(input("s 몇개?"))
# total = t*t_n + s*s_n
# if total <= 100000:
# 	print ("총액은 %d 입니다." % (total*0.95))
# else:
# 	print ("총액은 %d 입니다." % (total*0.75))
# # 5. 입력받은 수가 홀수 이면 "홀수 입니다." 짝수이면
# # "짝수입니다"를 출력하시오.(소수가 입력되면 MyError 반환)
# # try except 사용
# class MyError(Exception):
# 	def __str__(self):
# 		return "소수는 지원 안함"

# try:
# 	n = int(input("수를 입력하세요."))
# 	if n % 2 == 0 :
# 		print ("짝수입니다.")
# 	else:
# 		print ("홀수입니다.")
# except MyError as e:
# 	print (e)
# # 6. 키, 몸무게를 입력 받아 BMI를 계산하시오.
# # BMI = (몸무게)/(키*키)
# # 20 미만이면 '저체중', 20~24면 '정상', 24~30이면 '약비만',
# # 30 이상이면 '비만'을 출력하시오.
# h = int(input("키 입력 해"))
# w = int(input("몸무게 입력 해"))
# bmi = w/(pow(h,2))
# if bmi < 20:
# 	print ("저체중")
# elif bmi < 24:
# 	print ("정상")
# elif bmi < 30 :
# 	print ("약비만")
# else:
# 	print ("비만")
# # 7. 5개의 정수를 입력 받아 가장 큰 값을 파일에 입력하시오.
# k = input("수 5개 입력해[,로 구분해]")
# kl = k.split(",")
# max = 0
# for i in kl:
# 	i = int(i)
# 	if i>max:
# 		max = i
# f = open("a.txt","w")
# f.write(str(max))
# f.close

# webbrowser
# import webbrowser
# webbrowser.open("http://naver.com")
# webbrowser.open_new("http://naver.com")

# random
# import random
# print (random.random())
# print (random.randint(1,10))

# calendar
# import calendar
# print (calendar.calendar(2017))
# print (calendar.prmonth(2017,12))

# time
# import time

# print (time.strftime('%x',time.localtime(time.time())))
# print (time.strftime('%c',time.localtime(time.time())))

# for i in range(10):
# 	print (i)
# 	time.sleep(1)



# print (max([1,2,3,4,5,6]))
# print (abs(-11))
# # all : 모두 참인지 검사
# print (all([1,2,3,4,5]))
# print (all([1,2,3,0,5]))
# # any : 하나라도 참인지 검사
# print (any([1,2,3,4,5]))
# print (any([0,0,0,0,0]))
# print (any(["",0]))
# print (chr(97))
# print (ord('a'))
# print (dir([1,2,3]))
# print (dir({'1':'a'}))
# a,b = divmod(7,3)
# print (a)
# print (b)
# for i, name in enumerate(["A","B","C","D"]):
# 	print (i,name)
# print (eval('1+2'))
# print (eval("'hello'+'a'"))
# print (eval("divmod(4,3)"))

# def positive(x):
# 	return x > 0

# print (list(filter(positive,[1,-3,5,-8])))
# print (pow(2,4))
# print (list(range(1,10,2)))
# print (list(range(0,-10,-1)))

# print (sorted([3,1,2]))
# print (sorted("zero"))

# # 응용
# print (list(zip([1,2,3],[4,5,6])))
# print (list(zip("hello","asdfg")))

# sum = lambda a,b : a+b
# print(sum(3,4))

# myList = [lambda a,b:a+b, lambda a,b:a*b]
# print (myList[0](3,4))
# print (myList[1](3,4))

# # def two_times(x): 
# # 	return x*2

# # print (list(map(two_times,[1,2,3,4,5])))
# print (list(map(lambda a:a*3,[1,2,3,4,5])))




# 에러 처리
# 에러 메시지 만들기
# class MyError(Exception):
# 	def __str__(self):
# 		return "내가 만든 에러입니다."

# def say_nick(nick):
# 	if nick=="min":
# 		raise MyError()
# 	print (nick)

# try:
# 	say_nick("min")
# except MyError as e:
# 	print (e)



# 강제로 에러를 발생 시키는 경우
# class Bird:
# 	def fly(self):
# 		raise NotImplementedError

# class Eagle(Bird):
# 	def fly(self):
# 		print ("dd")

# eagle = Eagle()
# eagle.fly()





















# try-catch 문(자바의 try-catch)
# try:
# 	a = [1,2]
# 	print (a[3])
# except (IndexError,ZeroDivisionError) as e:
# 	pass


# try:
# 	a = [1,2]
# 	# print (a[3])
# 	4/0
# except IndexError as e:
# 	print (e)
# except ZeroDivisionError as e:
# 	print (e)


# 모듈 (gem 만들어서 import)
# 1. cal_module.py : sum, min, mul, div
# 사칙연산 결과를 확인하는 프로그램을 작성하시오.

# import sys
# sys.path.append("C:\\Users\\likelion\\python\\python_practice\\apple")

# import cal_module
# print (cal_module.sum(1,2))

# import mod2
# print (mod2.PI)
# a = mod2.Math()
# print (a.solv(2))
# print (mod2.sum(mod2.PI,4.4))


# import mod1
# from mod1 import sum

# print (sum(3,4))
# print (safe_sum("1",2))












# N실습 : ame : Test class
# 대소문자와 띄어쓰기가 있는 문장이 있다. 이 문장에 단어가 몇개 있는지
# 구하는 클래스 함수를 만드시오.
# ex. "My name is Hou" -> 4
# class Test:
# 	def count(self,sentence):

		# return len(sentence.split(" "))

		# count = 1
		# for i in sentence:
		# 	if i == " ":
		# 		count = count + 1
		# return count

# pey = Test()
# print (pey.count("My name is Hou"))


# class HousePark:
# 	lastname = "박"

# 	def __init__(self, name):
# 		self.fullname = self.lastname + name

# 	def travel(self, where):
# 		print ("%s, %s 여행을 가다." % (self.fullname, where))

# 	def __add__(self,other):
# 		print ("%s, %s 결혼함"%(self.fullname, other.fullname))

# class HouseKim(HousePark):
# 	lastname = "김"
# 	#오버라이딩
# 	def travel(self,where,day):
# 		print ("%s,%s 여행을 %s에 가다." % (self.fullname, where, day))

# pey = HousePark("호우")
# # pey.travel("러시아")
# pi = HouseKim("하야")
# pey + pi


# pi.travel("미얀마","내일")

# FourCal Class 만들기
# 클래스 함수 : add, mul, min, div
# __init__ : name 받게
# class FourCal:
# 	def __init__(self,name):
# 		self.name = name
# 	def add(self,n1,n2):
# 		result = n1+n2
# 		print ("%s님 %s + %s = %s 입니다." % (self.name,n1,n2,n1+n2))
# 	def min(self,n1,n2):
# 		result = n1-n2
# 		print ("%s님 %s - %s = %s 입니다." % (self.name,n1,n2,n1-n2))
# 	def mul(self,n1,n2):
# 		result = n1*n2
# 		print ("%s님 %s * %s = %s 입니다." % (self.name,n1,n2,n1*n2))
# 	def div(self,n1,n2):
# 		result = n1/n2
# 		print ("%s님 %s / %s = %s 입니다." % (self.name,n1,n2,n1/n2))

# pey = FourCal("호우")
# pey.add(1,2)
# pey.min(1,2)
# pey.mul(1,2)
# pey.div(1,2)

# pey = FourCal("호야우")
# pey.add(1,2)
# pey.min(1,2)
# pey.mul(1,2)
# pey.div(1,2)



# class Service:
# 	secret = "Hello"

# 	def __init__(self,name):
# 		self.name = name

# 	def add(self, num,num2):
# 		result = num + num2
# 		print ("%s님 %s + %s = %s 입니다" % (self.name,num,num2,num+num2))

# pey = Service("홍길동")
# # pey.setname("홍길동")
# pey.add(1,1)

# Class 변수
# class Service:
# 	secret="Hello"

# Service.secret = "Hi"

# a = Service()
# print (a.secret)

# b = Service()
# print (b.secret)

# Class, Module
# 자바에서 Class : 재사용성
# a=Class 
# class Calculator:
# 	#생성자
# 	def __init__(self):
# 		self.result = 0

# 	def add(self, num):
# 		self.result = self.result + num
# 		return self.result

# a = Calculator()
# b = Calculator()

# print (a.add(3))
# print (a.add(5))

# print (b.add(3))
# print (b.add(5))