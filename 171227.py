# 구구단을 gugu.txt에 저장하세요

# f = open("gugu.txt","w")
# for i in range(2,10):
# 	for j in range(1,10):
# 		data = str(i*j)+" "
# 		f.write(data)
# 	f.write("\n")
# f.close()


# 물건, 가격을 5개 입력 받아서 product.txt에 저장하고,
# 파일 읽어와서, 물건 가격 총 합 출력하는 프로그램 작성
# f = open("product.txt","w")
# for i in range(5):
# 	data = input("물건 가격을 입력하세요. 구분자 ,")
# 	f.write(data+"\n")
# f.close()

# f = open("product.txt","r")
# sum = 0

# lines = f.readlines()
# for line in lines:
# 	sum = sum + int(line.split(",")[1])
# print(sum)

# 1.profile.txt 파일을 만들어서, 이름, 나이를 입력 받아서 쓰기
# 2.만들어진 파일 읽어와서 print 하기
# file lego, input()
# f = open("profile.txt","w")
# name = input("Name?")
# age = input("Age?")
# f.write(name+"\n")
# f.write(age)
# f.close()

# f = open("profile.txt","r")
# d = f.read()
# print (d)


# close 하기 귀찮아서 만든거
# with open("aa.txt","a") as f:
# 	for i in range(10):
# 		data = "%d lines \n" % (i+100)
# 		f.write(data)	

# 옵션 : r, w, a
# f = open("aa.txt","a")
# for i in range(10):
# 	data = "%d lines \n" % (i+100)
# 	f.write(data)
# f.close()

# f = open("aa.txt","r")
# data = f.read() #파일 전체 문자열을 읽는 메소드
# print (data)
# f.close()

# f = open("aa.txt","r")
# lines = f.readlines() #모든 줄을 배열로 읽어들여
# for line in lines:
# 	print (line)
# f.close()

# f = open("aa.txt","r")

# while True:
# 	line = f.readline()
# 	if not line: break
# 	print (line)
# f.close()

# f = open("aa.txt","r")
# result = f.readline() #라인을 읽고 포인트를 다음 줄 내려
# print (result)
# f.close()

# f = open("aa.txt","w")
# for i in range(10):
# 	data = "%d lines \n" % i
# 	f.write(data)
# f.close()




# # 1. 1000 미만의 자연수 중에 3,5의 배수의 총합을 
# # 구하시오.
# def 함수():
# 	sum=0
# 	for i in range(1000):
# 		if i % 3 == 0 or i%5 == 0:
# 			sum = sum + i
# 	return sum

# print(함수())

# # print(sum(list([i for i in range(1000) if i%3==0 or i%5 ==0])))

# # 2. 1차원 점이 주어졌을때 가장 거리가 짧은 쌍을 
# # 출력하시오.
# # 입력 : [1,3,4,8,13,17,20] -> 출력 : [3,4]

# def short(list):
# 	result = []
# 	for i in range(len(list)-1):
# 		result.append(list[i+1]-list[i])
# 	index = result.index(min(result))
# 	return (list[index],list[index+1])

# print (short([1,3,4,8,13,17,20]))


# print ("a","b","c")
# print ("a""b""c")
# print ("a"+"b"+"c")


# <함수 연습>
# 3. def 계산기(choice, *args) 계산기 함수
# choice : sum, min, mul, div 4가지 지원
# def 계산기(choice, *args):
# 	if choice == "sum":
# 		result = 0
# 		for i in args:
# 			result = result + i
# 	elif choice == "min":
# 		result = 0
# 		for i in args:
# 			result = result - i
# 	elif choice == "mul":
# 		result = 1
# 		for i in args:
# 			result = result * i
# 	elif choice == "div":
# 		result = 0
# 		for i in args:
# 			result = result / i
# 	return result

# print(계산기("sum",1,2,3,4,5))
# print(계산기("min",1,2,3,4,5))
# print(계산기("mul",1,2,3,4,5))
# print(계산기("div",1,2,3,4,5))



# 4. def gugu(x)
# x단을 출력하시오.
# def 구구단(x):
# 	for i in range(1,10):
# 		print (x*i, end=" ")

# 구구단(2)
# 구구단(7)


# 1. 점수를 입력받아서, 90점 이상이면 A
# 80점 이상이면 B, 아니면 F 를 출력하는 
# 함수를 작성하시오.
# def grade(score):
# 	result = ""
# 	if score > 90:
# 		result = "A"
# 	elif score > 80:
# 		result = "B"
# 	else:
# 		result = "F"
# 	return result

# print (grade(95))
# print (grade(15))

# 2. 숫자 리스트를 입력 받아서, 최댓값을
# 출력하는 함수를 작성하시오.
# def max_list(list):
# 	result = 0
# 	for i in list:
# 		if i > result:
# 			result = i
# 	return result

# print (max_list([1,2,3,40,5]))
# print (max_list([20,3,4,5]))

# 변수의 효력 범위
# a = 1
# def bb(a):
# 	a=a+1
# 	return a
# a = bb(a)
# print (a)

# 값을 주지 않아도 디폴트 값이 있는 함수
# def say_myself(name,old,man=True):
# 	print("나의 이름은 %s 입니다." % name)
# 	print("나의 나이는 %s 입니다." % old)
# 	if man:
# 		print ("남자입니다.")
# 	else:
# 		print ("여자입니다.")

# say_myself("Min","30")
# say_myself("Min","30",False)

# def say(name):
# 	if name == "ko":
# 		return
# 	print (name)

# say("ko")
# say("ka")

# def sum(a,b):
# 	return a+b, a*b

# l,r = sum(1,2)
# print (l)
# print (r)

# # 입력인자를 아는 경우 + 모르는 경우
# def sum_mul(choice, *args):
# 	if choice == 'sum':
# 		result = 0
# 		for i in args:
# 			result = result + i
# 	elif choice == 'mul':
# 		result = 1
# 		for i in args:
# 			result = result * i
# 	return result

# print (sum_mul('sum',1,2,3,4,5,6))
# print (sum_mul('mul',1,4,5,6))



# def sum( *args ):
# 	sum = 0
# 	for i in args:
# 		sum = sum + i
# 	return sum

# print (sum(1,2,3,4,5))
# print (sum(1,2,3))
# print (sum(1,2,3,2,3,54,6))

# 입력 : 숫자 리스트 
# 출력 : 리스트 내에 element 들의 평균을 반환
# 함수내에서 for 문 사용
# def say_list(num_list):
# 	sum = 0
# 	for i in num_list:
# 		sum = sum + i
# 	return sum/len(num_list)

# print (say_list([1,2,3]))



# # 1. 입력 출력 ㅇ
# def sum(a,b):
# 	result = a+b
# 	return result

# # 2. 입력 출력 X
# def say():
# 	print ("hi")

# # 3. 입력 ㅇ출력x 
# def say_a(a):
# 	print (a)

# # 4. 입력 x 출력o 
# def say_100():
# 	return 100

# print (sum(1,2))
# a = say()
# print (a)
# b=say_a("hihihihihis")
# print (b)
# c = say_100()
# print (c)