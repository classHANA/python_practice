




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