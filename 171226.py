# 1을 더하기, 숫자 2개를 입력 받은다음
# 더하고, 2를 입력받으면 빼요. 3을 입력 받으면
# 종료
prompt = """
1. Add
2. Del
3. Quit
"""

n = 0
while n != 3:
	print (prompt)
	n = int(input("숫자를 입력하세요"))
	if n==1:
		a1 = int(input("첫번째 숫자"))
		a2 = int(input("두번째 숫자"))
		print ("결과는 %d 입니다" %(a1+a2))
	elif n==2:
		a1 = int(input("첫번째 숫자"))
		a2 = int(input("두번째 숫자"))
		print ("결과는 %d 입니다" %(a1-a2))
	else:
		pass