# input [15,20,10,23,56,75,43,23,43]
# output Input내의 최댓값 출력
# 이용 X : sort,[-1], max
# 이용 ㅇ : for, if 문

i = [15,20,10,23,56,75,43,23,43]
print(max(i))

maxLL = 0
for k in i:
	if k > maxLL:
		maxLL = k
print(maxLL)

# 구구단 한줄 코딩
# result = []
# # result = [num * 3 for num in a if num % 2==0]
# # 위 포맷 이용해서.
# result = [x*y for x in range(2,10) for y in range(1,10)]
# print (result)




a = [1,2,3,4]
result = []
# result = [num *3 for num in a]

# print (result)

# for i in range(10):
# 	print (i)

# for i in range(1,11):
# 	print (i)

# for , range 이용해서 구구단 출력
# 2 4 6 8 10 12 14 16 18
# 3 6 9 12 15 18 21 24 27
# for i in range(2,10):
# 	for j in range(1,10):
# 		print (i*j, end=" ")
# 	print ('')

