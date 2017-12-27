## 3. 성적이 90점 이상이면 A, 80점 이상이면 B, 70점 이상이면 C, 
##    나머지는 F 
## - 성적을 입력 받아 학점을 알려주시오.
## - 입력값은 "이름|성적" 으로 입력 받음 ex. "민종현|89" 출력: B

grade = input("성적을 입력하세요. 형식은 이름|성적")

score = int(grade.split("|")[1])

if score >90:
	print("A")
elif score >80:
	print("B")
elif score >70:
	print("C")
else:
	print("F")