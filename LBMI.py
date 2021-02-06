import math

def fibbo(num):
	n1, n2 = 1, 1
	n, sum = 0, 0
	while num > sum:
		n1, n2 = n2, n1 + n2
		n += 1
		sum += n1
	return n

def solution(total_lambs):
	if(total_lambs == 1):
		return 0
	return fibbo(total_lambs) - int(math.floor(math.log(total_lambs + 1, 2)))

print(solution(143))
print(solution(10))
print("--")
print(solution(8))
print(solution(7))
print("--")
print(solution(1))
