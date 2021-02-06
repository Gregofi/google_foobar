
def xor(n):
	mod = n & 3
	if mod == 0:
		return n
	if mod == 1:
		return 1
	if mod == 2:
		return n + 1
	if mod == 3:
		return 0

def interval(begin, end):
	return xor(begin - 1) ^ xor(end - 1)

def solution(start, length):
	offset = 0
	result = 0
	for i in range(0, length):
		result ^= interval(start, start + length - offset)
		print(result)
		start += length
		offset += 1
	return result

print(solution(17, 4))