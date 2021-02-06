def version_compare(x, y):
	xs = x.split('.', 1)
	ys = y.split('.', 1)
	if(int(xs[0]) == int(ys[0])):
		if(len(xs) == 1 and len(ys) == 1):
			return 0
		if(len(xs) == 1):
			return -1
		if(len(ys) == 1):
			return 1
		return version_compare(xs[1], ys[1])
	return int(xs[0]) - int(ys[0])

def solution(l):
	return sorted(l, cmp=version_compare)

print(solution(['1', '1']))
# print(version_compare( "1.0.12", "1.0.2"))