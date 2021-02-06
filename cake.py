
def solution(s):
	size = len(s)
	for slice_size in filter(lambda x : size % x == 0, range(1, size + 1)):
		slices = size // slice_size
		index = zip(range(0, slice_size*slices, slice_size), range(slice_size, slice_size*(slices + 1), slice_size))
		splitted = [s[i:j] for i, j in index]
		if(len(splitted) == splitted.count(splitted[0])):
			return slices

print(solution("abcabcabcabc"))
print(solution("a"))
print(solution("abcd"))
print(solution("aaaa"))
print(solution("abababababbababa"))
print(solution("abcddcba"))
print(solution("ababab"))
