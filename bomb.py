def solution(x, y):
    res = str(solve(int(x), int(y)))
    return res if res != "inf" else 'impossible'

def solve(x, y):
	if x == 1 and y == 1:
		return 0
	if x < 1 or y < 1 or x == y:
		return float("inf")

	x, y = max(x, y), min(x, y)
	coeff = (x // y) - (1 if x % y == 0 else 0) 
	return solve(x - coeff * y, y) + coeff



assert solution('4', '7') == "4"
assert solution('2', '1') == "1"
assert solution('1', '1') == "0"
assert solution('2', '4') == "impossible"
