from fractions import Fraction
from fractions import gcd
from collections import deque
import operator
import functools
import copy


def lcm(a, b):
	return abs(a*b) / gcd(a, b)


def lcm_list(l):
	res = l[0]
	for i in range(1, len(l)):
		res = lcm(l[i], res)
	return res


def convert_to_common(l):
	lcm = lcm_list([x.denominator for x in l])
	res = [x.numerator * lcm/x.denominator for x in l]
	res.append(lcm)
	return res


def transpose(matrix):
	return list(map(list, zip(*matrix)))

def get_identity(dim):
	mtx = [[0 for x in range(dim)] for y in range(dim)]
	for i in range(len(mtx)):
		mtx[i][i] = 1			
	return mtx


def normalize(row, num):
	return [Fraction(x.numerator*num.denominator, x.denominator*num.numerator) for x in row]


def inverse(m):
	matrix = copy.deepcopy(m)
	# Store the operations in here
	inv = get_identity(len(matrix))

	# beg stands for current row and column number.
	for beg in range(len(matrix)):
		# Find row that has non zero in its 'beg' column
		for i in range(beg, len(matrix)):
			if matrix[i][beg] != 0:
				# Swap this column to beg position
				matrix[beg], matrix[i] = matrix[i], matrix[beg]
				inv[beg], inv[i] = inv[i], inv[beg]
				i = beg
				# Make it so that the first number in row is one.
				inv[beg] = normalize(inv[beg], matrix[beg][beg])
				matrix[beg] = normalize(matrix[beg], matrix[beg][beg])
				for j in range(beg, len(matrix)):
					if j != i and matrix[j][beg] != 0:
						term = Fraction(matrix[i][beg].denominator * matrix[j][beg].numerator, matrix[i][beg].numerator * matrix[j][beg].denominator)
						matrix[j] = [x - y * term for x, y in zip(matrix[j], matrix[i])]
						inv[j]    = [x - y * term for x, y in zip(inv[j], inv[i])]
				break
	
	# Now eliminate other numbers
	for dg in range(len(matrix) - 1, -1, -1):
		for j in range(dg - 1, -1, -1):
			term = matrix[j][dg]
			for k in range(len(matrix)):
				inv[j][k] -= inv[dg][k]*term
				matrix[j][k] -= matrix[dg][k]*matrix[j][dg]

	return inv
		

def is_terminal(row):
	return sum(row) == 0


def swap_col(matrix, x, y):
	for i in range(len(matrix)):
		matrix[i][x], matrix[i][y] = matrix[i][y], matrix[i][x]
	return matrix


def print_matrix(matrix):
	for row in matrix:
		print(list(map(str, row)))
	print("\n")


def transform(matrix):
	non_terminal = []
	terminal = []
	indices = []
	# Split the matrix into terminal and non terminal states.
	# Also remember how the index from the original matrix changed,
	# because we need to update the columns as well.
	for i in range(len(matrix)):	
		if sum(matrix[i]) == 0:
			# Where the column was and where column will be now
			indices.append((i, len(terminal)))
			terminal.append( matrix[i] )
	for i in range(len(matrix)):
		if sum(matrix[i]) != 0:
			indices.append((i, len(terminal) + len(non_terminal)))
			non_terminal.append( matrix[i] )

	# Update columns
	for matrix in [terminal, non_terminal]:
		for row in matrix:
			new_row = []
			for i in range(len(row)):
				new_row.append(row[indices[i][0]])
			row[:] = new_row

	# Transform the matrix into fractions
	for row in non_terminal:
		denom = sum(row)
		row[:] = [Fraction(0) if x == 0 else Fraction(x, denom) for x in row]
	
	return (non_terminal, terminal, len(terminal))


def matrix_mult(m1, m2):
	res = [[0 for x in range(len(m2[0]))] for y in range(len(m1))]
	for i in range(len(m1)):
		for j in range(len(m2[0])):
			for k in range(len(m2)):
				res[i][j] += m1[i][k] * m2[k][j]
	return res


def substract_matrix(m1, m2):
	res = []
	for r1, r2 in zip(m1, m2):
		res.append( [x - y for x, y in zip(r1, r2)] )
	return res


def solution(matrix):
	not_terminal, terminal, t_cnt = transform(matrix)
	if sum(matrix[0]) == 0:
		res = [0] * (len(terminal) - 1)
		res.insert(0, 1)
		res.append(1)
		return res
	
	R = [row[:len(terminal)] for row in not_terminal]
	Q = [row[len(terminal):] for row in not_terminal]

	P = inverse(substract_matrix(get_identity(len(Q)), Q))
	res_mtx = matrix_mult(P, R)
	return convert_to_common(res_mtx[0]) 


matrix_example = [
	[0, 1, 0, 0, 0, 1], 
	[4, 0, 0, 3, 2, 0], 
	[0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0],
]
assert solution(matrix_example) == [0, 3, 2, 9, 14]
assert solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) == [7, 6, 8, 21]
assert solution([[0,0,0], [0,0,0], [0,0,0]]) == [1,0,0,1]
assert solution([[0]]) == [1,1]
assert solution([[0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],         
			[0, 0, 0, 0, 0, 0],
		]) == [0, 1, 1, 3, 5]
		
assert solution([
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]) == [0, 1, 1]
assert (
    solution([
        [0, 2, 1, 0, 0],
        [0, 0, 0, 3, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]) == [7, 6, 8, 21]
)
 
assert (
    solution([
        [0, 1, 0, 0, 0, 1],
        [4, 0, 0, 3, 2, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]) == [0, 3, 2, 9, 14]
)
 
assert (
    solution([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 6, 0, 0, 0],
        [7, 8, 9, 1, 0, 0],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]) == [1, 2, 3]
)
assert (
    solution([
        [0]
    ]) == [1, 1]
)
 
assert (
    solution([
        [0, 0, 12, 0, 15, 0, 0, 0, 1, 8],
        [0, 0, 60, 0, 0, 7, 13, 0, 0, 0],
        [0, 15, 0, 8, 7, 0, 0, 1, 9, 0],
        [23, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [37, 35, 0, 0, 0, 0, 3, 21, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) == [1, 2, 3, 4, 5, 15]
)
 
assert (
    solution([
        [0, 7, 0, 17, 0, 1, 0, 5, 0, 2],
        [0, 0, 29, 0, 28, 0, 3, 0, 16, 0],
        [0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
        [48, 0, 3, 0, 0, 0, 17, 0, 0, 0],
        [0, 6, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) == [4, 5, 5, 4, 2, 20]
)
 
assert (
    solution([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) == [1, 1, 1, 1, 1, 5]
)
 
assert ( solution([
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]))  == [2, 1, 1, 1, 1, 6]
 
assert (
    solution([
        [0, 86, 61, 189, 0, 18, 12, 33, 66, 39],
        [0, 0, 2, 0, 0, 1, 0, 0, 0, 0],
        [15, 187, 0, 0, 18, 23, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) == [6, 44, 4, 11, 22, 13, 100]
)
 
assert (
    solution([
        [0, 0, 0, 0, 3, 5, 0, 0, 0, 2],
        [0, 0, 4, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 4, 4, 0, 0, 0, 1, 1],
        [13, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 1, 8, 7, 0, 0, 0, 1, 3, 0],
        [1, 7, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) == [1, 1, 1, 2, 5]
)