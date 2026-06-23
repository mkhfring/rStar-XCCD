n = int(input())
A = [int(x) for x in input().split()]

count = 0
for i in range(n):
    minj = i
    for j in range(i,n):
        if (A[j] < A[minj]):
            minj = j

    if (minj != i):
        A[i],A[minj] = A[minj],A[i]
        count += 1

for k in range(n-1):
    print(str(A[k])+" ",end = "")
print(str(A[n-1]))
print(count)

