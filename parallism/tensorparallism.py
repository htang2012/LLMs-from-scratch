import torch

A = torch.tensor([ [1,2,3,4,5,6],
                   [1,2,3,4,5,6],
                   [1,2,3,4,5,6]]
                )
B = torch.tensor([ [1,2,3,4],
                   [1,2,3,4],
                   [1,2,3,4],
                   [1,2,3,4],
                   [1,2,3,4],
                   [1,2,3,4]
                   ])
                     
print(A@B)

##column split B
B1 = B[:,:2]
B2 = B[:,2:]

print(f'B1={B1}, \n B2={B2}')
print(f'A@B1={A@B1}, \n A@B2={A@B2}')
T1 = A@B1
T2 = A@B2
print(f"A@B={torch.cat((T1, T2), dim=1)}")

# Split A by rows
A1 = A[:2, :]  # First 2 rows
A2 = A[2:, :]  # Last row##raw split
print(f"A1 =\n{A1}\n")
print(f"A2 =\n{A2}\n")

# Compute partial results
T1 = A1 @ B
T2 = A2 @ B

print(f"A1 @ B =\n{T1}\n")
print(f"A2 @ B =\n{T2}\n")

# Combine results vertically (along rows)
final_result = torch.cat((T1, T2), dim=0)
print(f"A @ B (reconstructed) =\n{final_result}\n")

