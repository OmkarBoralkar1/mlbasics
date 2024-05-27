import numpy as np

g = int(input('Enter the number of matrices: '))

rows = []
columns = []

for i in range(g):
    r = int(input("Enter the number of rows for matrix {}: ".format(i + 1)))
    c = int(input("Enter the number of columns for matrix {}: ".format(i + 1)))
    rows.append(r)
    columns.append(c)

# Taking input for matrices
matrices = []
for i in range(g):
    print("\nEnter elements for matrix {} ({} x {}):".format(i + 1, rows[i], columns[i]))
    matrix = []
    for j in range(rows[i]):
        while True:
            row_input = input("Enter row {} ({} values separated by spaces): ".format(j + 1, columns[i]))
            row_values = [int(x) for x in row_input.split()]
            if len(row_values) == columns[i]:
                matrix.append(row_values)
                break
            else:
                print("Error: The number of values in the row must be equal to the number of columns.")

    matrices.append(matrix)

# Converting the matrix input to numpy arrays
matrix_arrays = [np.array(matrix, dtype=int) for matrix in matrices]

# Printing the matrices
print("\nMatrices:")
for i, matrix_array in enumerate(matrix_arrays):
    print("Matrix {}:".format(i + 1))
    print(matrix_array)

# Performing operations based on user input
operation = input("\nEnter the operation (add, subtract, multiply): ").lower()

if operation == 'add':
    max_rows = max(rows)
    max_columns = max(columns)
    result = np.zeros((max_rows, max_columns), dtype=int)
    for matrix_array, r, c in zip(matrix_arrays, rows, columns):
        result[:r, :c] += matrix_array
elif operation == 'subtract':
    max_rows = max(rows)
    max_columns = max(columns)
    result = np.zeros((max_rows, max_columns), dtype=int)
    result[:rows[0], :columns[0]] = matrix_arrays[0]
    for matrix_array, r, c in zip(matrix_arrays[1:], rows[1:], columns[1:]):
        result[:r, :c] -= matrix_array
elif operation == 'multiply':
    # Check if the number of columns in the first matrix equals the number of rows in the second matrix
    if all(columns[i] == rows[i + 1] for i in range(g - 1)):
        result = matrix_arrays[0].copy()
        for matrix_array in matrix_arrays[1:]:
            result = np.dot(result, matrix_array)
    else:
        print("Error: Number of columns in the first matrix must equal the number of rows in the second matrix.")
        result = None
else:
    print("Invalid operation. Please enter 'add', 'subtract', or 'multiply'.")

# Printing the result
if operation in ['add', 'subtract', 'multiply'] and result is not None:
    print("\nResult of {}:".format(operation.capitalize()))
    print(result)
