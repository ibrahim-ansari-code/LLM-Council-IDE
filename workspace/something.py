def print_triangle(height=5):
    """
    Prints a centered isosceles triangle using asterisks.
    
    Args:
        height: The number of rows in the triangle (default: 5)
    """
    for i in range(1, height + 1):
        print(' ' * (height - i) + '*' * (2 * i - 1))

if __name__ == "__main__":
    print_triangle(5)