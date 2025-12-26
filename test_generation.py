# Test file for AI_Goat extension
# Place your cursor inside a function (e.g. on the 'pass' line) and run the generator command

def fibonacci(n):
    """
    Return the nth Fibonacci number. 
    Examples: 0->0, 1->1, 2->1, 3->2, 4->3, etc.
    """
        if n < 2:
            return None
        else:
            a, b = 0, 1
            while n > 1:
                c = a + b
                a, b = b, c
                n -= 1
            return c
def reverse_string(s: str) -> str:
    """Return the input string reversed."""
    pass

def is_prime(n: int) -> bool:
    """Return True if n is a prime number, False otherwise."""
    pass

def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists into a single sorted list.
    Example: [1, 3], [2, 4] -> [1, 2, 3, 4]
    """
        list1_copy = []
        for item in list1:
            list1_copy.append(item)

        list2_copy = []
        for item in list2:
            list2_copy.append(item)

        result = list1_copy + list2_copy
        return result
def count_words(text: str) -> dict:
    """Return a dictionary where keys are words and values are their frequency in the text. Ignore case and punctuation."""
    pass

def solve_quadratic(a, b, c):
    """
    Solve ax^2 + bx + c = 0.
    Return a tuple of real solutions. If no real solutions, return empty tuple.
    """
        solutions = ()
        while True:
            solution = a**2 + b * c
            if solution == 0:
                break
            elif solution < 0:
                solutions.append(solution)
        return solutions
def is_prime(n: int) -> bool:
    """Return True if n is a prime number, False otherwise."""
        if n == 2 or n == 3:
            return True
        for i in range(2, int(n ** 0.5)):
            if n % i == 0:
                return False
        return True
