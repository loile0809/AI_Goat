def sum_sq_v1(fn):
    cases = [(0,0),(1,1),(2,5),(3,14),(5,55),(10,385)]
    return all(fn(n)==exp for n,exp in cases)

def sum_n_v1(fn):
    return all(fn(n)==n*(n+1)//2 for n in [0,1,2,5,10])

def factorial_v1(fn):
    cases = [(0,1),(1,1),(2,2),(3,6),(5,120)]
    return all(fn(n)==exp for n,exp in cases)

def fib_v1(fn):
    cases = [(0,0),(1,1),(2,1),(3,2),(5,5),(7,13)]
    return all(fn(n)==exp for n,exp in cases)

def is_prime_v1(fn):
    primes = [2,3,5,7,11]
    non = [0,1,4,6,8,9,10]
    return all(fn(p) for p in primes) and all(not fn(n) for n in non)

def pal_v1(fn):
    return fn("aba") and fn("racecar") and not fn("abc")

def sort_v1(fn):
    return fn([3,1,2])==[1,2,3] and fn([])==[]

def gcd_v1(fn):
    return fn(12,18)==6 and fn(7,3)==1

def lcm_v1(fn):
    return fn(4,6)==12 and fn(5,3)==15

def count_v1(fn):
    return fn(1,[1,2,1,1])==3 and fn("a","banana")==3

def clamp_v1(fn):
    return fn(5,0,3)==3 and fn(-1,0,3)==0 and fn(2,0,3)==2

def avg_v1(fn):
    return fn([2,4])==3 and fn([1,2,3,4])==2.5

def rev_v1(fn):
    return fn("abc")=="cba" and fn("")==""

def uniq_v1(fn):
    return fn([1,2,1,3,2])==[1,2,3]

def max_v1(fn):
    return fn([1,5,3])==5

def min_v1(fn):
    return fn([1,5,3])==1

def len_v1(fn):
    return fn("abc")==3 and fn("")==0

def even_v1(fn):
    return fn(2) and not fn(3)

def abs_v1(fn):
    return fn(-3)==3 and fn(3)==3

def pow2_v1(fn):
    return fn(0)==1 and fn(3)==8

def c_to_f_v1(fn):
    val = fn(100)
    return abs(val-212)<0.1

def is_leap_v1(fn):
    return fn(2020) is True and fn(2019) is False

def count_vowels_v1(fn):
    return fn("hello")==2 and fn("sky")==0

def last_el_v1(fn):
    return fn([1,2,3])==3 and fn([10])==10

def sqrt_int_v1(fn):
    return fn(4)==2 and fn(9)==3 and fn(10)==3

def xor_v1(fn):
    return fn(True, False) is True and fn(True, True) is False

def prod_v1(fn):
    return fn([1,2,3])==6 and fn([5,5])==25

def remove_digits_v1(fn):
    return fn("abc123def")=="abcdef"

def second_largest_v1(fn):
    return fn([1,5,3,4])==4

def anagram_v1(fn):
    return fn("listen", "silent") is True and fn("hello", "world") is False
