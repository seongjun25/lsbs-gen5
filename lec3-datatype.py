x = 15
type(x)

y = 3.14
type(y)

# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]

fruits[0]
fruits[1:]

numbers + numbers
numbers * 3

numbers + fruits + mixed_list

mixed_list[2][1]

len(mixed_list[2])

a = [1, 2, 3]
str(a[2]) + "hi"


numbers[2] = 10
numbers
del numbers[1]
numbers

type(numbers)
numbers.append(200)
numbers.append([1, 4])
numbers

del numbers[-1]
numbers.sort()
numbers

numbers.append([1, 4])
numbers[0] = 20
numbers.sort()

a = (10, 20, 30) # a = 10, 20, 30 과 동일
a

a[1]

b = (2, 4)
a + b
a * 3

d = (2,)
d
d[0]

a[2] = 3

tup_e = (1, 3, "a", [1, 2], (3, 1))
tup_e

# 딕셔너리 생성 예제
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}
print("Person:", person)

person.get('name')
person.keys()


a = {1 : 'hi'}
a[2] = 'b'
a['name'] = 'issac'
a

del a[2]
a

a.get(1)
a[1]
list(a.keys())[1]

list(a.values())[1]

'name' in a
'issac' in a.values()

s1 = set([1, 2, 3, 2, 2, 3])
list(s1)
tuple(s1)

s2 = set([4, 5, 6, 1, 2, 7])

s1 & s2  # 교집합
s1 | s2  # 합집합
s1 - s2
s2 - s1

# p.118 문제 9, 10, 11
a = dict()
a

a['name'] = "python"
a[('a',)] = "python"
a[[1]] =  "python"
a[250] = "python"
a

# Q.10
a = {"A": 90, "B": 80, "C": 70}
a
result = a.pop("B")
result
a

# Q.11 
a = [1,1,1,2,2,3,3,3,4,4,5]
a_set = set(a)
a_set
a = list(a_set)
a