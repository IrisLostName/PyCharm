from babel.plural import range_list_node

a=5
b=3
print(f"{a}+{b}={a+b}")
print(f"{a}*{b}={a*b}")

my_list= [1,2,3,4,5]
print(my_list)
my_list.append(6)
print(my_list)

for i in range(1,6):
    if i%2==0:
        print(f"{i} is even")
    else:
        print(f"{i} is odd")
