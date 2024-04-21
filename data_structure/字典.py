# 字典的创建
# 创建方式1
dic11 = {"name":"aaa","age":88}
#创建方式2
lista = dict(name="aaa",age=88,classa="61班")
print(dic11)

# 循环取值 1
for e in dic11:
    print("======")
    print(dic11["name"])
    print("======")

print(dic11["name"])
print("#循环取值2:")
#循环取值2
for i, e in lista.items():
    print(i , e)