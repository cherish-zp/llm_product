# 列表
list = ['张三', '李四', '王五']
print(list)
print(list[2])
## 范围取值
list1 = list[0]
print(list1)
## 获取元素索引
print(list.index("李四"))
print("==============")
list.remove("张三")
for e in list:
    print(e, list.index(e))
