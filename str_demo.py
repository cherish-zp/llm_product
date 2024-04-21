str = "zz"
str2 = 'nick to meet you '

print(str + str2)
### 格式化字符串


aaa = "请相{}账户转账¥{:0,.3f}".format(88888 , 1099992,3333)
print(aaa)

a = input("数字:")
if  a == "1" :
    print("1")
else:
    print("不等于1")