from transformers import AutoTokenizer, AutoModel

pretrained_model_name_or_path = '/Users/zhangpeng/bigmodel/Qwen-7B-Chat'

tokenize = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                         trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True).half().to('mps')

# 加载到内存
model = model.eval()

system_input = """
你是一位优秀的数据分析助手,你能对输入文本进行预处理及体征提取并理解文本内容。请你精确识别出文本中包含的实体类型，请你不要遗漏实体类型，如果你遗漏实体类型，你将受到严厉的惩罚。请你以指定的格式返回结果（必须按照指定的格式进行输出），指定的格式为[{"个人电话号码":["13889871342","13889871342","13889871342"]},{"个人姓名": ["张三","王五","李四","张三","刘七"]},{"IP地址":["127.0.0.1"]},...]。（注意：单个key后面的value也需要是数组的形式表示)）请你对输出结果不能进行总结，请你不要胡编乱造，请你只能以指定的格式进行输出，否则你将受到惩罚。请你对未预定义的实体类型在输出结果中一定不能输出，如果输出你将受到惩罚。请你严格按照预定义的实体类别进行识别，请你确保不会自行创建定义中不存在的实体类别，并确保没有任何遗漏或错误。如果你未严格按照要求进行工作，你将会受到严重的处罚，以下是预定义的实体类型：

###
个人姓名：例如张三、李四、王五、赵六。

生日：例如1990年1月1日、1985年12月25日、1995年10月10日、2000年8月8日。

性别：例如男、女、其他、未公开。

民族：例如汉族、蒙古族、回族、藏族。

国籍：例如中国、美国、英国、日本。

家庭关系：例如父亲、母亲、配偶、子女。

住址：例如北京市朝阳区某街道1号、上海市浦东新区某小区A栋、广州市天河区某路B座、深圳市南山区某花园C栋。

个人电话号码：个人电话号码一定是以阿拉伯数字开头，如果不是以阿拉伯数字开头的一定不是个人电话号码。例如13812345678、13923456789、13634567890、15045678901。

邮箱地址：例如zhangsan@example.com、lisi@example.com、wangwu@example.com、zhaoliu@example.com。

个人信息主体账号：例如user12345、member67890、accountABC123、vipXYZ456。

IP地址：例如192.168.0.1、10.1.1.2、2001:db8::ff00:42:8329、172.16.10.100。

个人职业：例如软件工程师、医生、教师、律师。

职位：例如产品经理、项目经理、财务经理、人力资源总监。

工作单位：例如阿里巴巴集团、腾讯科技有限公司、百度在线网络技术(北京)有限公司、华为技术有限公司。

学历：例如本科、硕士研究生、博士研究生、大专。

学位：例如文学学士、理学硕士、工学博士、法学学士。

硬件序列号：例如Dell Laptop - SN: ABC1234567890、iPhone 13 Pro Max - SN: DEF0987654321、Samsung Galaxy S22 Ultra - SN: GHI23456789012、Sony PlayStation 5 - SN: JKL34567890123。

设备MAC地址：例如00:14:22:01:23:45、34:56:78:9A:BC:DE、AB:CD:EF:01:23:45、FE:DC:BA:98:76:54。

经纬度：例如39.9042° N、116.4074° E、31.2304° N、 121.4737° E、23.1291° N、113.2644° E、22.5431° N、114.0573° E。

身份证号码：由固定长度的数字和字符组成。前6位数字代表户籍所在地的行政区划码，一般为省市区的代码；接下来的8位数字代表出生年月日，其中前4位表示年份，后两位表示月份，最后两位表示日期；之后的3位数字是顺序码，用于区分同一地区、同一日期出生的人，一般为从001到999的顺序号；最后一位字符是校验码，用于验证身份证号码的合法性。

军官证号码：例如GJ字第12345678号、GJ字第98765432号、GJ字第34567890号、GJ字第21098765号。

护照号码：例如P123456789、P987654321、P345678901、P210987654。

驾驶证号码：例如11X123456789012、C12345678901234、E98765432101234、B34567890123456。

社保卡号码：例如110101234567890123456789、310105987654321098765432、440305876543210123456789、510105123456789098765432。

居住证号码：例如BJ11010120220000001、SH31010520211234567、GD4403052020ABCDEFG、CD510105201912345678。

婚史：例如未婚、已婚、离异、丧偶。

宗教信仰：例如佛教徒、基督教徒、伊斯兰教徒。

性取向：例如异性恋、同性恋、双性恋、泛性恋。
###

用户:张鹏，已婚

"""
responses, history = model.chat(tokenize, system_input, history=[])
print(responses)

while True:
    _input = input("用户:")
    responses, history = model.chat(tokenize, _input, history=[])
    print("system:" + responses)
