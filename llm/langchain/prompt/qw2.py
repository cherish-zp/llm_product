from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
model_path = "/Users/zhangpeng/bigmodel/Qwen-7B-Chat"

system_input = """
你是一位优秀的数据分析助手，你能对输入文本进行预处理及体征提取并理解文本内容。请你精确识别出文本中包含的实体类型，请你不要遗漏实体类型，如果你遗漏实体类型，你将受到严厉的惩罚。请你以指定的格式返回结果（必须按照指定的格式进行输出），指定的格式为[{"个人电话号码":["13889871342","13889871342","13889871342"]},{"个人姓名": ["张三","王五","李四","张三","刘七"]},{"IP地址":["127.0.0.1"]},...]。请你对输出结果不能进行总结，请你不要胡编乱造，请你只能以指定的格式进行输出，否则你将受到惩罚。请你对未预定义的实体类型在输出结果中一定不能输出，如果输出你将受到惩罚。请你严格按照预定义的实体类别进行识别，请你确保不会自行创建定义中不存在的实体类别，并确保没有任何遗漏或错误。如果你未严格按照要求进行工作，你将会受到严重的处罚，以下是预定义的实体类型：

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
"""


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="mps", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

# 第一轮对话
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》