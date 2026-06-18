#!/usr/bin/env python3
"""
GradusRAG 知识库文档批量生成脚本

从零生成 60 篇专业领域文档，每篇 15000+ 字符。
调用 mimo-v2.5-pro API 自动生成。

用法：
  python scripts/generate_documents.py              # 全量生成
  python scripts/generate_documents.py --domain AI   # 只生成 AI 领域
  python scripts/generate_documents.py --resume       # 断点续跑
  python scripts/generate_documents.py --doc 01       # 只生成指定文档
"""

import os
import sys
import json
import time
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "expand_settings.yaml"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
PROGRESS_FILE = PROJECT_ROOT / "results" / "generate_progress.json"
LOG_FILE = PROJECT_ROOT / "results" / "generate_log.txt"

TARGET_CHARS = 15000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 60 篇文档定义（编号、文件名、主题、领域、关键词）
# ============================================================
DOCUMENTS = [
    # === AI/ML (01-10) ===
    {"num": "01", "file": "01_机器学习基础概念与算法分类.md", "topic": "机器学习基础概念与算法分类",
     "domain": "AI", "keywords": "监督学习,无监督学习,线性回归,决策树,SVM,随机森林,梯度提升,交叉验证,偏差方差权衡,特征工程"},
    {"num": "02", "file": "02_深度学习与神经网络架构.md", "topic": "深度学习与神经网络架构",
     "domain": "AI", "keywords": "神经网络,反向传播,卷积神经网络,循环神经网络,LSTM,Transformer,残差连接,批归一化,激活函数,梯度消失"},
    {"num": "03", "file": "03_自然语言处理核心技术.md", "topic": "自然语言处理核心技术",
     "domain": "AI", "keywords": "分词,词性标注,命名实体识别,Word2Vec,BERT,GPT,文本分类,情感分析,机器翻译,信息抽取"},
    {"num": "04", "file": "04_计算机视觉与图像识别.md", "topic": "计算机视觉与图像识别",
     "domain": "AI", "keywords": "图像分类,目标检测,YOLO,语义分割,GAN,扩散模型,ViT,ResNet,图像生成,三维视觉"},
    {"num": "05", "file": "05_强化学习原理与应用.md", "topic": "强化学习原理与应用",
     "domain": "AI", "keywords": "马尔可夫决策过程,Q-Learning,策略梯度,PPO,DQN,Actor-Critic,AlphaGo,多智能体,RLHF,模型预测控制"},
    {"num": "06", "file": "06_大语言模型与Transformer架构.md", "topic": "大语言模型与Transformer架构",
     "domain": "AI", "keywords": "自注意力机制,多头注意力,位置编码,GPT,BERT,LLaMA,指令微调,RLHF,涌现能力,思维链推理"},
    {"num": "07", "file": "07_检索增强生成技术.md", "topic": "检索增强生成技术(RAG)",
     "domain": "AI", "keywords": "稠密检索,稀疏检索,BM25,向量数据库,知识图谱检索,重排序,RRF融合,Self-RAG,上下文学习,幻觉缓解"},
    {"num": "08", "file": "08_联邦学习与隐私计算.md", "topic": "联邦学习与隐私计算",
     "domain": "AI", "keywords": "横向联邦,纵向联邦,联邦平均,差分隐私,安全多方计算,同态加密,可信执行环境,数据孤岛,模型聚合,隐私保护"},
    {"num": "09", "file": "09_知识图谱构建与应用.md", "topic": "知识图谱构建与应用",
     "domain": "AI", "keywords": "实体抽取,关系抽取,知识表示,图数据库,Neo4j,图神经网络,知识推理,知识融合,本体建模,SPARQL"},
    {"num": "10", "file": "10_AI伦理与可解释性.md", "topic": "AI伦理与可解释性",
     "domain": "AI", "keywords": "算法公平性,偏见检测,模型可解释性,SHAP,LIME,注意力可视化,AI安全,对齐问题,负责任AI,监管框架"},

    # === 医学 (11-20) ===
    {"num": "11", "file": "11_人体解剖学基础.md", "topic": "人体解剖学基础",
     "domain": "医学", "keywords": "细胞组织,骨骼系统,肌肉系统,消化系统,呼吸系统,循环系统,神经系统,泌尿系统,内分泌系统,免疫系统"},
    {"num": "12", "file": "12_常见疾病诊断与治疗.md", "topic": "常见疾病诊断与治疗",
     "domain": "医学", "keywords": "高血压,糖尿病,冠心病,肺炎,胃炎,贫血,甲状腺疾病,骨质疏松,抑郁症,诊疗规范"},
    {"num": "13", "file": "13_临床检验与影像诊断.md", "topic": "临床检验与影像诊断",
     "domain": "医学", "keywords": "血常规,尿常规,生化检查,肿瘤标志物,X光,CT,MRI,超声,PET-CT,病理检查"},
    {"num": "14", "file": "14_药理学基础与合理用药.md", "topic": "药理学基础与合理用药",
     "domain": "医学", "keywords": "药物代谢动力学,药物效应动力学,抗生素,降压药,降糖药,抗肿瘤药,药物相互作用,不良反应,个体化用药,药物警戒"},
    {"num": "15", "file": "15_中医基础理论.md", "topic": "中医基础理论",
     "domain": "医学", "keywords": "阴阳五行,脏腑学说,经络学说,气血津液,病因病机,辨证论治,四诊八纲,中药方剂,针灸推拿,中西医结合"},
    {"num": "16", "file": "16_流行病学与公共卫生.md", "topic": "流行病学与公共卫生",
     "domain": "医学", "keywords": "疾病分布,病因推断,队列研究,病例对照,随机对照试验,传染病防控,疫苗接种,慢性病管理,健康促进,卫生应急"},
    {"num": "17", "file": "17_外科学基础与常见手术.md", "topic": "外科学基础与常见手术",
     "domain": "医学", "keywords": "无菌术,麻醉,创伤处理,普外科手术,骨科手术,心胸外科,微创手术,腹腔镜,围手术期管理,术后并发症"},
    {"num": "18", "file": "18_内科学常见疾病.md", "topic": "内科学常见疾病",
     "domain": "医学", "keywords": "心血管疾病,呼吸系统疾病,消化系统疾病,肾脏疾病,血液病,内分泌疾病,风湿免疫病,感染性疾病,老年医学,多学科会诊"},
    {"num": "19", "file": "19_儿科学与儿童保健.md", "topic": "儿科学与儿童保健",
     "domain": "医学", "keywords": "新生儿疾病,小儿生长发育,儿童营养,预防接种,小儿肺炎,小儿腹泻,先天性心脏病,儿童神经系统疾病,儿童心理发育,母乳喂养"},
    {"num": "20", "file": "20_急诊医学与急救处理.md", "topic": "急诊医学与急救处理",
     "domain": "医学", "keywords": "心肺复苏,创伤急救,中毒处理,休克救治,急性胸痛,急性脑卒中,多发伤,急腹症,灾难医学,院前急救"},

    # === 教育 (21-30) ===
    {"num": "21", "file": "21_教育心理学基础理论.md", "topic": "教育心理学基础理论",
     "domain": "教育", "keywords": "行为主义,认知主义,建构主义,学习动机,认知发展,最近发展区,多元智力,自我效能感,归因理论,学习迁移"},
    {"num": "22", "file": "22_教学设计与课程开发.md", "topic": "教学设计与课程开发",
     "domain": "教育", "keywords": "ADDIE模型,布鲁姆教学目标,翻转课堂,项目式学习,混合式教学,课程标准,教学评价,逆向设计,差异化教学,教学策略"},
    {"num": "23", "file": "23_教育评价方法与技术.md", "topic": "教育评价方法与技术",
     "domain": "教育", "keywords": "形成性评价,终结性评价,标准化测试,表现性评价,档案袋评价,教育测量,信度效度,项目分析,课堂观察,学习分析"},
    {"num": "24", "file": "24_信息技术与教育融合.md", "topic": "信息技术与教育融合",
     "domain": "教育", "keywords": "智慧教育,在线学习平台,教育大数据,人工智能教育应用,虚拟现实教学,自适应学习,数字素养,教育信息化,MOOC,学习管理系统"},
    {"num": "25", "file": "25_特殊教育与融合教育.md", "topic": "特殊教育与融合教育",
     "domain": "教育", "keywords": "特殊儿童分类,个别化教育计划,融合教育,学习障碍,自闭症教育,智力障碍,听觉障碍,视觉障碍,辅助技术,支持服务体系"},
    {"num": "26", "file": "26_教育管理与学校领导.md", "topic": "教育管理与学校领导",
     "domain": "教育", "keywords": "教育行政管理,学校组织管理,教师专业发展,教育质量管理,校园文化建设,课程领导力,教育治理,校本管理,教育督导,家校合作"},
    {"num": "27", "file": "27_比较教育学.md", "topic": "比较教育学",
     "domain": "教育", "keywords": "比较教育方法论,美国教育,英国教育,芬兰教育,日本教育,中国教育改革,PISA测评,教育国际化,教育公平,教育体系比较"},
    {"num": "28", "file": "28_教育研究方法论.md", "topic": "教育研究方法论",
     "domain": "教育", "keywords": "定量研究,定性研究,混合方法研究,实验设计,问卷调查,访谈法,案例研究,行动研究,扎根理论,教育统计分析"},
    {"num": "29", "file": "29_学习科学与认知发展.md", "topic": "学习科学与认知发展",
     "domain": "教育", "keywords": "认知负荷理论,工作记忆,元认知,自我调节学习,协作学习,情境认知,具身认知,脑科学与教育,注意力,知识建构"},
    {"num": "30", "file": "30_教育政策与改革.md", "topic": "教育政策与改革",
     "domain": "教育", "keywords": "教育政策分析,新课程改革,高考改革,双减政策,教育公平,素质教育,职业教育改革,学前教育,高等教育大众化,教育立法"},

    # === 法律 (31-40) ===
    {"num": "31", "file": "31_法学基础理论与法理学.md", "topic": "法学基础理论与法理学",
     "domain": "法律", "keywords": "法的概念与特征,法律规则,法律原则,法律关系,法律责任,法律推理,法律解释,法治原理,法的价值,法律渊源"},
    {"num": "32", "file": "32_宪法与行政法.md", "topic": "宪法与行政法",
     "domain": "法律", "keywords": "宪法基本原则,公民基本权利,国家机构,行政行为,行政处罚,行政许可,行政复议,行政诉讼,政府信息公开,依法行政"},
    {"num": "33", "file": "33_民法总则与民事权利.md", "topic": "民法总则与民事权利",
     "domain": "法律", "keywords": "民事主体,民事法律行为,代理制度,诉讼时效,物权,债权,人格权,婚姻家庭,继承法,侵权责任"},
    {"num": "34", "file": "34_刑法总论与犯罪构成.md", "topic": "刑法总论与犯罪构成",
     "domain": "法律", "keywords": "犯罪构成四要件,犯罪主体,犯罪主观方面,正当防卫,紧急避险,犯罪未遂,共同犯罪,刑罚种类,量刑制度,刑事政策"},
    {"num": "35", "file": "35_合同法与商事法律.md", "topic": "合同法与商事法律",
     "domain": "法律", "keywords": "合同订立,合同效力,合同履行,合同解除,违约责任,买卖合同,公司法,合伙企业法,票据法,破产法"},
    {"num": "36", "file": "36_知识产权法.md", "topic": "知识产权法",
     "domain": "法律", "keywords": "专利法,商标法,著作权法,商业秘密,知识产权保护,专利申请,商标注册,著作权侵权,技术合同,国际知识产权公约"},
    {"num": "37", "file": "37_劳动法与社会保障法.md", "topic": "劳动法与社会保障法",
     "domain": "法律", "keywords": "劳动合同,工资制度,工时制度,劳动争议,社会保险,养老保险,医疗保险,工伤保险,失业保险,劳动仲裁"},
    {"num": "38", "file": "38_环境与资源保护法.md", "topic": "环境与资源保护法",
     "domain": "法律", "keywords": "环境保护法,污染防治,环境影响评价,碳排放交易,自然资源保护,水污染防治,大气污染防治,固体废物管理,环境公益诉讼,生态补偿"},
    {"num": "39", "file": "39_国际法与国际关系.md", "topic": "国际法与国际关系",
     "domain": "法律", "keywords": "国际法基本原则,国际条约,联合国体系,国际人权法,国际经济法,WTO规则,国际争端解决,国际人道法,外交关系法,海洋法"},
    {"num": "40", "file": "40_诉讼法与司法制度.md", "topic": "诉讼法与司法制度",
     "domain": "法律", "keywords": "民事诉讼,刑事诉讼,行政诉讼,证据制度,审判程序,执行程序,仲裁制度,调解制度,司法改革,法律援助"},

    # === 金融 (41-50) ===
    {"num": "41", "file": "41_金融学基础理论.md", "topic": "金融学基础理论",
     "domain": "金融", "keywords": "货币与信用,利率理论,金融市场,金融机构,货币政策,资产定价,投资组合理论,金融风险管理,资本结构,公司金融"},
    {"num": "42", "file": "42_银行与信贷业务.md", "topic": "银行与信贷业务",
     "domain": "金融", "keywords": "商业银行经营,存款业务,贷款业务,信用评估,风险管理,资本充足率,巴塞尔协议,银行监管,信贷审批,不良贷款"},
    {"num": "43", "file": "43_证券市场与投资分析.md", "topic": "证券市场与投资分析",
     "domain": "金融", "keywords": "股票市场,债券市场,基金投资,技术分析,基本面分析,投资组合理论,CAPM模型,有效市场假说,行为金融学,量化投资"},
    {"num": "44", "file": "44_保险学原理与实务.md", "topic": "保险学原理与实务",
     "domain": "金融", "keywords": "保险基本原则,人寿保险,财产保险,健康保险,保险精算,再保险,保险监管,偿付能力,保险科技,保险资金运用"},
    {"num": "45", "file": "45_公司金融与财务管理.md", "topic": "公司金融与财务管理",
     "domain": "金融", "keywords": "财务报表分析,资本预算,资本成本,股利政策,并购重组,企业估值,财务杠杆,营运资金管理,财务风险,内部控制"},
    {"num": "46", "file": "46_风险管理与内部控制.md", "topic": "风险管理与内部控制",
     "domain": "金融", "keywords": "风险识别,风险评估,市场风险,信用风险,操作风险,流动性风险,COSO框架,风险对冲,压力测试,风险文化建设"},
    {"num": "47", "file": "47_金融科技与数字化转型.md", "topic": "金融科技与数字化转型",
     "domain": "金融", "keywords": "区块链金融,数字货币,智能投顾,移动支付,大数据风控,开放银行,监管科技,数字银行,金融科技监管,云计算金融"},
    {"num": "48", "file": "48_绿色金融与可持续发展.md", "topic": "绿色金融与可持续发展",
     "domain": "金融", "keywords": "绿色信贷,绿色债券,ESG投资,碳金融,可持续发展目标,气候风险,环境信息披露,绿色基金,转型金融,碳中和"},
    {"num": "49", "file": "49_国际金融与外汇市场.md", "topic": "国际金融与外汇市场",
     "domain": "金融", "keywords": "外汇市场,汇率决定理论,国际收支,国际资本流动,跨境投资,人民币国际化,外汇储备,国际金融危机,全球金融治理,贸易融资"},
    {"num": "50", "file": "50_金融监管与合规.md", "topic": "金融监管与合规",
     "domain": "金融", "keywords": "金融监管体系,银行监管,证券监管,保险监管,反洗钱,投资者保护,金融消费者权益,监管科技,合规管理,系统性风险"},

    # === 跨域 (51-60) ===
    {"num": "51", "file": "51_AI在医疗诊断中的应用.md", "topic": "AI在医疗诊断中的应用",
     "domain": "跨域", "keywords": "医学影像AI,病理AI,辅助诊断系统,药物研发AI,电子病历分析,远程医疗,可穿戴设备,临床决策支持,医疗机器人,精准医疗"},
    {"num": "52", "file": "52_教育技术中的机器学习应用.md", "topic": "教育技术中的机器学习应用",
     "domain": "跨域", "keywords": "自适应学习系统,智能辅导系统,学习分析,知识追踪,自动评分,教育推荐系统,学情预警,教育数据挖掘,个性化学习路径,AI教师助手"},
    {"num": "53", "file": "53_金融科技中的法律监管.md", "topic": "金融科技中的法律监管",
     "domain": "跨域", "keywords": "数字货币监管,区块链法律问题,智能合约法律效力,数据隐私保护,算法监管,跨境支付监管,P2P网贷监管,金融消费者保护,沙盒监管,监管科技法律"},
    {"num": "54", "file": "54_医学教育中的模拟与虚拟现实.md", "topic": "医学教育中的模拟与虚拟现实",
     "domain": "跨域", "keywords": "医学模拟教学,虚拟手术训练,标准化病人,临床技能训练,VR解剖教学,AR辅助手术,远程医学教育,医学模拟中心,胜任力导向教育,混合现实教学"},
    {"num": "55", "file": "55_金融风险中的法律合规.md", "topic": "金融风险中的法律合规",
     "domain": "跨域", "keywords": "金融风险法律防控,合规风险管理,金融犯罪防范,反洗钱法律,内幕交易,市场操纵,金融消费者权益保护,破产清算,金融仲裁,跨境金融监管合作"},
    {"num": "56", "file": "56_人工智能在法律领域的应用.md", "topic": "人工智能在法律领域的应用",
     "domain": "跨域", "keywords": "智能法律检索,合同审查AI,法律文书生成,智能司法,案件预测,法律知识图谱,在线纠纷解决,法律援助AI,电子证据分析,AI法官助手"},
    {"num": "57", "file": "57_教育心理学与消费者行为.md", "topic": "教育心理学与消费者行为",
     "domain": "跨域", "keywords": "学习动机与消费动机,认知偏差,决策心理,行为经济学教育应用,营销心理学,消费者决策过程,品牌认知,社会影响,数字消费行为,金融素养教育"},
    {"num": "58", "file": "58_医疗数据隐私与信息安全.md", "topic": "医疗数据隐私与信息安全",
     "domain": "跨域", "keywords": "医疗数据分类,患者隐私保护,电子健康档案安全,医疗数据共享,去标识化技术,联邦学习医疗应用,网络安全等级保护,医疗物联网安全,数据泄露应急,健康医疗大数据"},
    {"num": "59", "file": "59_行为经济学与金融教育.md", "topic": "行为经济学与金融教育",
     "domain": "跨域", "keywords": "有限理性,启发式偏差,前景理论,心理账户,过度自信,羊群效应,金融素养,投资者教育,行为干预,助推理论"},
    {"num": "60", "file": "60_AI伦理与教育公平.md", "topic": "AI伦理与教育公平",
     "domain": "跨域", "keywords": "算法偏见与教育公平,数字鸿沟,AI对教育就业的影响,智能教育伦理,数据驱动教育决策,教育AI监管,技术赋能弱势群体,教育公平评估,终身学习,STEM教育公平"},
]


# ============================================================
# LLM 调用（复用 expand 脚本的 LLMCaller）
# ============================================================
class LLMCaller:
    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.model = llm_cfg.get("model", "mimo-v2.5-pro")
        self.api_key = llm_cfg.get("api_key", "")
        self.base_url = llm_cfg.get("base_url", "")
        self.temperature = llm_cfg.get("temperature", 0.5)
        self.max_tokens = llm_cfg.get("max_tokens", 32768)

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError("请在配置文件中配置有效的 api_key")

        from openai import OpenAI
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)
        logger.info(f"LLM: {self.model} @ {self.base_url}")

    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content.strip()

        # 截断自动续写
        if response.choices[0].finish_reason == "length":
            logger.warning("输出被截断，自动续写...")
            cont = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "请继续上面的内容，从断开处接着写，不要重复已有内容。"},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content += "\n\n" + cont.choices[0].message.content.strip()

        return content


SYSTEM_PROMPT = """你是一位资深的学科专家和教材编写者。请撰写一篇关于指定主题的专业文档。

要求：
1. 内容丰富、有深度，每个知识点都要展开详细解释
2. 至少包含 6-8 个章节，每章节 2-4 个子节
3. 包含核心概念定义、原理机制、实际案例、具体数据、历史发展、前沿趋势
4. 语言风格：专业教材级别，准确严谨、通俗易懂
5. 使用 Markdown 格式（## 章节，### 子节）
6. 目标字数：{target} 字
7. 语言：中文
8. 不要用"本文将介绍"等过渡语，直接进入内容
"""


def make_prompt(doc: dict, target: int) -> str:
    return f"""请撰写一篇关于"{doc['topic']}"的专业文档。

领域：{doc['domain']}
建议覆盖的关键词：{doc['keywords']}

目标字数：{target} 字

请直接输出完整的 Markdown 文档。"""


# ============================================================
# 进度管理
# ============================================================
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}, "started_at": datetime.now().isoformat()}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 主逻辑
# ============================================================
def generate_document(llm: LLMCaller, doc: dict, target: int, dry_run: bool = False) -> dict:
    fname = doc["file"]
    fpath = DOCS_DIR / fname

    if dry_run:
        logger.info(f"[{fname}] 将生成 {target} 字")
        return {"file": fname, "status": "dry_run", "target": target}

    logger.info(f"[{fname}] 生成中...")
    t0 = time.monotonic()

    try:
        content = llm.call(
            SYSTEM_PROMPT.format(target=target),
            make_prompt(doc, target),
        )
        elapsed = time.monotonic() - t0

        if not content.endswith("\n"):
            content += "\n"

        fpath.write_text(content, encoding="utf-8")
        actual = len(content)
        logger.info(f"[{fname}] 完成: {actual} 字 ({elapsed:.1f}秒)")
        return {"file": fname, "chars": actual, "seconds": round(elapsed, 1), "status": "done"}

    except Exception as e:
        logger.error(f"[{fname}] 失败: {e}")
        return {"file": fname, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="GradusRAG 知识库文档批量生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析不实际生成")
    parser.add_argument("--doc", type=str, default="", help="只生成指定文档（如 '01' 或 '01,02'）")
    parser.add_argument("--resume", action="store_true", help="跳过已生成的文档")
    parser.add_argument("--target", type=int, default=TARGET_CHARS, help=f"目标字数（默认 {TARGET_CHARS}）")
    parser.add_argument("--domain", type=str, default="", help="只生成指定领域（AI, 医学, 教育, 法律, 金融, 跨域）")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化 LLM
    try:
        llm = LLMCaller(config)
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        sys.exit(1)

    # 确保输出目录存在
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 过滤文档
    docs = DOCUMENTS
    if args.domain:
        docs = [d for d in docs if d["domain"] == args.domain]
        logger.info(f"仅生成 {args.domain} 领域: {len(docs)} 篇")
    if args.doc:
        nums = [n.strip().zfill(2) for n in args.doc.split(",")]
        docs = [d for d in docs if d["num"] in nums]
        logger.info(f"仅生成指定文档: {[d['file'] for d in docs]}")

    # 进度
    progress = load_progress() if args.resume else {"completed": {}, "started_at": datetime.now().isoformat()}
    completed = progress.get("completed", {})

    # 执行
    results = []
    success = skipped = failed = 0

    logger.info(f"{'=' * 60}")
    logger.info(f"开始生成 {len(docs)} 篇文档, 目标 {args.target} 字/篇")
    logger.info(f"{'=' * 60}")

    for i, doc in enumerate(docs):
        fname = doc["file"]

        if args.resume and fname in completed and completed[fname].get("status") == "done":
            logger.info(f"[{i+1}/{len(docs)}] {fname} 已完成，跳过")
            skipped += 1
            continue

        result = generate_document(llm, doc, args.target, args.dry_run)
        results.append(result)

        if result["status"] == "done":
            success += 1
            completed[fname] = result
            progress["completed"] = completed
            save_progress(progress)
        elif result["status"] == "skip":
            skipped += 1
        else:
            failed += 1

        if i < len(docs) - 1 and not args.dry_run:
            time.sleep(1)

    # 汇总
    logger.info(f"\n{'=' * 60}")
    logger.info(f"生成完成! 成功: {success}, 跳过: {skipped}, 失败: {failed}")
    logger.info(f"{'=' * 60}")

    summary = {
        "total": len(docs), "success": success, "skipped": skipped, "failed": failed,
        "target_chars": args.target, "results": results, "timestamp": datetime.now().isoformat(),
    }
    summary_path = PROJECT_ROOT / "results" / "generate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
