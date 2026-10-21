# 30 条黄金测试集
EVAL_DATASET = [
    # ========== 明确答案类 (20 条) ==========
    {"question": "上班时间是什么时候？", "ground_truth": "上午 9:00 至下午 18:00"},
    {"question": "中午休息多久？", "ground_truth": "休息 1 小时"},
    {"question": "请假需要提前多久申请？", "ground_truth": "提前一天"},
    {"question": "请假需要谁审批？", "ground_truth": "直属主管"},
    {"question": "年假满1年有多少天？", "ground_truth": "5天"},
    {"question": "年假满3年有多少天？", "ground_truth": "10天"},
    {"question": "年假满5年有多少天？", "ground_truth": "15天"},
    {"question": "差旅费报销需要哪些材料？", "ground_truth": "发票、行程单、审批单"},
    {"question": "差旅费报销缺发票可以吗？", "ground_truth": "三者缺一不可"},
    {"question": "单笔报销超过5000元需要谁审批？", "ground_truth": "财务总监审批"},
    {"question": "报销申请需要在几天内提交？", "ground_truth": "30天"},
    {"question": "员工请假通过什么方式提交？", "ground_truth": "系统提交"},
    {"question": "本制度适用于哪些人？", "ground_truth": "公司全体员工"},
    {"question": "本制度由哪个部门负责解释？", "ground_truth": "人力资源部"},
    {"question": "本制度什么时候开始施行？", "ground_truth": "发布之日起"},
    {"question": "满3年工龄年假几天？", "ground_truth": "10天"},
    {"question": "报销单超过5000需要财务总监吗？", "ground_truth": "财务总监"},
    {"question": "怎么申请休假？", "ground_truth": "系统提交申请"},
    {"question": "中午休息从几点到几点？", "ground_truth": "中午休息 1 小时"}, # PDF里只写了中午休息1小时
    {"question": "出差报销期限是多久？", "ground_truth": "费用发生后 30 天内"},

    # ========== 拒答/超纲类 (10 条) ==========
    {"question": "公司股票代码是多少？", "ground_truth": "未提及"},
    {"question": "公司有几个员工？", "ground_truth": "未提及"},
    {"question": "公司的年营收是多少？", "ground_truth": "未提及"},
    {"question": "CEO叫什么名字？", "ground_truth": "未提及"},
    {"question": "公司总部在哪里？", "ground_truth": "未提及"},
    {"question": "辞职流程是什么？", "ground_truth": "未提及"},
    {"question": "加班有加班费吗？", "ground_truth": "未提及"},
    {"question": "产假多少天？", "ground_truth": "未提及"},
    {"question": "年终奖怎么算？", "ground_truth": "未提及"},
    {"question": "公司考勤使用什么软件？", "ground_truth": "未提及"}
]