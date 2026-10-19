from langchain_core.tools import tool

@tool
def query_policy_detail(policy_no: str) -> str:
    """根据保单号查询保单详情，当用户问到具体保单信息时使用此工具。"""
    # 模拟去数据库查询
    mock_data = {
        "P123": "保单号 P123，险种：重疾险，保额：50万，生效日期：2023-01-01。",
        "P456": "保单号 P456，险种：医疗险，保额：200万，生效日期：2024-05-20。"
    }
    return mock_data.get(policy_no.upper(), f"抱歉，未找到保单 {policy_no} 的记录。")