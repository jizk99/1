import zipfile
import os

files = {
    "requirements.txt": """openai>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
""",
    ".env": """OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-2024-05-13
""",
    "config.py": """import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "your-key")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    profit_margin_min = 0.15
    max_discount = 0.8

settings = Settings()
""",
    "models/__init__.py": "",
    "models/schemas.py": """from dataclasses import dataclass
from typing import List

@dataclass
class Product:
    sku: str
    name: str
    our_price: float
    our_stock: int
    platform: str

@dataclass
class CompetitorInfo:
    platform: str
    competitor_name: str
    price: float
    stock: int
    avg_rating: float
    recent_comment_sentiment: float

@dataclass
class MarketReport:
    product_name: str
    summary: str
    recommended_action: str
    confidence: float
""",
    "llm/__init__.py": "",
    "llm/client.py": """import json
from openai import OpenAI
from config import settings

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.llm_model

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] 调用失败: {e}")
            return '{"recommended_action": "hold", "confidence": 0.5, "summary": "LLM调用失败，默认维持现状"}'
""",
    "agents/__init__.py": "",
    "agents/base.py": """class BaseAgent:
    def __init__(self, name: str):
        self.name = name
    def run(self, *args, **kwargs):
        raise NotImplementedError
""",
    "agents/data_collection.py": """import random
from typing import List
from agents.base import BaseAgent
from models.schemas import Product, CompetitorInfo

class DataCollectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataCollectionAgent")

    def fetch_competitor_data(self, product: Product) -> List[CompetitorInfo]:
        print(f"[{self.name}] 正在采集竞品数据 (SKU: {product.sku}) ...")
        competitors = [
            CompetitorInfo("taobao", "竞店A", round(product.our_price * random.uniform(0.85, 1.1), 2),
                           random.randint(0,500), round(random.uniform(4.0,5.0),1),
                           round(random.uniform(-0.2,0.8),2)),
            CompetitorInfo("taobao", "竞店B", round(product.our_price * random.uniform(0.9,1.15), 2),
                           random.randint(0,500), round(random.uniform(4.0,5.0),1),
                           round(random.uniform(-0.2,0.8),2)),
            CompetitorInfo("jd", "京东专营店", round(product.our_price * random.uniform(0.92,1.08), 2),
                           random.randint(0,300), round(random.uniform(4.2,5.0),1),
                           round(random.uniform(-0.1,0.9),2)),
            CompetitorInfo("pdd", "拼多多旗舰", round(product.our_price * random.uniform(0.7,1.0), 2),
                           random.randint(0,800), round(random.uniform(3.8,4.8),1),
                           round(random.uniform(-0.3,0.7),2)),
        ]
        print(f"[{self.name}] 采集完成，获得 {len(competitors)} 条竞品信息。")
        return competitors
""",
    "agents/market_analysis.py": """import json
from typing import List
from agents.base import BaseAgent
from models.schemas import Product, CompetitorInfo, MarketReport
from llm.client import LLMClient

class MarketAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("MarketAnalysisAgent")
        self.llm = LLMClient()

    def analyze(self, product: Product, competitors: List[CompetitorInfo]) -> MarketReport:
        print(f"[{self.name}] 启动 LLM 长链推理分析 ...")
        comp_details = []
        for c in competitors:
            comp_details.append(f"{c.platform}/{c.competitor_name}: 价格{c.price}, 库存{c.stock}, 评分{c.avg_rating}, 情感{c.recent_comment_sentiment}")

        system_prompt = \"\"\"你是一个经验丰富的电商运营策略分析师。你需要根据提供的我方商品数据、竞品数据，进行多步推理（长链推理），并给出行动建议。
推理时请遵循以下步骤：
1. 价格竞争力分析：计算我方与竞品均价差异，判断价格优势/劣势。
2. 库存健康度：结合我方库存与竞品库存、市场情绪，预判滞销或断货风险。
3. 服务质量比较：对比评分和评论情感，评估非价格竞争力。
4. 综合决策：给出明确的动作（hold/reduce_price/increase_price/restock）及置信度(0-1)，并提供简短的解释文本。
请以 JSON 格式输出，包含字段: recommended_action, confidence, summary。\"\"\"

        user_prompt = f\"\"\"
我方商品: {product.name} (SKU: {product.sku})
平台: {product.platform}
当前价格: {product.our_price} 元
当前库存: {product.our_stock} 件

竞品数据 ({len(competitors)} 个):
{chr(10).join(comp_details)}

请开始推理并给出 JSON。
\"\"\"
        response = self.llm.chat(system_prompt, user_prompt)
        try:
            data = json.loads(response)
            report = MarketReport(
                product_name=product.name,
                summary=data.get("summary", ""),
                recommended_action=data.get("recommended_action", "hold"),
                confidence=float(data.get("confidence", 0.6))
            )
        except (json.JSONDecodeError, KeyError):
            report = MarketReport(product.name, "LLM返回格式异常，默认保持", "hold", 0.5)

        print(f"[{self.name}] 分析完成：{report.recommended_action} (置信度 {report.confidence:.0%})")
        return report
""",
    "agents/decision.py": """from typing import Dict, Any
from agents.base import BaseAgent
from models.schemas import Product, MarketReport
from config import settings

class DecisionAgent(BaseAgent):
    def __init__(self):
        super().__init__("DecisionAgent")
        self.profit_margin_min = settings.profit_margin_min
        self.max_discount = settings.max_discount

    def decide(self, product: Product, market_report: MarketReport) -> Dict[str, Any]:
        print(f"[{self.name}] 综合决策中...")
        alternate_action = "restock" if product.our_stock < 50 else "hold"
        action = market_report.recommended_action
        reason = market_report.summary

        if action != alternate_action:
            print(f"[{self.name}] 冲突检测：市场建议 {action}，库存建议 {alternate_action}，启动冲突消解...")
            if product.our_stock < 20 and action == "reduce_price":
                final_action = "restock"
                reason = "库存极低，优先补货避免断货（覆盖降价建议）。" + reason
            elif product.our_stock > 200 and action == "increase_price":
                final_action = "reduce_price"
                reason = "库存过高，提价将恶化滞销，改为小幅降价清仓。" + reason
            else:
                final_action = action
                reason = "采用市场分析建议，库存建议仅作参考。" + reason
        else:
            final_action = action

        new_price = product.our_price
        restock_amount = 0

        if final_action == "reduce_price":
            min_price = product.our_price * (1 - self.profit_margin_min)
            new_price = max(product.our_price * 0.9, min_price)
        elif final_action == "increase_price":
            new_price = round(product.our_price * 1.05, 2)
        elif final_action == "restock":
            restock_amount = max(100, product.our_stock * 2)

        decision = {
            "sku": product.sku,
            "action": final_action,
            "new_price": new_price,
            "restock_amount": restock_amount,
            "reason": reason,
            "market_report_summary": market_report.summary,
            "confidence": market_report.confidence,
        }
        print(f"[{self.name}] 最终决策：{final_action} | 新价格:{decision['new_price']} | 补货:{restock_amount}")
        return decision
""",
    "agents/execution.py": """from typing import Dict, Any
from agents.base import BaseAgent

class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ExecutionAgent")

    def execute(self, decision: Dict[str, Any]) -> bool:
        print(f"[{self.name}] 执行操作中...")
        sku = decision["sku"]
        action = decision["action"]
        if action in ("reduce_price", "increase_price"):
            print(f"  >>> 修改价格: {sku} -> {decision['new_price']} 元")
        if action == "restock":
            print(f"  >>> 创建补货单: {sku} 补货 {decision['restock_amount']} 件")
        if action == "hold":
            print(f"  >>> 维持现状")
        print(f"[{self.name}] 执行完成。")
        return True
""",
    "orchestrator.py": """from agents.data_collection import DataCollectionAgent
from agents.market_analysis import MarketAnalysisAgent
from agents.decision import DecisionAgent
from agents.execution import ExecutionAgent
from models.schemas import Product

class EcommerceAgentOrchestrator:
    def __init__(self):
        self.data_agent = DataCollectionAgent()
        self.analysis_agent = MarketAnalysisAgent()
        self.decision_agent = DecisionAgent()
        self.execution_agent = ExecutionAgent()

    def run_cycle(self, product: Product):
        print("\\n" + "="*60)
        print(f"运营决策循环开始 -> 商品: {product.name}")
        print("="*60)
        competitors = self.data_agent.fetch_competitor_data(product)
        market_report = self.analysis_agent.analyze(product, competitors)
        decision = self.decision_agent.decide(product, market_report)
        success = self.execution_agent.execute(decision)
        print("运营决策循环结束。\\n")
        return {
            "product": product.name,
            "competitor_count": len(competitors),
            "market_summary": market_report.summary,
            "final_action": decision["action"],
            "new_price": decision["new_price"],
            "restock_amount": decision["restock_amount"],
            "confidence": decision["confidence"],
            "executed": success
        }
""",
    "main.py": """from orchestrator import EcommerceAgentOrchestrator
from models.schemas import Product

if __name__ == "__main__":
    product = Product(
        sku="SKU12345",
        name="无线降噪蓝牙耳机",
        our_price=299.0,
        our_stock=180,
        platform="taobao"
    )
    orchestrator = EcommerceAgentOrchestrator()
    result = orchestrator.run_cycle(product)
    print("========== 最终决策摘要 ==========")
    print(f"商品: {result['product']}")
    print(f"分析竞品数: {result['competitor_count']}")
    print(f"市场推理: {result['market_summary']}")
    print(f"执行动作: {result['final_action']}")
    print(f"新价格: {result['new_price']}")
    print(f"补货量: {result['restock_amount']}")
    print(f"置信度: {result['confidence']}")
"""
}

# Create zip file
zip_name = "跨平台电商运营决策Agent系统.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path, content in files.items():
        zipf.writestr(file_path, content.encode('utf-8'))

print(f"✅ 已生成 {zip_name}，解压后按照 README 配置 .env 即可运行。")
