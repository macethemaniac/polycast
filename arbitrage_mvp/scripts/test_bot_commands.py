import sys, asyncio, json
sys.path.insert(0, 'arbitrage_mvp/src')
from types import SimpleNamespace
import bot

class AsyncMsg:
    def __init__(self):
        self.texts = []
    async def reply_text(self, text, parse_mode=None):
        self.texts.append(text)
        return self
    async def edit_text(self, text, parse_mode=None):
        self.texts.append(text)
        return self

class MockUpdate(SimpleNamespace):
    def __init__(self, chat_id=123, args=None):
        self.message = AsyncMsg()
        self.effective_chat = SimpleNamespace(id=chat_id)
        self._args = args or []

class MockContext(SimpleNamespace):
    def __init__(self, args=None):
        self.args = args or []
        self.job_queue = None
        self.bot = None

async def run_all():
    results = {}
    # start_command
    u = MockUpdate()
    c = MockContext()
    await bot.start_command(u, c)
    results['start'] = u.message.texts

    # help
    u = MockUpdate()
    c = MockContext()
    await bot.help_command(u, c)
    results['help'] = u.message.texts

    # scan (may call external services)
    u = MockUpdate()
    c = MockContext()
    await bot.scan_command(u, c)
    results['scan'] = u.message.texts

    # price without args
    u = MockUpdate()
    c = MockContext(args=[])
    await bot.price_command(u, c)
    results['price_no_args'] = u.message.texts

    # price with pair
    u = MockUpdate()
    c = MockContext(args=['BTC/USDT'])
    await bot.price_command(u, c)
    results['price_with_args'] = u.message.texts

    # polyarb
    u = MockUpdate()
    c = MockContext()
    await bot.polyarb_command(u, c)
    results['polyarb'] = u.message.texts

    # crossarb default
    u = MockUpdate()
    c = MockContext()
    await bot.crossarb_command(u, c)
    results['crossarb_default'] = u.message.texts

    # crossarb with param
    u = MockUpdate()
    c = MockContext(args=['0.2'])
    await bot.crossarb_command(u, c)
    results['crossarb_param'] = u.message.texts

    # alerts status
    u = MockUpdate(chat_id=8404888863)
    c = MockContext()
    await bot.alerts_command(u, c)
    results['alerts_status'] = u.message.texts

    # alerts enable
    u = MockUpdate(chat_id=8404888863)
    c = MockContext(args=['enable'])
    await bot.alerts_command(u, c)
    results['alerts_enable'] = u.message.texts

    # alerts status after enable
    u = MockUpdate(chat_id=8404888863)
    c = MockContext()
    await bot.alerts_command(u, c)
    results['alerts_status2'] = u.message.texts

    # alerts disable
    u = MockUpdate(chat_id=8404888863)
    c = MockContext(args=['disable'])
    await bot.alerts_command(u, c)
    results['alerts_disable'] = u.message.texts

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    asyncio.run(run_all())
