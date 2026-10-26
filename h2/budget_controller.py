class BudgetController:
    def __init__(self, limit):
        self.limit, self.provider_calls, self.interaction_depth = limit, 0, 0
        self.usage = []
        self.terminal = False

    def begin(self):
        if self.terminal or self.provider_calls >= self.limit:
            raise RuntimeError('provider request forbidden after terminal/budget')
        self.provider_calls += 1
        self.usage.append(None)  # attempted request, including failed transport

    def received(self, usage):
        self.interaction_depth += 1  # successfully returned model responses, including stop
        self.usage[-1] = usage

    def tokens(self):
        def total(key):
            values = [u.get(key) if isinstance(u, dict) else None for u in self.usage]
            return sum(values) if all(type(v) is int and v >= 0 for v in values) else None
        incoming, outgoing = total('prompt_tokens'), total('completion_tokens')
        return {'input_tokens': incoming, 'output_tokens': outgoing,
                'total_tokens': incoming + outgoing if incoming is not None and outgoing is not None else None}
