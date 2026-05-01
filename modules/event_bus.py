"""
to be written:
Event bus for connecting modules without direct imports
"""

class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_name, callback):
        self._subscribers.setdefault(event_name, []).append(callback)

    def publish(self, event_name, data=None):
        for cb in self._subscribers.get(event_name, []):
            cb(data)

bus = EventBus()
