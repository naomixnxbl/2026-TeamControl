"""Generic version-keyed cache primitive."""


class VersionedCache:
    """Stores values keyed against a version tag.

    When `invalidate(new_version)` sees a new tag, the store is cleared
    so the next `get(key, compute)` re-runs `compute()`.
    """

    __slots__ = ("_version", "_store")

    def __init__(self):
        self._version = None
        self._store = {}

    def invalidate(self, new_version):
        if new_version != self._version:
            self._store.clear()
            self._version = new_version
            return True
        return False

    def get(self, key, compute):
        if key not in self._store:
            self._store[key] = compute()
        return self._store[key]

    def put(self, key, value):
        self._store[key] = value

    def has(self, key):
        return key in self._store

    @property
    def version(self):
        return self._version
