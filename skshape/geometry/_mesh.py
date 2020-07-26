
_MESH_ID_COUNTER = 0

def _get_mesh_id():
    global _MESH_ID_COUNTER
    _MESH_ID_COUNTER += 1
    return _MESH_ID_COUNTER


class Mesh(object):

    def __init__(self):
        self.id = _get_mesh_id()
        self.timestamp = 0

    def has_submeshes(self):
        return False

    def submeshes(self):
        return []

