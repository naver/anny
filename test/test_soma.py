import unittest

import anny


class TestSomaRigVertexCount(unittest.TestCase):

    def test_soma_rig_soma_topology_vertex_count_matches_anny_rig(self):
        anny_model = anny.Anny(rig="anny", topology="soma")
        soma_model = anny.Anny(rig="soma", topology="soma")
        self.assertEqual(soma_model.template_vertices.shape,
                         anny_model.template_vertices.shape)

    def test_soma_rig_default_topology_vertex_count_matches_anny_rig(self):
        anny_model = anny.Anny(rig="anny", topology="anny")
        soma_model = anny.Anny(rig="soma", topology="anny")
        self.assertEqual(soma_model.template_vertices.shape,
                         anny_model.template_vertices.shape)

if __name__ == "__main__":
    unittest.main()
