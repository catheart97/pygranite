from unittest import result
import pygranite as pg
import numpy as np
import unittest

class DummyLoader(pg.WindfieldLoader):
    def __init__(self, size=10, _3D=True):
        pg.WindfieldLoader.__init__(self)
        self._3D = _3D
        if self._3D:
            self.DataU  = np.random.rand(size, size, size)
            self.DataV  = np.random.rand(size, size, size)
            self.DataW  = np.random.rand(size, size, size)
        else:
            self.DataU  = np.random.rand(size, size)
            self.DataV  = np.random.rand(size, size)

    def hasNext(self):
        return True

    def next(self):
        return [ self.DataU, self.DataV ] + ( [self.DataW] if self._3D else [] )

class TestGPUMethods(unittest.TestCase):

    def test_compute_3D(self):

        for size in range(10, 50):
            for _ in range(10):
                size = 10

                loader = DummyLoader(size=size)

                settings = pg.IntegratorSettings()
                settings.Space = pg.Space3D
                settings.MinimumAliveParticles = 1
                settings.InterpolateWindfields = False
                settings.Integrator = pg.ExplicitEuler
                settings.DeltaT = 1
                settings.WindfieldTimeDistance = 1
                settings.MaximumSimulationTime = 1000
                settings.GridScale = [1, 1, 1]
                settings.SaveInterval = 1
                settings.AdditionalCompute = [
                    "u", "v", "w"
                ]

                particle = [
                    np.random.randint(1, size),
                    np.random.randint(1, size),
                    np.random.randint(1, size)
                ]

                start_set = pg.TrajectorySet([ particle, particle ])

                expected = [ 
                    loader.DataU[particle[2]][particle[1]][particle[0]],
                    loader.DataV[particle[2]][particle[1]][particle[0]],
                    loader.DataW[particle[2]][particle[1]][particle[0]]
                ]

                integrator = pg.TrajectoryIntegrator(settings, loader, start_set)
                result_set = integrator.compute()

                got = [
                    result_set.computeInfo(0)[0][0],
                    result_set.computeInfo(1)[0][0],
                    result_set.computeInfo(2)[0][0]
                ]

                self.assertAlmostEqual(expected[0], got[0])
                self.assertAlmostEqual(expected[1], got[1])
                self.assertAlmostEqual(expected[2], got[2])
    
    def test_compute_2D(self):

        for size in range(10, 50):
            for _ in range(10):
                size = 10

                loader = DummyLoader(size=size, _3D=False)

                settings = pg.IntegratorSettings()
                settings.Space = pg.Space2D
                settings.MinimumAliveParticles = 1
                settings.InterpolateWindfields = False
                settings.Integrator = pg.ExplicitEuler
                settings.DeltaT = 1
                settings.WindfieldTimeDistance = 1
                settings.MaximumSimulationTime = 1000
                settings.GridScale = [1, 1]
                settings.SaveInterval = 1
                settings.AdditionalCompute = [
                    "u", "v"
                ]

                particle = [
                    np.random.randint(1, size),
                    np.random.randint(1, size)
                ]

                start_set = pg.TrajectorySet([ particle, particle ])

                expected = [ 
                    loader.DataU[particle[1]][particle[0]],
                    loader.DataV[particle[1]][particle[0]]
                ]

                integrator = pg.TrajectoryIntegrator(settings, loader, start_set)
                result_set = integrator.compute()

                got = [
                    result_set.computeInfo(0)[0][0],
                    result_set.computeInfo(1)[0][0]
                ]

                self.assertAlmostEqual(expected[0], got[0])
                self.assertAlmostEqual(expected[1], got[1])


if __name__ == "__main__":
    print("\n### Running C++ tests...\n") # must be called before python tests
    pg.test_run() # c++ side tests
    print("\n### Running python tests...\n")
    unittest.main() # python side tests
    print()